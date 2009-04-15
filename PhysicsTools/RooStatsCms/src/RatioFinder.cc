// @(#)root/hist:$Id: RatioFinder.cc,v 1.1.1.1 2009/04/15 08:40:01 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   07/10/2008

#include "assert.h"

#include <iostream>

#include "TMath.h"

#include "RooRealVar.h"

#include "PhysicsTools/RooStatsCms/interface/Rsc.h"
#include "PhysicsTools/RooStatsCms/interface/RatioFinder.h"




/*----------------------------------------------------------------------------*/

/**
Init all the members and perform a check on the input models.
<EM>var_name</EM> is the name of the variable to study, e.g. the ratio or the 
luminiosity. <EM>variables</EM> is the list of vars that is passed internally 
to the single istances of LimitCalculator.
**/

RatioFinder::RatioFinder(const char* name,
                         const char* title,
                         RooAbsPdf* sb_model,
                         RooAbsPdf* b_model,
                         const char* var_name,
                         RooArgList variables,
                         ConstrBlockArray* c_array):
    StatisticalMethod(name,title,true),
    m_sb_model(sb_model),
    m_b_model(b_model),
    m_c_array(c_array),
    m_is_lumi(false),
    m_max_attempts(100),
    m_nbins(100){

    m_variables.add(variables);

    // Locate the ratio Var in the sb model
    m_ratio = static_cast<RooRealVar*>
                         (sb_model->getParameters(m_variables)->find(var_name));

    if (m_ratio == NULL){ // we did not find any ratio in the sb model
        std::cout << "[RatioFinder::RatioFinder] "
                  << "No parameter " << var_name
                  << " found in the sb_model. Aborting..."
                  << std::endl;
        abort();
        }

    // Check if the ratio is the same in both models
    RooRealVar* ratio_b = static_cast<RooRealVar*>
                (b_model->getParameters(RooArgSet(m_variables))->find(var_name));
    if (ratio_b != 0){ // we did not find any ratio in the sb model
        m_is_lumi=true;
        std::cout << "[RatioFinder::RatioFinder] "
                  << "The " << var_name
                  << " variable is present also in the b model."
                  << "Is it a luminosity?"
                  << std::endl;
        }
    }

/*----------------------------------------------------------------------------*/

RatioFinder::~RatioFinder(){
    // Nothing to free!
    }

/*----------------------------------------------------------------------------*/

/**
Find the upper and lower ratios to obtain the given CL. The interpolation 
between the nearest values will be performed in RatioFinderResults.
The technique to get near to the value is a simple bisection one.
When near to the desired CL instead of taking a point in the middle of the two.
**/
RatioFinderResults* RatioFinder::findRatio(unsigned int n_toys,
                                           double init_lower_ratio,
                                           double init_upper_ratio,
                                           double n_sigma,
                                           double CL_level,
                                           double delta_ratios_min,
                                           bool dump_Limit_results){

    /*
    Find the value of -2lnQ corresponding to the number of sigmas.
    This is done only once per scan.
    If the number of sigmas is 0, the median of the -2lnQ distribution in the B
    only hypothesis is taken into account.
    Otherwise the quantiles of the distribution are calculated. Since the 
    quantity comes from an histogram, it is clear that the statistic 
    accumulated and the binning play a role.
    */

    if (is_verbose())
        std::cout << "\nFinding the -2lnQ value for "
                    << n_sigma << " sigma(s)...\n";

    double m2lnQ=0;

    if (not m_is_lumi){

        LimitResults* res = m_get_LimitResults(n_toys);
        LimitPlot* p = res->getPlot("temp","temp",100);
        p->draw();
        TH1F* b_histo = p->getBhisto();

        if (fabs(n_sigma) < 0.0001){
            m2lnQ = Rsc::getMedian(b_histo);
            if (is_verbose())
                std::cout << " -> we take the median, which is located in "
                        << m2lnQ << std::endl;
            }

        else{
            int index=0;
    //         if (n_sigma>0)
    //             index=0;

            if (n_sigma<0){
    //            n_sigma*=-1;
                index=1;
                }

            double CL = TMath::Erf(fabs(n_sigma))*2-1;
            double* d = Rsc::getHistoPvals(b_histo,CL);
            m2lnQ=d[index];
            if (is_verbose())
                std::cout << " -> we take the value located in "
                        << m2lnQ << std::endl;
            delete d;
            }

        delete p;

        res->setM2lnQValue_data(m2lnQ);
        p = res->getPlot("temp","temp",100);

        p->draw();

        TString plot_name(GetName());
        plot_name+="_initialm2lnQdistributions_";
        plot_name+="_n_toys";
        plot_name+=n_toys;
        plot_name+="_n_sigmas";
        plot_name+=n_sigma;
        plot_name+=".png";

        plot_name.ReplaceAll(" ","");

        p->dumpToImage(plot_name.Data());

        delete p;
        delete res;
    }

    // End of the typical -2lnQ calculation

    // Start calculating CLs for the two extremes.


    if (is_verbose())
        std::cout << "\nBuilding distributions for extremes in " 
                  << init_lower_ratio << " and "
                  << init_upper_ratio << std::endl;

    double upper_CL = m_get_CLs(init_lower_ratio,n_toys,m2lnQ);
    double lower_CL = m_get_CLs(init_upper_ratio,n_toys,m2lnQ);

    double lower_ratio = init_lower_ratio;
    double upper_ratio = init_upper_ratio;

    double temp_ratio,temp_CL;
    double upper_weight,lower_weight;

    int attempts=m_max_attempts;

    /*
    Cycle for the intermediate points. Stop if:
        - the m_max_attempts is reached.
        - the two ratios are near "enough".
    */

    std::map <double,double> m_points;

    if (is_verbose()){
        std::cout << "\n\n\n-----------------------------------------------\n"
                    << "Scan situation:\n"
                    << " - Left side (Low Ratio, High CL) = ("
                    << lower_ratio << "," << upper_CL << ")\n"
                    << " - Right side (High Ratio, Low CL) = ("
                    << upper_ratio << "," << lower_CL << ")\n"
                    << "-----------------------------------------------\n\n";
        }

    m_points[upper_ratio]=lower_CL;
    m_points[lower_ratio]=upper_CL;

    bool do_average=true;
    while (attempts!=0 and 
           (upper_ratio-lower_ratio > delta_ratios_min) and
           fabs(upper_CL-lower_CL) > CL_level/10. ){


        // A test slightly redundant for runtime.
        assert(upper_ratio > lower_ratio);

        // Increase the size of the interval if it gets too small
        if (upper_CL < CL_level and lower_CL < CL_level){
            do_average=false;
            //lower_ratio*=0.8;
            temp_ratio=lower_ratio*0.8;
            }
        if (upper_CL > CL_level and lower_CL > CL_level){
            do_average=false;
            //upper_ratio*=1.4;
            temp_ratio=upper_ratio*1.4;
            }
        /*
          Elaborate a new point for the scan:
           - "Near" the value of CL make a weighted average
           - "Far" from the CL value use plain average
        "Near" and "far" are considered wrt CLs values.
        */

        if (fabs(upper_CL-CL_level) < CL_level/5. or
            fabs(lower_CL-CL_level) < CL_level/5.){
            upper_weight = 1./fabs(CL_level-lower_CL);
            lower_weight = 1./fabs(CL_level-upper_CL);

            }
        else
            upper_weight=lower_weight=1.;

        if (do_average)
            temp_ratio=m_weighted_average(upper_ratio,
                                          upper_weight,
                                          lower_ratio,
                                          lower_weight);
        else
            std::cout << "Not averaging!\n";
        // Reassign the new value
        temp_CL = m_get_CLs(temp_ratio,n_toys,m2lnQ);

        if (temp_CL <= CL_level){
            std::cout << "temp_CL <= CL_level\n";
            upper_ratio=temp_ratio;
            //lower_CL = m_get_CLs(upper_ratio,n_toys,m2lnQ);
            lower_CL = temp_CL;//
            }
        else{
            std::cout << "temp_CL > CL_level\n";
            lower_ratio=temp_ratio;
            upper_CL = temp_CL;//m_get_CLs(lower_ratio,n_toys,m2lnQ);
            }

        m_points[upper_ratio]=lower_CL;
        m_points[lower_ratio]=upper_CL;

        if (is_verbose()){
            std::cout<<"\n\n\n-----------------------------------------------\n"
                      << "Scan situation:\n"
                      << " - Left side (Low Ratio, High CL) = ("
                      << lower_ratio << "," << upper_CL << ")\n"
                      << " - Right side (High Ratio, Low CL) = ("
                      << upper_ratio << "," << lower_CL << ")\n"
                      << "-----------------------------------------------\n\n";
            }

        attempts-=1;
        }



    if (attempts!=0){
        TString name = GetName();
        name += "_results";
        TString title = GetTitle();
        title += " Results";

        return new RatioFinderResults(name.Data(),
                                      title.Data(),
                                      n_toys,
                                      CL_level,
                                      upper_ratio,
                                      lower_CL,
                                      lower_ratio,
                                      upper_CL,
                                      delta_ratios_min,
                                      m_points);
        }

    else{
        std::cerr << "\n[RatioFinder::findRatio] " 
                  << "Maximum number of attempts to find "
                  << " CL = " << CL_level << " reached (" 
                  << m_max_attempts << "). "
                  << "I may be stuck for sum reason "
                  << "(too few toys?). Returning 0...\n"; 

        return NULL;
        }
    }

/*----------------------------------------------------------------------------*/

/**
Perform a wheighted average of two values.
**/
double RatioFinder::m_weighted_average(double h_val,
                                       double h_weight,
                                       double l_val,
                                       double l_weight){

   double wavg=(h_val*h_weight+l_val*l_weight)/(h_weight+l_weight);
   if (is_verbose()){
            std::cout << "Calculating new value:\n"
                      << "( "<< h_val<< " * " << h_weight << " + " << l_val
                      << " * " << l_weight <<" )/( "<< h_weight << " + "
                      << l_weight<< " ) = " << wavg << std::endl;
            }

    return (wavg);
    }

/*----------------------------------------------------------------------------*/

/**
Set the new value of the ratio variable and calculate the CLs value.
**/
double RatioFinder::m_get_CLs(double ratio,
                              unsigned int n_toys,
                              double& m2lnQ_on_data){

    if (is_verbose())
        std::cout << "\nGetting Cls with:\n"
                  << " - " << m_ratio->GetName() << " set to " << ratio << std::endl
                  << " - n_toys  set to " << n_toys << std::endl
                  << " - m2lnQ_on_data set to " << m2lnQ_on_data << std::endl;

    // Set the val
    m_ratio->setVal(ratio);

    // Get the results
    LimitResults* res = m_get_LimitResults(n_toys);


    LimitPlot* p = res->getPlot("temp","temp",m_nbins);
    p->draw();
    TH1F* b_histo = p->getBhisto();
    m2lnQ_on_data = Rsc::getMedian(b_histo);
    if (is_verbose())
        std::cout << "-2lnQ was calculated as the median of the"
                    << " b distribution.\n --> -2lnQ = "<< m2lnQ_on_data <<"\n";
    delete p;

    // Put the -2lnQ value
    res->setM2lnQValue_data(m2lnQ_on_data);

    // Get the plot and print it for control reasons.
//     TString plot_name("m2lnQ_distributions_R_");
//     plot_name+=ratio;

    TString plot_name(GetName());
    plot_name+="_m2lnQ_distributions_";
    plot_name+=m_ratio->GetName();
    plot_name+=ratio;
    plot_name+="_m2lnQ";
    plot_name+=m2lnQ_on_data;

    plot_name.ReplaceAll(" ",""); // remove the spaces of the double

    p=res->getPlot(plot_name.Data(),plot_name.Data(),m_nbins);

    p->draw();

    p->dumpToImage((plot_name+".png").Data());

    // end of the control plots creation

    double CLs=res->getCLs();

    std::cout << "For ratio " << ratio << " CLs is " << CLs << std::endl;

    delete p;
    delete res;

    return CLs;

    }

/*----------------------------------------------------------------------------*/

/**
Perform the construction of the -2lnQ distributions.
The value of the variable to study is changed in the m_get_CLs function. Since 
we have the pointer as a member the change persists.
**/
LimitResults* RatioFinder::m_get_LimitResults(unsigned int n_toys){

    LimitCalculator calc("m2lnQmethod",
                         "-2lnQ method",
                         m_sb_model,
                         m_b_model,
                         &m_variables,
                         m_c_array);

    LimitResults* res = calc.calculate (n_toys,true);

    return res;
    }

/*----------------------------------------------------------------------------*/

void RatioFinder::print(const char* options){

    std::cout << "Ratio finder Object " << GetName() << " " << GetTitle()
              << std::endl;
    std::cout << "SB model: ";
    m_sb_model->Print(options);
    std::cout << "B model: ";
    m_b_model->Print(options);
    std::cout << "Variables: ";
    m_variables.Print(options);
    if (m_c_array != NULL){
        std::cout << "Constraint Array: ";
        m_c_array->print(options);
        }
    }

/*----------------------------------------------------------------------------*/
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
