// @(#)root/hist:$Id: LimitResults.cc,v 1.3 2009/04/15 11:10:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"
#include "PhysicsTools/RooStatsCms/interface/LimitResults.h"


/*----------------------------------------------------------------------------*/

LimitResults::LimitResults(const char* name,
                           const char* title,
                           std::vector<float>& m2lnq_sb_vals,
                           std::vector<float>& m2lnq_b_vals,
                           float m2lnq_data):
    StatisticalMethod(name,title,true),
    m_m2lnQ_data(m2lnq_data),
    m_CLsb(-100),
    m_CLb(-100){


    int vector_size = m2lnq_b_vals.size();

    assert(vector_size>0);

    m_m2lnQ_sb.reserve(vector_size);
    m_m2lnQ_b.reserve(vector_size);

    for (int i=0;i< vector_size;++i){
        m_m2lnQ_sb.push_back(m2lnq_sb_vals[i]);
        m_m2lnQ_b.push_back(m2lnq_b_vals[i]);
        }

    }
/*----------------------------------------------------------------------------*/

LimitResults::LimitResults():
    StatisticalMethod("Defaultname","Default title",true),
    m_m2lnQ_data(0),
    m_CLsb(-100),
    m_CLb(-100){
    }

/*----------------------------------------------------------------------------*/

LimitResults::~LimitResults(){

    m_m2lnQ_sb.clear();
    m_m2lnQ_b.clear();

    }


/*----------------------------------------------------------------------------*/

double LimitResults::getCLsb(){

    m_build_CLsb();

    return m_CLsb;

    }

/*----------------------------------------------------------------------------*/

void LimitResults::m_build_CLsb(){

    int n_toys=m_m2lnQ_sb.size();
    double right_of_measured=0;

    for (int i=0;i<n_toys;++i)
        if (m_m2lnQ_sb[i]>m_m2lnQ_data)
            ++right_of_measured;

    if (right_of_measured==0){
        double epsilon=0.00001;
        std::cerr << "[LimitResults::m_build_CLsb()] ERROR: "
                  << " CLsb is 0. This might be the result of the use of few toys ("
                  << n_toys << "). Putting a small number ("
                  << epsilon << ") for safe divisions...\n";
        m_CLsb=epsilon;
        }
    else
        m_CLsb = right_of_measured/n_toys;

    }

/*----------------------------------------------------------------------------*/

double LimitResults::getCLb(){

    m_build_CLb();

    return m_CLb;

    }

/*----------------------------------------------------------------------------*/

double LimitResults::getCLs(){

    double CLs= getCLsb()/getCLb();

    if (CLs>1){
        std::cerr << "[LimitResults::getCLs] ERROR: "
                  << "CLs is greater than 1 (CLs = " << CLs 
                  << "). This might be due to the low number of toys. Putting 1 instead.\n";
        return 1;
        }
    else
        return CLs;

    }

/*----------------------------------------------------------------------------*/

void LimitResults::m_build_CLb(){

    int n_toys=m_m2lnQ_b.size();
    double right_of_measured=0;

    for (int i=0;i<n_toys;++i)
        if (m_m2lnQ_b[i]>m_m2lnQ_data)
            ++right_of_measured;

    if (right_of_measured==0){
        double epsilon=0.00001;
        std::cerr << "[LimitResults::m_build_CLb()] ERROR: "
                  << " CLb is 0. This might be the result of the use of few toys ("
                  << n_toys << "). Putting a small number ("
                  << epsilon << ") for safe divisions...\n";
        m_CLb=epsilon;
        }
    else
        m_CLb = right_of_measured/n_toys;

    }

/*----------------------------------------------------------------------------*/

void LimitResults::add(LimitResults* other){

    // Add the -2lnQ values to the existing ones!
    int other_size=getM2lnQValues_sb().size();

    for (int i=0;i<other_size;++i){
        m_m2lnQ_sb.push_back(other->getM2lnQValues_sb()[i]);
        m_m2lnQ_b.push_back(other->getM2lnQValues_b()[i]);
        }

    // Put them to a negartive value to recalculate the value.
    m_CLb=m_CLsb=-100;

    }

/*----------------------------------------------------------------------------*/


void LimitResults::print(const char* options){

    std::cout << "\nResults " << GetName() << ":\n"
              << " - Number of toys: " << m_m2lnQ_b.size() << std::endl
              << " - -2lnQ on data: " << m_m2lnQ_data << std::endl
              << " - CLb " << getCLb() << std::endl
              << " - CLsb " << getCLsb() << std::endl
              << " - CLs " << getCLs() << std::endl;
    }

/*----------------------------------------------------------------------------*/

LimitPlot* LimitResults::getPlot(const char* name,const char* title, int n_bins){

    TString plot_name;
    if (TString(name)==""){
        plot_name+=GetName();
        plot_name+="_plot";
        }
    else
        plot_name=name;

    TString plot_title;
    if (TString(title)==""){
        plot_title+=GetTitle();
        plot_title+="_plot (";
        plot_title+=m_m2lnQ_b.size();
        plot_title+=" toys)";
        }
    else
        plot_title=title;

    LimitPlot* p=new LimitPlot(plot_name.Data(),
                               plot_title.Data(),
                               m_m2lnQ_sb,
                               m_m2lnQ_b,
                               m_m2lnQ_data,
                               n_bins,
                               true);
    return p;
    }

/*----------------------------------------------------------------------------*/
/*
double LimitResults::m_getMedian(std::vector<float>& vals){

    if (not m_vectors_sorted){
        std::sort(m_m2lnQ_sb.begin(), m_m2lnQ_sb.end(), m_sorting_criterium);
        std::sort(m_m2lnQ_b.begin(), m_m2lnQ_b.end(), m_sorting_criterium);
        m_vectors_sorted=true;
        }
    int vsize=vals.size();

    if (vsize%2 == 1)
        return vals[(vsize-1)/2]; // remember! it starts from 0!

    else{
        return 0.5*(vals[vsize/2]);
        }

    }*/

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/


// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
