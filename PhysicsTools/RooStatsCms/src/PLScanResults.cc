// @(#)root/hist:$Id: PLScanResults.cc,v 1.1.1.1 2009/04/15 08:40:01 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008


#include <iostream>

#include "TMath.h"

#include "TCanvas.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TMatrix.h"
#include "TLine.h"
#include "TF1.h"
#include "TGraphErrors.h"

#include "PhysicsTools/RooStatsCms/interface/PLScanResults.h"


/*----------------------------------------------------------------------------*/

PLScanResults::PLScanResults(const char* name,
                             const char* title,
                             std::vector<double> points_grid,
                             std::vector<double> NLL_values,
                             const char* scanned_var_name,
                             double genval):
    StatisticalMethod(name,title,true),
    m_scanned_parameter_name(scanned_var_name),
    m_generated_value(genval){

    m_add_scan_values(points_grid,NLL_values);
    m_fill_scan_range_extremes();
    m_find_scan_minimum();
    m_fill_NLL_shifted_values();

    }

/*----------------------------------------------------------------------------*/

PLScanResults::PLScanResults():
    StatisticalMethod("Default name","Default title",true),
    m_scanned_parameter_name("defaultCtor"),
    m_generated_value(0){
    }

/*----------------------------------------------------------------------------*/

/**
Find the scan minimum and the left and right points. Approximate the minimum 
with a parabola.
**/

void PLScanResults::m_find_scan_minimum(){

    assert(m_NLL_values.size()>3);

    // Find the minimum towards a scan of the values. Record the index
    int min_indexes[3];
    double min_values[3];
    for (int i=0;i<3;++i)
        min_values[i]=1e50;

    std::vector<double>::iterator it;
    int index=0;
    for (it=m_NLL_values.begin();it<m_NLL_values.end();++it){
        for (int i=0;i<3;++i)
        if (*it<min_values[0]){
            // values
            min_values[2]=min_values[1];
            min_values[1]=min_values[0];
            min_values[0]=*it;
            // indexes
            min_indexes[2]=min_indexes[1];
            min_indexes[1]=min_indexes[0];
            min_indexes[0]=index;
            }
        else if (*it>min_values[0] and *it<min_values[1]){
            // values
            min_values[2]=min_values[1];
            min_values[1]=*it;
            // indexes
            min_indexes[2]=min_indexes[1];
            min_indexes[1]=index;
            }
        else if (*it>min_values[0] and *it>min_values[1] and *it<min_values[2]){
            // values
            min_values[2]=*it;
            // indexes
            min_indexes[2]=index;
            }
        ++index;
        }


//     std::cout << "m_NLL_values.size() = " << m_NLL_values.size() << std::endl;
// 
//     for (int i=0;i< m_NLL_values.size();++i){
//         std::cout << " - Val " << i << m_NLL_values[i] << "\n";
//         }

    if (is_verbose()){
        std::cout << "[PLScanResults::m_find_scan_minimum] "
                  << " three lowest points indeces:\n";
        for (int i=0;i<3;++i){
            std::cout << " - x: " << m_points_grid[min_indexes[i]]
                      << " - y: " << m_NLL_values[min_indexes[i]]
                      << std::endl;
            }
        }

    // Fill the coordinates of the 3 lowest points and get the parabola params
    double a,b,c;
    double x[3],y[3];


    for (int i=0;i<3;++i){
        x[i]=m_points_grid[min_indexes[i]];
        y[i]=m_NLL_values[min_indexes[i]];
        }

    /*
     if the minimum point is the one at the right, do not approximate, just return it
    */
    bool is_rightest=true;
    for (it=m_points_grid.begin();it<m_points_grid.end();++it)
        if (x[0] > *it){
            is_rightest=false;
            break;
            }

    if (not is_rightest){
        m_parabola(x,y,a,b,c);

        m_scan_minimum[0]=-1*b/(2*a);
        m_scan_minimum[1]=c-b*b/(4*a);
        }
    else{
        m_scan_minimum[0]=x[0];
        m_scan_minimum[1]=y[0];
        }
    

    }

/*----------------------------------------------------------------------------*/

/**
Add the points to the scan. Shift the points by the offset to be able to have 
always the minimum y of the NLL at 0.
**/

void PLScanResults::m_add_scan_values(std::vector<double> points_grid,
                                      std::vector<double> NLL_values){

    for (std::vector<double>::iterator it=points_grid.begin();
         it<points_grid.end();++it)
        m_points_grid.push_back(*it);

    for (std::vector<double>::iterator it=NLL_values.begin();
         it<NLL_values.end();++it)
        m_NLL_values.push_back(*it);

    }

/*----------------------------------------------------------------------------*/

void PLScanResults::m_fill_scan_range_extremes(){
    m_scan_range_min=1e50;
    m_scan_range_max=-1e50;

    for (std::vector<double>::iterator it=m_points_grid.begin();
         it<m_points_grid.end();
         ++it){
        if (*it>m_scan_range_max)
            m_scan_range_max=*it;
        if (*it<m_scan_range_min)
            m_scan_range_min=*it;
        }
    }

/*----------------------------------------------------------------------------*/

void PLScanResults::m_fill_NLL_shifted_values(){

    for (std::vector<double>::iterator it=m_NLL_values.begin();
         it<m_NLL_values.end();
         ++it)
        m_NLL_shifted_values.push_back(*it-m_scan_minimum[1]);
    }

/*----------------------------------------------------------------------------*/

double PLScanResults::getDeltaNLLfromCL(double CL){

    double erfinv= TMath::ErfInverse (CL);
    double DeltaNll= erfinv*erfinv;

    return DeltaNll;
    }

/*----------------------------------------------------------------------------*/

double PLScanResults::getDeltaNLLfromCL_limit(double CL){
    double sqrt_DeltaNLLfromCL_limit=TMath::ErfInverse(2*CL-1);
    return sqrt_DeltaNLLfromCL_limit*sqrt_DeltaNLLfromCL_limit;

    }

/*----------------------------------------------------------------------------*/

double PLScanResults::getCLfromDeltaNLL(double DeltaNll){

    return TMath::ErfInverse(sqrt(DeltaNll));
    }

/*----------------------------------------------------------------------------*/

/**
Add the content of a scan to the current scan. The cahched confidence levels 
are reset and the points of the scan added to the existing ones.
**/

void PLScanResults::add(PLScanResults addendum){

    // Clear the maps
    m_CL_UL_map.clear();
    m_CL_LL_map.clear();
    m_NLL_shifted_values.clear();

    // add the new points 
    m_add_scan_values(addendum.m_points_grid,addendum.m_NLL_values);
    m_fill_scan_range_extremes();
    m_find_scan_minimum();
    m_fill_NLL_shifted_values();
    }

/*----------------------------------------------------------------------------*/

/**
Produce a plot of the scan.
**/

PLScanPlot* PLScanResults::getPlot(const char* name, const char* title){

    if (TString(name)=="")
        name=GetName();

    if (TString(title)=="")
        title=GetTitle();

    PLScanPlot* plot = new PLScanPlot(name,
                                      title,
                                      m_scanned_parameter_name.Data(),
                                      m_points_grid,
                                      m_NLL_shifted_values,
                                      m_scan_minimum[0]);

    return plot;

    }

/*----------------------------------------------------------------------------*/

void PLScanResults::print(const char* options){
    std::cout << "\n PLScanResults object: \n\n"
              << " - Name:" << GetName() << std::endl
              << " - Title:" << GetTitle() << std::endl
              << " - Stored Scanned points: " << m_points_grid.size() 
                << std::endl
              << " - Generated value (default= " << DUMMY_VALUE
                 << "):" << m_generated_value << std::endl
              << " - Minimum: ("<< m_scan_minimum[0] << ","
                                << m_scan_minimum[1] << ")\n"
              << " - Extremes: ["<< m_scan_range_min << ","
                                 << m_scan_range_max << "]\n"
              << " - Scanned variable name : " << m_scanned_parameter_name
                                 << std::endl; 
    }

/*----------------------------------------------------------------------------*/

/**
Use the inversion of the 3x3 coefficients matrix to find the parabola
**/

void PLScanResults::m_parabola(double* x, double* y,
                double& a, double& b, double& c){

    TMatrixT<double> m(3,3);

    for (int i=0;i<3;++i){
        m[i][0]=x[i]*x[i];
        m[i][1]=x[i];
        m[i][2]=1;
        }

    m.InvertFast();

    double coeff[3];

    for (int i=0;i<3;++i)
        coeff[i]=m[i][0]*y[0]+m[i][1]*y[1]+m[i][2]*y[2];

    a=coeff[0];
    b=coeff[1];
    c=coeff[2];

    }

/*----------------------------------------------------------------------------*/

/**
Convert the CL to a DeltaLL value and then intersect the value with the scan.
**/

double PLScanResults::getUL(double deltaNll){

    if (m_CL_UL_map.count(deltaNll)==0){
        //double deltaNll=getDeltaNLLfromCL(ConfidenceLevel);
        double ul;
        TF1 poly0("orizzline","pol0",m_scan_minimum[0],m_scan_range_max);
        poly0.SetParameter(0,deltaNll);
        bool found =m_intersect(&poly0,ul,m_scan_minimum[0],m_scan_range_max);
        std::cout << "[PLScanResults::getUL] Caching Limit..\n";
        if (found)
            m_CL_UL_map[deltaNll]=ul;
        else
            m_CL_UL_map[deltaNll]=m_scan_range_max;
        }
    return m_CL_UL_map[deltaNll];
    }

/*----------------------------------------------------------------------------*/

double PLScanResults::getLL(double deltaNll){

    if (m_CL_LL_map.count(deltaNll)==0){
        //double deltaNll=getDeltaNLLfromCL(ConfidenceLevel);
        double ll;
        TF1 poly0("orizzline","pol0",m_scan_range_min,m_scan_minimum[0]);
        poly0.SetParameter(0,deltaNll);
        bool found =m_intersect(&poly0,ll,m_scan_range_min,m_scan_minimum[0]);
        std::cout << "[PLScanResults::getLL] Caching Limit..\n";
        if (found)
            m_CL_LL_map[deltaNll]=ll;
        else
            m_CL_LL_map[deltaNll]=m_scan_range_min;
        }
    return m_CL_LL_map[deltaNll];
    }

/*----------------------------------------------------------------------------*/

/**
Get the upper limit intersecting the fitted FC graph with the Likelihood Scan.
The interpolation between the points of the scan will be linear.
The procedure to find the intersection:
 - Find the two nearest points to the poly fitted to the FC graph
 - Create a line passing through these 2 points 
 - Get the mean of the fitted line in the interval between the scan points
 - Find the intersection.
**/

double PLScanResults::getUL(TGraphErrors* FC_graph,double x_min, double x_max){

    if (x_min==0 and x_max==0){
        x_min=m_scan_range_min;
        x_min=m_scan_range_max;
        }

    // Fit the FC graph with a poly
    int n_points=FC_graph->GetN();
    int pol_degree=0;
    if (n_points>=3)
        pol_degree=3;
    else
        pol_degree=n_points;

    TString pol_name="pol";
    pol_name+=pol_degree;
    TF1 poly ("FC_interpolation",
              pol_name.Data(),
              x_min,
              x_max);

    FC_graph->Fit(&poly,"R");

    double ul;
    bool found =m_intersect(&poly,ul,m_scan_minimum[0],m_scan_range_max);
    if (found)
        return ul;
    else
        return m_scan_range_max;

    }

/*----------------------------------------------------------------------------*/

double PLScanResults::getLL(TGraphErrors* FC_graph,double x_min, double x_max){

    if (x_min==0 and x_max==0){
        x_min=m_scan_range_min;
        x_min=m_scan_range_max;
        }

    // Fit the FC graph with a poly
    int n_points=FC_graph->GetN();
    int pol_degree=0;
    if (n_points>=3)
        pol_degree=3;
    else
        pol_degree=n_points;

    TString pol_name="pol";
    pol_name+=pol_degree;
    TF1 poly ("FC_interpolation",
              pol_name.Data(),
              m_scan_range_min,
              m_scan_range_max);

    FC_graph->Fit(&poly,"R");

    double ll;
    bool found =m_intersect(&poly,ll,m_scan_range_min,m_scan_minimum[0]);
    if (found)
        return ll;
    else
        return m_scan_range_min;

    }

/*----------------------------------------------------------------------------*/

void PLScanResults::m_pol1(double* p1, double* p2, double& m, double& q){
    m=(p2[1]-p1[1])/(p2[0]-p1[0]);
    q=p1[1]-m*p1[0];
    }

/*----------------------------------------------------------------------------*/

bool PLScanResults::m_intersect(TF1* function,
                                double& intersection_asc,
                                double range_min,
                                double range_max){

    // find in the range the points immediately upper and lower the orizz line
    double x,y;
    double dist_prec;
    double dist;

    bool flip=false;
    bool in_roi=false;

    int prec_index=0;
    int flip_index=0;

    // cycle on the points of the scan
    for (int index=0;index<m_NLL_shifted_values.size();++index){
        x=m_points_grid[index];
        y=m_NLL_shifted_values[index];

        // see if the scan crosses the function interpolating FC
        if (x> range_max or x<range_min)
            continue;

        else if (not in_roi){
            in_roi=true;
            dist_prec=y-function->Eval(x);
            prec_index=index;
            //std::cout << "\n\n In roi --> x = " << x << std::endl;
            continue;
            }

        dist=y-function->Eval(x);
        //std::cout << " x = " << x <<  " dist = " << dist << std::endl;

        if (dist*dist_prec < 0){ // do we have a flip?
            flip=true;
            flip_index=index;
            //std::cout << " FLIP!!" <<std::endl;
            break;
            }
        prec_index=index;
        }

    if (not flip)
        if (dist>0)
            intersection_asc = range_max;
        else
            intersection_asc = range_min;
    else{

        double p1_scan[2];
        p1_scan[0]=m_points_grid[prec_index];
        p1_scan[1]=m_NLL_shifted_values[prec_index];

        double p2_scan[2];
        p2_scan[0]=m_points_grid[flip_index];
        p2_scan[1]=m_NLL_shifted_values[flip_index];

        double m_scan,q_scan;
        m_pol1(p1_scan,p2_scan,m_scan,q_scan);

        double p1_fit[2];
        p1_fit[0]=m_points_grid[prec_index];
        p1_fit[1]=function->Eval(p1_fit[0]);

        double p2_fit[2];
        p2_fit[0]=m_points_grid[flip_index];
        p2_fit[1]=function->Eval(p2_fit[0]);

        double m_fit,q_fit;
        m_pol1(p1_fit,p2_fit,m_fit,q_fit);

        intersection_asc = (q_fit - q_scan)/ (m_scan -m_fit); //to be improved
        }
    return true;
    }
//     // find in the range the points immediately upper and lower the orizz line
//     double x,y;
//     double dist_up=1e40;
//     double dist_down=-1e40;
//     double dist;
//     int index_up,index_down;
// 
//     bool found_up,found_down;
//     found_up=found_down=false;
// 
//     for (int index=0;index<m_NLL_shifted_values.size();++index){
//         x=m_points_grid[index];
//         y=m_NLL_shifted_values[index];
// 
//         if (x> range_max or x<range_min)
//             continue;
// 
//         dist=y-function->Eval(x);
// 
//         if (dist>0)
//             if (dist<dist_up){
//                 found_up=true;
//                 dist_up=dist;
//                 index_up=index;
//                 }
// 
//         if (dist<0)
//             if (dist>dist_down){
//                 found_down=true;
//                 dist_down=dist;
//                 index_down=index;
//                 } 
//        }
// 
//     // if no uppper/lower point
//     if (not (found_up and found_down)){
//         std::cout << "ERROR: no upper/lower point in "
//                   << " the scan for Specified function...\n ";
// 
//         return false;
// 
//         }
// 
//     // Build the points 
//     double point_up[2];
//     point_up[0]=m_points_grid[index_up];
//     point_up[1]=m_NLL_shifted_values[index_up];
// 
//     double point_down[2];
//     point_down[0]=m_points_grid[index_down];
//     point_down[1]=m_NLL_shifted_values[index_down];
// 
//     double m,q;
//     m_pol1(point_up,point_down,m,q);
// 
//     // intersecting function mean between the 2 points
//     double intersection_ord = 0.5*(function->Eval(point_up[0])+
//                                    function->Eval(point_up[1]));
// 
//     // intersection between two segments
//     intersection_asc = (intersection_ord - q)/m;
// 
//     return true;
//     }

/*----------------------------------------------------------------------------*/

/**
Check the coverage of the result. The interval is considered to be two sided.
**/
bool PLScanResults::isCovering(double CL){
    return (m_generated_value<getUL(getDeltaNLLfromCL(CL)) and m_generated_value >getLL(getDeltaNLLfromCL(CL)));
    }

/*----------------------------------------------------------------------------*/
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
