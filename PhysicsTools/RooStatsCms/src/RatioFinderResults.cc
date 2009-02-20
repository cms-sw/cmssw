// @(#)root/hist:$Id: RatioFinderResults.cc,v 1.1 2009/01/06 12:22:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   07/10/2008

#include "assert.h"

#include <iostream>

#include "PhysicsTools/RooStatsCms/interface/RatioFinderResults.h"

/// To build the cint dictionaries
//ClassImp(RatioFinderResults)

/*----------------------------------------------------------------------------*/
RatioFinderResults::RatioFinderResults(const char* name,
                                       const char* title,
                                       int n_toys,
                                       double CL_level,
                                       double upper_ratio,
                                       double lower_CL,
                                       double lower_ratio,
                                       double upper_CL,
                                       double delta_ratios_min,
                                       std::map<double,double> points):

    StatisticalMethod(name,title,true),
    m_n_toys(n_toys),
    m_CL_level(CL_level),
    m_upper_ratio(upper_ratio),
    m_lower_ratio(lower_ratio),
    m_upper_CL(upper_CL),
    m_lower_CL(lower_CL),
    m_delta_ratios_min(delta_ratios_min){

    // Copy the map of values into the member map
    for (std::map<double,double>::iterator iter=points.begin();
         iter!=points.end();
         ++iter){
        if (is_verbose())
            std::cout << " - (ratio,CL) = (" << iter->first << ","
                                             << iter->second << ")\n";
        m_points[iter->first]=iter->second;
        }
    }

/*----------------------------------------------------------------------------*/

RatioFinderResults::RatioFinderResults():
    StatisticalMethod("RatioFinderResults","RatioFinderResults",true),
    m_n_toys(0),
    m_CL_level(0),
    m_upper_ratio(0),
    m_lower_ratio(0),
    m_upper_CL(0),
    m_lower_CL(0),
    m_delta_ratios_min(0){}

/*----------------------------------------------------------------------------*/

RatioFinderResults::~RatioFinderResults(){
    // Nothing to free!
    }

/*----------------------------------------------------------------------------*/
/**
Get an interpolation of the two best values of the ratios that were found.
The possible options at the moment are:
 - <EM>Linear</EM> for a linear approximation
**/

double RatioFinderResults::getInterpolatedRatio(const char* option){

    TString s_option(option);

    if (s_option = "Linear" ){
        double p1[2];
        p1[0] = m_upper_ratio;
        p1[1] = m_lower_CL;
        double p2[2];
        p2[0] = m_lower_ratio;
        p2[1] = m_upper_CL;

        double m=(p2[1]-p1[1])/(p2[0]-p1[0]);
        double q = p1[1] - m * p1[0];

    return ((m_CL_level - q) / m);

        }
    else {
        std::cerr << "[RatioFinderResults::getInterpolatedRatio] "
                  << "Option " << option << " not recognised. Aborting..\n";
        abort();
        }
    }

/*----------------------------------------------------------------------------*/

RatioFinderPlot* RatioFinderResults::getPlot(const char* name,
                                             const char* title){

    TString s_name(name);
    TString s_title(title);

    if (s_name==""){
        s_name=GetName();
        s_name+="_plot";
        s_title=GetTitle();
        s_title+=" plot";
        }

    RatioFinderPlot* p = new RatioFinderPlot(s_name.Data(),
                               s_title.Data(),
                               m_CL_level,
                               m_points);

    if (is_verbose())
        std::cout << "Returning the ratio finder plot\n";

    return p;

    }

/*----------------------------------------------------------------------------*/

void RatioFinderResults::print(const char* options){

    std::cout << "A RatioFinderResults object " << GetName() << "\"" 
              << GetTitle() << "\"\n";

    std::cout << " - Number of toys per point: " << m_n_toys << std::endl
              << " - Confidence level to reach: " << m_CL_level << std::endl
              << " - Lower ratio: " << m_lower_ratio << std::endl
              << " - Upper CL: " << m_upper_CL << std::endl
              << " - Upper ratio: " << m_upper_ratio << std::endl
              << " - Lower CL: " << m_lower_CL << std::endl
              << " - Ratio-Cl values: " << std::endl;
    for (std::map<double,double>::iterator iter=m_points.begin();
         iter!=m_points.end();
         ++iter){
        std::cout << "   * (ratio,CL) = (" << iter->first << ","
                                           << iter->second << ")\n";
        }
    }

