// @(#)root/hist:$Id: StatisticalPlot.cc,v 1.1.1.1 2009/04/15 08:40:01 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008


#include "PhysicsTools/RooStatsCms/interface/StatisticalPlot.h"
#include "TROOT.h"

/*----------------------------------------------------------------------------*/

StatisticalPlot::StatisticalPlot(const char* name,
                                 const char* title,
                                 bool verbosity):
    TNamed(name,title){
    setVerbosity(verbosity);
    //m_canvas=new TCanvas(name,title);
    gROOT->SetStyle("Plain");
    }

/*----------------------------------------------------------------------------*/

void StatisticalPlot::setVerbosity(bool verbosity){
    m_verbose=verbosity;
    }

/*----------------------------------------------------------------------------*/

bool StatisticalPlot::is_verbose(){
    return m_verbose;
    }

/*----------------------------------------------------------------------------*/

StatisticalPlot::~StatisticalPlot(){
    //delete m_canvas;
    }

/*----------------------------------------------------------------------------*/

// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
