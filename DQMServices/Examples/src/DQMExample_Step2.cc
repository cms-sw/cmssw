#include "DQMServices/Examples/interface/DQMExample_Step2.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

//
// -------------------------------------- Constructor --------------------------------------------
//
DQMExample_Step2::DQMExample_Step2(const edm::ParameterSet& ps)
{
  edm::LogInfo("DQMExample_Step2") <<  "Constructor  DQMExample_Step2::DQMExample_Step2 " << std::endl;

  //DQMStore
  dbe_ = edm::Service<DQMStore>().operator->();
  
  // Get parameters from configuration file
  numMonitorName_      =  ps.getParameter<std::string>("numMonitorName");
  denMonitorName_      =  ps.getParameter<std::string>("denMonitorName");

}

//
// -- Destructor
//
DQMExample_Step2::~DQMExample_Step2()
{
  edm::LogInfo("DQMExample_Step2") <<  "Destructor DQMExample_Step2::~DQMExample_Step2 " << std::endl;
}

//
// -------------------------------------- beginJob --------------------------------------------
//
void DQMExample_Step2::beginJob()
{
  edm::LogInfo("DQMExample_Step2") <<  "DQMExample_Step2::beginJob " << std::endl;
}
//
// -------------------------------------- beginRun --------------------------------------------
//
void DQMExample_Step2::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) 
{
  edm::LogInfo("DQMExample_Step2") <<  "DQMExample_Step2::beginRun" << std::endl;
  
  //book at beginRun
}
//
// -------------------------------------- beginLuminosityBlock --------------------------------------------
//
void DQMExample_Step2::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                                            edm::EventSetup const& context) 
{
  edm::LogInfo("DQMExample_Step2") <<  "DQMExample_Step2::beginLuminosityBlock" << std::endl;
}


//
// -------------------------------------- Analyze --------------------------------------------
//
void DQMExample_Step2::analyze(edm::Event const& e, edm::EventSetup const& eSetup)
{
  edm::LogInfo("DQMExample_Step2") <<  "DQMExample_Step2::analyze" << std::endl;



}
//
// -------------------------------------- endLuminosityBlock --------------------------------------------
//
void DQMExample_Step2::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) 
{
  edm::LogInfo("DQMExample_Step2") <<  "DQMExample_Step2::endLuminosityBlock" << std::endl;
}


//
// -------------------------------------- endRun --------------------------------------------
//
void DQMExample_Step2::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  edm::LogInfo("DQMExample_Step2") <<  "DQMExample_Step2::endRun" << std::endl;

  // create and cd into new folder
  dbe_->setCurrentFolder("What_I_do_in_the_client/Ratio");

  //get available histograms
  MonitorElement* numerator = dbe_->get(numMonitorName_);
  MonitorElement* denominator = dbe_->get(denMonitorName_);

  if (!numerator || !denominator)
    {
      edm::LogError("DQMExample_Step2") <<  "MEs not found!" << std::endl;
      return;
    }


  //book new histogram
  h_ptRatio = dbe_->book1D("ptRatio","pt ratio pf matched objects",50,0.,100.);
  h_ptRatio->setAxisTitle("pt [GeV]");

  for (int iBin=1; iBin<numerator->getNbinsX(); ++iBin)
    {
      if(denominator->getBinContent(iBin) == 0)
	h_ptRatio->setBinContent(iBin, 0.);
      else
	h_ptRatio->setBinContent(iBin, numerator->getBinContent(iBin) / denominator->getBinContent(iBin));
    }
}

//
// -------------------------------------- endJob --------------------------------------------
//
void DQMExample_Step2::endJob()
{
  edm::LogInfo("DQMExample_Step2") <<  "DQMExample_Step2::endJob" << std::endl;
}


