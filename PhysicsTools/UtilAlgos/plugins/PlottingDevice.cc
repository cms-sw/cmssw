// -*- C++ -*-
//
// Package:    PlottingDevice
// Class:      PlottingDevice
// 
/**\class PlottingDevice PlottingDevice.cc Workspace/PlottingDevice/src/PlottingDevice.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Thu May 15 14:37:59 CEST 2008
// $Id: PlottingDevice.cc,v 1.7 2009/12/18 17:52:25 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// class decleration
//

#include "PhysicsTools/UtilAlgos/interface/Plotter.h"
#include "PhysicsTools/UtilAlgos/interface/VariableHelper.h"

class PlottingDevice : public edm::EDAnalyzer {
   public:
      explicit PlottingDevice(const edm::ParameterSet&);
      ~PlottingDevice();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
  std::string vHelperInstance_;
  std::string plotDirectoryName_;
  Plotter * plotter_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PlottingDevice::PlottingDevice(const edm::ParameterSet& iConfig)
{
  vHelperInstance_ = iConfig.getParameter<std::string>("@module_label");
  plotDirectoryName_="PlottingDevice";
  
  //configure the inputtag distributor
  if (iConfig.exists("InputTags"))
    edm::Service<InputTagDistributorService>()->init(vHelperInstance_,iConfig.getParameter<edm::ParameterSet>("InputTags"));
  
  //configure the variable helper
  edm::Service<VariableHelperService>()->init(vHelperInstance_,iConfig.getParameter<edm::ParameterSet>("Variables"));

  //configure the plotting device
  edm::ParameterSet plotPset = iConfig.getParameter<edm::ParameterSet>("Plotter");
  std::string plotterName = plotPset.getParameter<std::string>("ComponentName");
  plotter_ = PlotterFactory::get()->create(plotterName, plotPset);
}


PlottingDevice::~PlottingDevice(){}


//
// member functions
//

// ------------ method called to for each event  ------------
void
PlottingDevice::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  plotter_->setDir(plotDirectoryName_);

  plotter_->fill(plotDirectoryName_, iEvent);
}


void PlottingDevice::beginJob(){}
void PlottingDevice::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(PlottingDevice);
