// -*- C++ -*-
//
// Package:    SiPixelPhase1EfficiencyExtras
// Class:      SiPixelPhase1EfficiencyExtras
//
/**\class 

 Description: Create the Phase 1 extra efficiency trend plots

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jack Sisson, Julie Hogan
//         Created:  7 July, 2021
//
//

// Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

// DQM Framework
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <algorithm>

class SiPixelPhase1EfficiencyExtras : public DQMEDHarvester {
public:
  explicit SiPixelPhase1EfficiencyExtras(const edm::ParameterSet& conf);
  ~SiPixelPhase1EfficiencyExtras() override;

protected:
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

  void dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) override;

  const std::string effFolderName_;
  const std::string vtxFolderName_;
  const std::string instLumiFolderName_;
};

SiPixelPhase1EfficiencyExtras::SiPixelPhase1EfficiencyExtras(const edm::ParameterSet& iConfig)
    : effFolderName_(iConfig.getParameter<std::string>("EffFolderName")),
      vtxFolderName_(iConfig.getParameter<std::string>("VtxFolderName")),
      instLumiFolderName_(iConfig.getParameter<std::string>("InstLumiFolderName")) {
  edm::LogInfo("PixelDQM") << "SiPixelPhase1EfficiencyExtras::SiPixelPhase1EfficiencyExtras: Hello!";
}

SiPixelPhase1EfficiencyExtras::~SiPixelPhase1EfficiencyExtras() {
  edm::LogInfo("PixelDQM") << "SiPixelPhase1EfficiencyExtras::~SiPixelPhase1EfficiencyExtras: Destructor";
}

void SiPixelPhase1EfficiencyExtras::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelPhase1EfficiencyExtras::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  iBooker.setCurrentFolder(effFolderName_);

  //Get the existing histos
  MonitorElement* vtx_v_lumi = iGetter.get(vtxFolderName_ + "/NumberOfGoodPVtxVsLS_GenTk");

  MonitorElement* scalLumi_v_lumi = iGetter.get(instLumiFolderName_ + "/lumiVsLS");

  MonitorElement* eff_v_lumi_forward =
      iGetter.get(effFolderName_ + "/hitefficiency_per_Lumisection_per_PXDisk_PXForward");

  MonitorElement* eff_v_lumi_barrel =
      iGetter.get(effFolderName_ + "/hitefficiency_per_Lumisection_per_PXLayer_PXBarrel");

  //set up some booleans that will tell us which graphs to create
  bool createNvtx = true;
  bool createInstLumi = true;

  //check which of the MEs exist and respond appropriately
  if (!eff_v_lumi_forward) {
    edm::LogWarning("SiPixelPhase1EfficiencyExtras")
        << "no hitefficiency_per_Lumisection_per_PXDisk_PXForward ME is available in " << effFolderName_;
    return;
  }
  if (!eff_v_lumi_barrel) {
    edm::LogWarning("SiPixelPhase1EfficiencyExtras")
        << "no hitefficiency_per_Lumisection_per_PXLayer_PXBarrel ME is available in " << effFolderName_;
    return;
  }
  if (!vtx_v_lumi) {
    edm::LogWarning("SiPixelPhase1EfficiencyExtras")
        << "no NumberOfGoodPVtxVsLS_GenTK ME is available in " << vtxFolderName_;
    createNvtx = false;
  }
  if (!scalLumi_v_lumi) {
    edm::LogWarning("SiPixelPhase1EfficiencyExtras") << "no lumiVsLS ME is available in " << instLumiFolderName_;
    createInstLumi = false;
  }

  //If the existing MEs are empty, set the boolean to skip booking
  if (vtx_v_lumi && vtx_v_lumi->getEntries() == 0)
    createNvtx = false;
  if (scalLumi_v_lumi && scalLumi_v_lumi->getEntries() == 0)
    createInstLumi = false;

  // if the max mean lumi is not higher than zero, do not create profiles with respect to lumi
  if (createInstLumi and scalLumi_v_lumi->getTProfile()->GetMaximum() <= 0.)
    createInstLumi = false;

  double eff = 0.0;

  //Will pass if nvtx ME exists and is not empty
  if (createNvtx) {
    //Book new histos
    MonitorElement* eff_v_vtx_barrel =
        iBooker.book2D("hitefficiency_per_meanNvtx_per_PXLayer_PXBarrel",
                       "hitefficiency_per_meanNvtx_per_PXLayer_PXBarrel; meanNvtx; PXLayer",
                       500,
                       0,
                       100,
                       3,
                       .5,
                       3.5);

    MonitorElement* eff_v_vtx_forward =
        iBooker.book2D("hitefficiency_per_meanNvtx_per_PXDisk_PXForward",
                       "hitefficiency_per_meanNvtx_per_PXDisk_PXForward; meanNvtx; PXDisk",
                       500,
                       0,
                       100,
                       7,
                       -3.5,
                       3.5);

    //initialize variables
    int numLumiNvtx = int(vtx_v_lumi->getNbinsX());
    double nvtx = 0.0;
    int binNumVtx = 0;

    //For loop to loop through lumisections
    for (int iLumi = 1; iLumi < numLumiNvtx - 1; iLumi++) {
      //get the meanNvtx for each lumi
      nvtx = vtx_v_lumi->getBinContent(iLumi);

      //Filter out useless iterations
      if (nvtx != 0) {
        //Grab the bin number for the nvtx
        binNumVtx = eff_v_vtx_barrel->getTH2F()->FindBin(nvtx);

        //loop through the layers
        for (int iLayer = 1; iLayer < 8; iLayer++) {
          //get the eff at the lumisection and layer
          eff = eff_v_lumi_forward->getBinContent(iLumi - 1, iLayer);

          //set the efficiency in the new histo
          eff_v_vtx_forward->setBinContent(binNumVtx, iLayer, eff);
        }

        //loop through the layers
        for (int iLayer = 1; iLayer < 5; iLayer++) {
          //get the efficiency for each lumi at each layer
          eff = eff_v_lumi_barrel->getBinContent(iLumi - 1, iLayer);

          //set the efficiency
          eff_v_vtx_barrel->setBinContent(binNumVtx, iLayer, eff);
        }
      }
    }
  }
  // Will pass if InstLumi ME exists, is not empty, and max mean lumi is larger than zero
  if (createInstLumi) {
    //Get the max value of inst lumi for plot (ensuring yMax2 is larger than zero)
    int yMax2 = std::max(1., scalLumi_v_lumi->getTProfile()->GetMaximum());
    yMax2 *= 1.1;

    //Book new histos
    MonitorElement* eff_v_scalLumi_barrel =
        iBooker.book2D("hitefficiency_per_scalLumi_per_PXLayer_PXBarrel",
                       "hitefficiency_per_scalLumi_per_PXLayer_PXBarrel; scal inst lumi E30; PXLayer",
                       500,
                       0,
                       yMax2,
                       3,
                       .5,
                       3.5);

    MonitorElement* eff_v_scalLumi_forward =
        iBooker.book2D("hitefficiency_per_scalLumi_per_PXDisk_PXForward",
                       "hitefficiency_per_scalLumi_per_PXDisk_PXForward; scal inst lumi E30; PXDisk",
                       500,
                       0,
                       yMax2,
                       7,
                       -3.5,
                       3.5);

    //initialize variables
    int numLumiScal = int(scalLumi_v_lumi->getNbinsX());
    double scalLumi = 0.0;
    int binNumScal = 0;

    //For loop to loop through lumisections
    for (int iLumi = 1; iLumi < numLumiScal - 1; iLumi++) {
      //get the inst lumi for each lumi
      scalLumi = scalLumi_v_lumi->getBinContent(iLumi);

      //Filter out useless iterations
      if (scalLumi > 0) {
        //Grab the bin number for the inst lumi
        binNumScal = eff_v_scalLumi_barrel->getTH2F()->FindBin(scalLumi);

        //loop through the layers
        for (int iLayer = 1; iLayer < 8; iLayer++) {
          //get the eff at the lumisection and layer
          eff = eff_v_lumi_forward->getBinContent(iLumi - 1, iLayer);

          //set the efficiency in the new histo
          eff_v_scalLumi_forward->setBinContent(binNumScal, iLayer, eff);
        }

        //loop through the layers
        for (int iLayer = 1; iLayer < 5; iLayer++) {
          //get the eff at the lumisection and layer
          eff = eff_v_lumi_barrel->getBinContent(iLumi - 1, iLayer);

          //set the efficiency in the new histo
          eff_v_scalLumi_barrel->setBinContent(binNumScal, iLayer, eff);
        }
      }
    }
  } else
    return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelPhase1EfficiencyExtras);
