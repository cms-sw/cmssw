// -*- C++ -*-
//
// Package:    SiPixelPhase1ResidualsExtra
// Class:      SiPixelPhase1ResidualsExtra
//
/**\class 

 Description: Create the Phsae 1 pixel DRnR plots

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alessandro Rossi
//         Created:  25th May 2021
//
//
#include "DQM/SiPixelPhase1Track/interface/SiPixelPhase1ResidualsExtra.h"
// Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// DQM Framework
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQMServices/Core/interface/DQMStore.h"
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
// DataFormats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
//
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace edm;

SiPixelPhase1ResidualsExtra::SiPixelPhase1ResidualsExtra(const edm::ParameterSet& iConfig)
    : DQMEDHarvester(iConfig), conf_(iConfig) {
  LogInfo("PixelDQM") << "SiPixelPhase1ResidualsExtra::SiPixelPhase1ResidualsExtra: Got DQM BackEnd interface" << endl;
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  minHits_ = conf_.getParameter<int>("MinHits");
}

SiPixelPhase1ResidualsExtra::~SiPixelPhase1ResidualsExtra() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  LogInfo("PixelDQM") << "SiPixelPhase1ResidualsExtra::~SiPixelPhase1ResidualsExtra: Destructor" << endl;
}

void SiPixelPhase1ResidualsExtra::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {}

void SiPixelPhase1ResidualsExtra::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  bookMEs(iBooker);
  fillMEs(iBooker, iGetter);
}

//------------------------------------------------------------------
// Used to book the MEs
//------------------------------------------------------------------
void SiPixelPhase1ResidualsExtra::bookMEs(DQMStore::IBooker& iBooker) {
  iBooker.cd();

  //New residual plots for the PXBarrel separated by inner and outer modules per layer
  iBooker.setCurrentFolder(topFolderName_ + "/PXBarrel");

  for (std::string layer : {"1", "2", "3", "4"}) {
    float mean_range = 20.;
    float rms_range = 200.;
    if (layer == "1") {
      mean_range = 50.;
      rms_range = 1000.;
    }
    residuals_["residual_mean_x_Inner_PXLayer_" + layer] =
        iBooker.book1D("residual_mean_x_Inner_PXLayer_" + layer,
                       "Mean of Track Residuals X Inner Modules for Layer " + layer + ";mean(x_rec-x_pred)[#mum]",
                       100,
                       -1 * mean_range,
                       mean_range);
    residuals_["residual_mean_x_Outer_PXLayer_" + layer] =
        iBooker.book1D("residual_mean_x_Outer_PXLayer_" + layer,
                       "Mean of Track Residuals X Outer Modules for Layer " + layer + ";mean(x_rec-x_pred)[#mum]",
                       100,
                       -1 * mean_range,
                       mean_range);
    residuals_["residual_mean_y_Inner_PXLayer_" + layer] =
        iBooker.book1D("residual_mean_y_Inner_PXLayer_" + layer,
                       "Mean of Track Residuals Y Inner Modules for Layer " + layer + ";mean(y_rec-y_pred)[#mum]",
                       100,
                       -1 * mean_range,
                       mean_range);
    residuals_["residual_mean_y_Outer_PXLayer_" + layer] =
        iBooker.book1D("residual_mean_y_Outer_PXLayer_" + layer,
                       "Mean of Track Residuals Y Outer Modules for Layer " + layer + ";mean(y_rec-y_pred)[#mum]",
                       100,
                       -1 * mean_range,
                       mean_range);

    residuals_["residual_rms_x_Inner_PXLayer_" + layer] =
        iBooker.book1D("residual_rms_x_Inner_PXLayer_" + layer,
                       "RMS of Track Residuals X Inner Modules for Layer " + layer + ";rms(x_rec-x_pred)[#mum]",
                       100,
                       0.,
                       rms_range);
    residuals_["residual_rms_x_Outer_PXLayer_" + layer] =
        iBooker.book1D("residual_rms_x_Outer_PXLayer_" + layer,
                       "RMS of Track Residuals X Outer Modules for Layer " + layer + ";rms(x_rec-x_pred)[#mum]",
                       100,
                       0.,
                       rms_range);
    residuals_["residual_rms_y_Inner_PXLayer_" + layer] =
        iBooker.book1D("residual_rms_y_Inner_PXLayer_" + layer,
                       "RMS of Track Residuals Y Inner Modules for Layer " + layer + ";rms(y_rec-y_pred)[#mum]",
                       100,
                       0.,
                       rms_range);
    residuals_["residual_rms_y_Outer_PXLayer_" + layer] =
        iBooker.book1D("residual_rms_y_Outer_PXLayer_" + layer,
                       "RMS of Track Residuals Y Outer Modules for Layer " + layer + ";rms(y_rec-y_pred)[#mum]",
                       100,
                       0.,
                       rms_range);
    ///Normalized Resiuduals Plots
    DRnR_["NormRes_mean_x_Inner_PXLayer_" + layer] = iBooker.book1D(
        "NormRes_mean_x_Inner_PXLayer_" + layer,
        "Mean of Normalized Track Residuals X Inner Modules for Layer " + layer + ";mean((x_rec-x_pred)/x_err)",
        100,
        -1 * 1,
        1);
    DRnR_["NormRes_mean_x_Outer_PXLayer_" + layer] = iBooker.book1D(
        "NormRes_mean_x_Outer_PXLayer_" + layer,
        "Mean of Normalized Track Residuals X Outer Modules for Layer " + layer + ";mean((x_rec-x_pred)/x_err)",
        100,
        -1 * 1,
        1);
    DRnR_["NormRes_mean_y_Inner_PXLayer_" + layer] = iBooker.book1D(
        "NormRes_mean_y_Inner_PXLayer_" + layer,
        "Mean of Normalized Track Residuals Y Inner Modules for Layer " + layer + ";mean((y_rec-y_pred)/y_err)",
        100,
        -1 * 1,
        1);
    DRnR_["NormRes_mean_y_Outer_PXLayer_" + layer] = iBooker.book1D(
        "NormRes_mean_y_Outer_PXLayer_" + layer,
        "Mean of Normalized Track Residuals Y Outer Modules for Layer " + layer + ";mean((y_rec-y_pred)/y_err)",
        100,
        -1 * 1,
        1);

    DRnR_["DRnR_x_Inner_PXLayer_" + layer] = iBooker.book1D(
        "DRnR_x_Inner_PXLayer_" + layer,
        "RMS of Normalized Track Residuals X Inner Modules for Layer " + layer + ";rms((x_rec-x_pred)/x_err)",
        100,
        0.,
        2);
    DRnR_["DRnR_x_Outer_PXLayer_" + layer] = iBooker.book1D(
        "DRnR_x_Outer_PXLayer_" + layer,
        "RMS of Normalized Track Residuals X Outer Modules for Layer " + layer + ";rms((x_rec-x_pred)/x_err)",
        100,
        0.,
        2);
    DRnR_["DRnR_y_Inner_PXLayer_" + layer] = iBooker.book1D(
        "DRnR_y_Inner_PXLayer_" + layer,
        "RMS of Normalized Track Residuals Y Inner Modules for Layer " + layer + ";rms((y_rec-y_pred)/y_err)",
        100,
        0.,
        2);
    DRnR_["DRnR_y_Outer_PXLayer_" + layer] = iBooker.book1D(
        "DRnR_y_Outer_PXLayer_" + layer,
        "RMS of Normalized Track Residuals Y Outer Modules for Layer " + layer + ";rms((y_rec-y_pred)/y_err)",
        100,
        0.,
        2);
  }

  //New residual plots for the PXForward separated by inner and outer modules
  iBooker.setCurrentFolder(topFolderName_ + "/PXForward");

  residuals_["residual_mean_x_Inner"] = iBooker.book1D(
      "residual_mean_x_Inner", "Mean of Track Residuals X Inner Modules;mean(x_rec-x_pred)[#mum]", 100, -20., 20.);
  residuals_["residual_mean_x_Outer"] = iBooker.book1D(
      "residual_mean_x_Outer", "Mean of Track Residuals X Outer Modules;mean(x_rec-x_pred)[#mum]", 100, -20., 20.);
  residuals_["residual_mean_y_Inner"] = iBooker.book1D(
      "residual_mean_y_Inner", "Mean of Track Residuals Y Inner Modules;mean(y_rec-y_pred)[#mum]", 100, -20., 20.);
  residuals_["residual_mean_y_Outer"] = iBooker.book1D(
      "residual_mean_y_Outer", "Mean of Track Residuals Y Outer Modules;mean(y_rec-y_pred)[#mum]", 100, -20., 20.);

  residuals_["residual_rms_x_Inner"] = iBooker.book1D(
      "residual_rms_x_Inner", "RMS of Track Residuals X Inner Modules;rms(x_rec-x_pred)[#mum]", 100, 0., 200.);
  residuals_["residual_rms_x_Outer"] = iBooker.book1D(
      "residual_rms_x_Outer", "RMS of Track Residuals X Outer Modules;rms(x_rec-x_pred)[#mum]", 100, 0., 200.);
  residuals_["residual_rms_y_Inner"] = iBooker.book1D(
      "residual_rms_y_Inner", "RMS of Track Residuals Y Inner Modules;rms(y_rec-y_pred)[#mum]", 100, 0., 200.);
  residuals_["residual_rms_y_Outer"] = iBooker.book1D(
      "residual_rms_y_Outer", "RMS of Track Residuals Y Outer Modules;rms(y_rec-y_pred)[#mum]", 100, 0., 200.);
  //Normalize Residuals inner/outer
  DRnR_["NormRes_mean_x_Inner"] =
      iBooker.book1D("NormRes_mean_x_Inner",
                     "Mean of Normalized Track Residuals X Inner Modules;mean((x_rec-x_pred)/x_err)",
                     100,
                     -1.,
                     1.);
  DRnR_["NormRes_mean_x_Outer"] =
      iBooker.book1D("NormRes_mean_x_Outer",
                     "Mean of Normalized Track Residuals X Outer Modules;mean((x_rec-x_pred)/x_err)",
                     100,
                     -1.,
                     1.);
  DRnR_["NormRes_mean_y_Inner"] =
      iBooker.book1D("NormRes_mean_y_Inner",
                     "Mean of Normalized Track Residuals Y Inner Modules;mean((y_rec-y_pred)/y_err)",
                     100,
                     -1.,
                     1.);
  DRnR_["NormRes_mean_y_Outer"] =
      iBooker.book1D("NormRes_mean_y_Outer",
                     "Mean of Normalized Track Residuals Y Outer Modules;mean((y_rec-y_pred)/y_err)",
                     100,
                     -1.,
                     1.);

  DRnR_["DRnR_x_Inner"] = iBooker.book1D(
      "DRnR_x_Inner", "RMS of Normalized Track Residuals X Inner Modules;rms((x_rec-x_pred)/x_err)", 100, 0., 2.);
  DRnR_["DRnR_x_Outer"] = iBooker.book1D(
      "DRnR_x_Outer", "RMS of Normalized Track Residuals X Outer Modules;rms((x_rec-x_pred)/x_err)", 100, 0., 2.);
  DRnR_["DRnR_y_Inner"] = iBooker.book1D(
      "DRnR_y_Inner", "RMS of Normalized Track Residuals Y Inner Modules;rms((y_rec-y_pred)/y_err)", 100, 0., 2.);
  DRnR_["DRnR_y_Outer"] = iBooker.book1D(
      "DRnR_y_Outer", "RMS of Normalized Track Residuals Y Outer Modules;rms((y_rec-y_pred)/y_err)", 100, 0., 2.);

  //New residual plots for the PXForward separated by positive and negative side
  iBooker.setCurrentFolder(topFolderName_ + "/PXForward");

  residuals_["residual_mean_x_pos"] = iBooker.book1D(
      "residual_mean_x_pos", "Mean of Track Residuals X pos. Side;mean(x_rec-x_pred)[#mum]", 100, -20., 20.);
  residuals_["residual_mean_x_neg"] = iBooker.book1D(
      "residual_mean_x_neg", "Mean of Track Residuals X neg. Side;mean(x_rec-x_pred)[#mum]", 100, -20., 20.);
  residuals_["residual_mean_y_pos"] = iBooker.book1D(
      "residual_mean_y_pos", "Mean of Track Residuals Y pos. Side;mean(y_rec-y_pred)[#mum]", 100, -20., 20.);
  residuals_["residual_mean_y_neg"] = iBooker.book1D(
      "residual_mean_y_neg", "Mean of Track Residuals Y neg. Side;mean(y_rec-y_pred)[#mum]", 100, -20., 20.);

  residuals_["residual_rms_x_pos"] =
      iBooker.book1D("residual_rms_x_pos", "RMS of Track Residuals X pos. Side;rms(x_rec-x_pred)[#mum]", 100, 0., 200.);
  residuals_["residual_rms_x_neg"] =
      iBooker.book1D("residual_rms_x_neg", "RMS of Track Residuals X neg. Side;rms(x_rec-x_pred)[#mum]", 100, 0., 200.);
  residuals_["residual_rms_y_pos"] =
      iBooker.book1D("residual_rms_y_pos", "RMS of Track Residuals Y pos. Side;rms(y_rec-y_pred)[#mum]", 100, 0., 200.);
  residuals_["residual_rms_y_neg"] =
      iBooker.book1D("residual_rms_y_neg", "RMS of Track Residuals Y neg. Side;rms(y_rec-y_pred)[#mum]", 100, 0., 200.);
  //Normalized Residuals pos/neg
  DRnR_["NormRes_mean_x_pos"] = iBooker.book1D(
      "NormRes_mean_x_pos", "Mean of Normalized Track Residuals X pos. Side;mean((x_rec-x_pred)/x_err)", 100, -1., 1.);
  DRnR_["NormRes_mean_x_neg"] = iBooker.book1D(
      "NormRes_mean_x_neg", "Mean of Normalized Track Residuals X neg. Side;mean((x_rec-x_pred)/x_err)", 100, -1., 1.);
  DRnR_["NormRes_mean_y_pos"] = iBooker.book1D(
      "NormRes_mean_y_pos", "Mean of Normalized Track Residuals Y pos. Side;mean((y_rec-y_pred)/y_err)", 100, -1., 1.);
  DRnR_["NormRes_mean_y_neg"] = iBooker.book1D(
      "NormRes_mean_y_neg", "Mean of Normalized Track Residuals Y neg. Side;mean((y_rec-y_pred)/y_err)", 100, -1., 1.);

  DRnR_["DRnR_x_pos"] = iBooker.book1D(
      "DRnR_x_pos", "RMS of Normalized Track Residuals X pos. Side;rms((x_rec-x_pred)/x_err)", 100, 0., 2.);
  DRnR_["DRnR_x_neg"] = iBooker.book1D(
      "DRnR_x_neg", "RMS of Normalized Track Residuals X neg. Side;rms((x_rec-x_pred)/x_err)", 100, 0., 2.);
  DRnR_["DRnR_y_pos"] = iBooker.book1D(
      "DRnR_y_pos", "RMS of Normalized Track Residuals Y pos. Side;rms((y_rec-y_pred)/y_err)", 100, 0., 2.);
  DRnR_["DRnR_y_neg"] = iBooker.book1D(
      "DRnR_y_neg", "RMS of Normalized Track Residuals Y neg. Side;rms((y_rec-y_pred)/y_err)", 100, 0., 2.);

  //Reset the iBooker
  iBooker.setCurrentFolder("PixelPhase1/");
}

//------------------------------------------------------------------
// Fill the MEs
//------------------------------------------------------------------
void SiPixelPhase1ResidualsExtra::fillMEs(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  //Fill additional residuals plots
  //PXBarrel

  //constexpr int minHits_ = 30;  //Miniminal number of hits needed for module to be filled in histograms

  for (std::string layer : {"1", "2", "3", "4"}) {
    MonitorElement* me_x =
        iGetter.get("PixelPhase1/Tracks/PXBarrel/residual_x_per_SignedModule_per_SignedLadder_PXLayer_" + layer);
    MonitorElement* me_y =
        iGetter.get("PixelPhase1/Tracks/PXBarrel/residual_y_per_SignedModule_per_SignedLadder_PXLayer_" + layer);
    MonitorElement* me2_x = iGetter.get(
        "PixelPhase1/Tracks/ResidualsExtra/PXBarrel/DRnR_x_per_SignedModule_per_SignedLadder_PXLayer_" + layer);
    MonitorElement* me2_y = iGetter.get(
        "PixelPhase1/Tracks/ResidualsExtra/PXBarrel/DRnR_y_per_SignedModule_per_SignedLadder_PXLayer_" + layer);

    if (me_x == nullptr || me_y == nullptr || me2_x == nullptr || me2_y == nullptr) {
      edm::LogWarning("SiPixelPhase1ResidualsExtra")
          << "Residuals plots for Pixel BPIX Layer" << layer << " not found. Skipping ResidualsExtra plots generation.";
      continue;
    }

    for (int i = 1; i <= me_x->getNbinsY(); i++) {
      if (i == (me_x->getNbinsY() / 2 + 1))
        continue;  //Middle bin of y axis is empty

      if (i <= me_x->getNbinsY() / 2) {
        bool iAmInner = (i % 2 == 0);  //Check whether current ladder is inner or outer ladder
        if (iAmInner) {
          for (int j : {1, 2, 3, 4, 6, 7, 8, 9}) {
            if (me_x->getBinEntries(j, i) < minHits_)  //Fill only if number of hits is above threshold
              continue;
            residuals_["residual_mean_x_Inner_PXLayer_" + layer]->Fill(me_x->getBinContent(j, i) * 1e4);
            residuals_["residual_mean_y_Inner_PXLayer_" + layer]->Fill(me_y->getBinContent(j, i) * 1e4);
            residuals_["residual_rms_x_Inner_PXLayer_" + layer]->Fill(me_x->getBinError(j, i) * 1e4);
            residuals_["residual_rms_y_Inner_PXLayer_" + layer]->Fill(me_y->getBinError(j, i) * 1e4);
            DRnR_["NormRes_mean_x_Inner_PXLayer_" + layer]->Fill(me2_x->getBinContent(j, i));
            DRnR_["NormRes_mean_y_Inner_PXLayer_" + layer]->Fill(me2_y->getBinContent(j, i));
            DRnR_["DRnR_x_Inner_PXLayer_" + layer]->Fill(me2_x->getBinError(j, i));
            DRnR_["DRnR_y_Inner_PXLayer_" + layer]->Fill(me2_y->getBinError(j, i));
          }
        } else {
          for (int j : {1, 2, 3, 4, 6, 7, 8, 9}) {
            if (me_x->getBinEntries(j, i) < minHits_)  //Fill only if number of hits is above threshold
              continue;
            residuals_["residual_mean_x_Outer_PXLayer_" + layer]->Fill(me_x->getBinContent(j, i) * 1e4);
            residuals_["residual_mean_y_Outer_PXLayer_" + layer]->Fill(me_y->getBinContent(j, i) * 1e4);
            residuals_["residual_rms_x_Outer_PXLayer_" + layer]->Fill(me_x->getBinError(j, i) * 1e4);
            residuals_["residual_rms_y_Outer_PXLayer_" + layer]->Fill(me_y->getBinError(j, i) * 1e4);
            DRnR_["NormRes_mean_x_Outer_PXLayer_" + layer]->Fill(me2_x->getBinContent(j, i));
            DRnR_["NormRes_mean_y_Outer_PXLayer_" + layer]->Fill(me2_y->getBinContent(j, i));
            DRnR_["DRnR_x_Outer_PXLayer_" + layer]->Fill(me2_x->getBinError(j, i));
            DRnR_["DRnR_y_Outer_PXLayer_" + layer]->Fill(me2_y->getBinError(j, i));
          }
        }
      } else {
        bool iAmInner = (i % 2 == 1);  //Check whether current ladder is inner or outer ladder
        if (iAmInner) {
          for (int j : {1, 2, 3, 4, 6, 7, 8, 9}) {
            if (me_x->getBinEntries(j, i) < minHits_)  //Fill only if number of hits is above threshold
              continue;
            residuals_["residual_mean_x_Inner_PXLayer_" + layer]->Fill(me_x->getBinContent(j, i) * 1e4);
            residuals_["residual_mean_y_Inner_PXLayer_" + layer]->Fill(me_y->getBinContent(j, i) * 1e4);
            residuals_["residual_rms_x_Inner_PXLayer_" + layer]->Fill(me_x->getBinError(j, i) * 1e4);
            residuals_["residual_rms_y_Inner_PXLayer_" + layer]->Fill(me_y->getBinError(j, i) * 1e4);
            DRnR_["NormRes_mean_x_Inner_PXLayer_" + layer]->Fill(me2_x->getBinContent(j, i));
            DRnR_["NormRes_mean_y_Inner_PXLayer_" + layer]->Fill(me2_y->getBinContent(j, i));
            DRnR_["DRnR_x_Inner_PXLayer_" + layer]->Fill(me2_x->getBinError(j, i));
            DRnR_["DRnR_y_Inner_PXLayer_" + layer]->Fill(me2_y->getBinError(j, i));
          }
        } else {
          for (int j : {1, 2, 3, 4, 6, 7, 8, 9}) {
            if (me_x->getBinEntries(j, i) < minHits_)  //Fill only if number of hits is above threshold
              continue;
            residuals_["residual_mean_x_Outer_PXLayer_" + layer]->Fill(me_x->getBinContent(j, i) * 1e4);
            residuals_["residual_mean_y_Outer_PXLayer_" + layer]->Fill(me_y->getBinContent(j, i) * 1e4);
            residuals_["residual_rms_x_Outer_PXLayer_" + layer]->Fill(me_x->getBinError(j, i) * 1e4);
            residuals_["residual_rms_y_Outer_PXLayer_" + layer]->Fill(me_y->getBinError(j, i) * 1e4);
            DRnR_["NormRes_mean_x_Outer_PXLayer_" + layer]->Fill(me2_x->getBinContent(j, i));
            DRnR_["NormRes_mean_y_Outer_PXLayer_" + layer]->Fill(me2_y->getBinContent(j, i));
            DRnR_["DRnR_x_Outer_PXLayer_" + layer]->Fill(me2_x->getBinError(j, i));
            DRnR_["DRnR_y_Outer_PXLayer_" + layer]->Fill(me2_y->getBinError(j, i));
          }
        }
      }
      for (int j : {1, 2, 3, 4, 6, 7, 8, 9}) {
        if (me_x->getBinEntries(j, i) < minHits_) {
          me2_x->setBinContent(j, i, 0);
          me2_y->setBinContent(j, i, 0);
          me2_x->setBinEntries(me2_x->getBin(j, i), 0);
          me2_y->setBinEntries(me2_y->getBin(j, i), 0);
        } else {
          me2_x->setBinContent(j, i, me2_x->getBinError(j, i));
          me2_y->setBinContent(j, i, me2_y->getBinError(j, i));
          me2_x->setBinEntries(me2_x->getBin(j, i), 1);
          me2_y->setBinEntries(me2_y->getBin(j, i), 1);
        }
      }
    }
  }

  //PXForward separating outer and inner modules as well as positive and negative side
  for (std::string ring : {"1", "2"}) {
    MonitorElement* me_x =
        iGetter.get("PixelPhase1/Tracks/PXForward/residual_x_per_PXDisk_per_SignedBladePanel_PXRing_" + ring);
    MonitorElement* me_y =
        iGetter.get("PixelPhase1/Tracks/PXForward/residual_y_per_PXDisk_per_SignedBladePanel_PXRing_" + ring);
    MonitorElement* me2_x = iGetter.get(
        "PixelPhase1/Tracks/ResidualsExtra/PXForward/DRnR_x_per_PXDisk_per_SignedBladePanel_PXRing_" + ring);
    MonitorElement* me2_y = iGetter.get(
        "PixelPhase1/Tracks/ResidualsExtra/PXForward/DRnR_y_per_PXDisk_per_SignedBladePanel_PXRing_" + ring);

    if (me_x == nullptr || me_y == nullptr || me2_x == nullptr || me2_y == nullptr) {
      edm::LogWarning("SiPixelPhase1ResidualsExtra")
          << "Residuals plots for Pixel FPIX Ring" << ring << " not found. Skipping ResidualsExtra plots generation.";
      continue;
    }

    bool posSide = false;
    for (int j = 1; j <= me_x->getNbinsX(); j++) {
      if (j == 4)
        continue;  //fourth x-bin in profile plots is empty

      if (j == 5)
        posSide = true;  //change to postive side

      for (int i = 1; i <= me_x->getNbinsY(); i++) {
        if (i == me_x->getNbinsY() / 2)
          continue;  //Middle bins of y axis is empty
        if (i == (me_x->getNbinsY() / 2) + 1)
          continue;
        if (me_x->getBinEntries(j, i) >= minHits_) {  //Fill only if number of hits is above threshold

          bool iAmInner = (i % 2 == 0);  //separate inner and outer modules
          if (iAmInner) {
            residuals_["residual_mean_x_Inner"]->Fill(me_x->getBinContent(j, i) * 1e4);
            residuals_["residual_mean_y_Inner"]->Fill(me_y->getBinContent(j, i) * 1e4);
            residuals_["residual_rms_x_Inner"]->Fill(me_x->getBinError(j, i) * 1e4);
            residuals_["residual_rms_y_Inner"]->Fill(me_y->getBinError(j, i) * 1e4);
            DRnR_["NormRes_mean_x_Inner"]->Fill(me2_x->getBinContent(j, i));
            DRnR_["NormRes_mean_y_Inner"]->Fill(me2_y->getBinContent(j, i));
            DRnR_["DRnR_x_Inner"]->Fill(me2_x->getBinError(j, i));
            DRnR_["DRnR_y_Inner"]->Fill(me2_y->getBinError(j, i));
          } else {
            residuals_["residual_mean_x_Outer"]->Fill(me_x->getBinContent(j, i) * 1e4);
            residuals_["residual_mean_y_Outer"]->Fill(me_y->getBinContent(j, i) * 1e4);
            residuals_["residual_rms_x_Outer"]->Fill(me_x->getBinError(j, i) * 1e4);
            residuals_["residual_rms_y_Outer"]->Fill(me_y->getBinError(j, i) * 1e4);
            DRnR_["NormRes_mean_x_Outer"]->Fill(me2_x->getBinContent(j, i));
            DRnR_["NormRes_mean_y_Outer"]->Fill(me2_y->getBinContent(j, i));
            DRnR_["DRnR_x_Outer"]->Fill(me2_x->getBinError(j, i));
            DRnR_["DRnR_y_Outer"]->Fill(me2_y->getBinError(j, i));
          }

          if (!posSide) {  //separate postive and negative side
            residuals_["residual_mean_x_neg"]->Fill(me_x->getBinContent(j, i) * 1e4);
            residuals_["residual_mean_y_neg"]->Fill(me_y->getBinContent(j, i) * 1e4);
            residuals_["residual_rms_x_neg"]->Fill(me_x->getBinError(j, i) * 1e4);
            residuals_["residual_rms_y_neg"]->Fill(me_y->getBinError(j, i) * 1e4);
            DRnR_["NormRes_mean_x_neg"]->Fill(me2_x->getBinContent(j, i));
            DRnR_["NormRes_mean_y_neg"]->Fill(me2_y->getBinContent(j, i));
            DRnR_["DRnR_x_neg"]->Fill(me2_x->getBinError(j, i));
            DRnR_["DRnR_y_neg"]->Fill(me2_y->getBinError(j, i));
          } else {
            residuals_["residual_mean_x_pos"]->Fill(me_x->getBinContent(j, i) * 1e4);
            residuals_["residual_mean_y_pos"]->Fill(me_y->getBinContent(j, i) * 1e4);
            residuals_["residual_rms_x_pos"]->Fill(me_x->getBinError(j, i) * 1e4);
            residuals_["residual_rms_y_pos"]->Fill(me_y->getBinError(j, i) * 1e4);
            DRnR_["NormRes_mean_x_pos"]->Fill(me2_x->getBinContent(j, i));
            DRnR_["NormRes_mean_y_pos"]->Fill(me2_y->getBinContent(j, i));
            DRnR_["DRnR_x_pos"]->Fill(me2_x->getBinError(j, i));
            DRnR_["DRnR_y_pos"]->Fill(me2_y->getBinError(j, i));
          }
          me2_x->setBinContent(j, i, me2_x->getBinError(j, i));
          me2_y->setBinContent(j, i, me2_y->getBinError(j, i));
          me2_x->setBinEntries(me2_x->getBin(j, i), 1);
          me2_y->setBinEntries(me2_y->getBin(j, i), 1);
        } else {
          me2_x->setBinContent(j, i, 0);
          me2_y->setBinContent(j, i, 0);
          me2_x->setBinEntries(me2_x->getBin(j, i), 0);
          me2_y->setBinEntries(me2_y->getBin(j, i), 0);
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelPhase1ResidualsExtra);
