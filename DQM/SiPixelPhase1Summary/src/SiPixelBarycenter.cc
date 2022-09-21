// -*- C++ -*-
//
// Package:    SiPixelPhase1Summary
// Class:      SiPixelBarycenter
//
/**\class 

 Description: Create the pixel barycenter plots

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Danilo Meuser
//         Created:  26th May 2021
//
//
#include "DQM/SiPixelPhase1Summary/interface/SiPixelBarycenter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiPixelPhase1Summary/interface/SiPixelBarycenterHelper.h"

#include <string>
#include <iostream>

using namespace std;
using namespace edm;

SiPixelBarycenter::SiPixelBarycenter(const edm::ParameterSet& iConfig)
    : DQMEDHarvester(iConfig),
      alignmentToken_(esConsumes<edm::Transition::EndRun>()),
      gprToken_(esConsumes<edm::Transition::EndRun>()),
      trackerTopologyToken_(esConsumes<edm::Transition::EndRun>()) {
  LogInfo("PixelDQM") << "SiPixelBarycenter::SiPixelBarycenter: Got DQM BackEnd interface" << endl;
}

void SiPixelBarycenter::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {}

void SiPixelBarycenter::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {}

void SiPixelBarycenter::dqmEndRun(DQMStore::IBooker& iBooker,
                                  DQMStore::IGetter& iGetter,
                                  const edm::Run& iRun,
                                  edm::EventSetup const& c) {
  bookBarycenterHistograms(iBooker);

  const Alignments* alignments = &c.getData(alignmentToken_);
  const Alignments* gpr = &c.getData(gprToken_);
  const TrackerTopology* tTopo = &c.getData(trackerTopologyToken_);

  fillBarycenterHistograms(iBooker, iGetter, alignments->m_align, gpr->m_align, *tTopo);
}
//------------------------------------------------------------------
// Used to book the barycenter histograms
//------------------------------------------------------------------
void SiPixelBarycenter::bookBarycenterHistograms(DQMStore::IBooker& iBooker) {
  iBooker.cd();

  iBooker.setCurrentFolder("PixelPhase1/Barycenter");
  //Book one histogram for each subdetector
  for (const std::string& subdetector : subdetectors_) {
    barycenters_[subdetector] =
        iBooker.book1D("barycenters_" + subdetector,
                       "Position of the barycenter for " + subdetector + ";Coordinate;Position [mm]",
                       3,
                       0.5,
                       3.5);
    barycenters_[subdetector]->setBinLabel(1, "x");
    barycenters_[subdetector]->setBinLabel(2, "y");
    barycenters_[subdetector]->setBinLabel(3, "z");
  }

  //Reset the iBooker
  iBooker.setCurrentFolder("PixelPhase1/");
}

//------------------------------------------------------------------
// Fill the Barycenter histograms
//------------------------------------------------------------------
void SiPixelBarycenter::fillBarycenterHistograms(DQMStore::IBooker& iBooker,
                                                 DQMStore::IGetter& iGetter,
                                                 const std::vector<AlignTransform>& input,
                                                 const std::vector<AlignTransform>& GPR,
                                                 const TrackerTopology& tTopo) {
  const auto GPR_translation_pixel = GPR[0].translation();
  const std::map<DQMBarycenter::coordinate, float> GPR_pixel = {{DQMBarycenter::t_x, GPR_translation_pixel.x()},
                                                                {DQMBarycenter::t_y, GPR_translation_pixel.y()},
                                                                {DQMBarycenter::t_z, GPR_translation_pixel.z()}};

  DQMBarycenter::TkAlBarycenters barycenters;
  barycenters.computeBarycenters(input, tTopo, GPR_pixel);

  auto Xbarycenters = barycenters.getX();
  auto Ybarycenters = barycenters.getY();
  auto Zbarycenters = barycenters.getZ();

  //Fill histogram for each subdetector
  for (std::size_t i = 0; i < subdetectors_.size(); ++i) {
    barycenters_[subdetectors_[i]]->setBinContent(1, Xbarycenters[i]);
    barycenters_[subdetectors_[i]]->setBinContent(2, Ybarycenters[i]);
    barycenters_[subdetectors_[i]]->setBinContent(3, Zbarycenters[i]);

    // zero the errors for better comparison display
    barycenters_[subdetectors_[i]]->setBinError(1, 0.);
    barycenters_[subdetectors_[i]]->setBinError(2, 0.);
    barycenters_[subdetectors_[i]]->setBinError(3, 0.);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelBarycenter);
