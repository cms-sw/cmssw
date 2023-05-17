/*
 *  See header file for a description of this class.
 *
 *  \author Suchandra Dutta , Giorgia Mila
 */

#include "DQM/TrackingMonitor/interface/TrackSplittingMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include <string>

TrackSplittingMonitor::TrackSplittingMonitor(const edm::ParameterSet& iConfig)
    : conf_(iConfig),
      mfToken_(esConsumes()),
      tkGeomToken_(esConsumes()),
      dtGeomToken_(esConsumes()),
      cscGeomToken_(esConsumes()),
      rpcGeomToken_(esConsumes()),
      splitTracksToken_(consumes<std::vector<reco::Track> >(conf_.getParameter<edm::InputTag>("splitTrackCollection"))),
      splitMuonsToken_(mayConsume<std::vector<reco::Muon> >(conf_.getParameter<edm::InputTag>("splitMuonCollection"))),
      plotMuons_(conf_.getParameter<bool>("ifPlotMuons")),
      pixelHitsPerLeg_(conf_.getParameter<int>("pixelHitsPerLeg")),
      totalHitsPerLeg_(conf_.getParameter<int>("totalHitsPerLeg")),
      d0Cut_(conf_.getParameter<double>("d0Cut")),
      dzCut_(conf_.getParameter<double>("dzCut")),
      ptCut_(conf_.getParameter<double>("ptCut")),
      norchiCut_(conf_.getParameter<double>("norchiCut")) {}

void TrackSplittingMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                           edm::Run const& /* iRun */,
                                           edm::EventSetup const& /* iSetup */) {
  std::string MEFolderName = conf_.getParameter<std::string>("FolderName");
  ibooker.setCurrentFolder(MEFolderName);

  // bin declarations
  int ddxyBin = conf_.getParameter<int>("ddxyBin");
  double ddxyMin = conf_.getParameter<double>("ddxyMin");
  double ddxyMax = conf_.getParameter<double>("ddxyMax");

  int ddzBin = conf_.getParameter<int>("ddzBin");
  double ddzMin = conf_.getParameter<double>("ddzMin");
  double ddzMax = conf_.getParameter<double>("ddzMax");

  int dphiBin = conf_.getParameter<int>("dphiBin");
  double dphiMin = conf_.getParameter<double>("dphiMin");
  double dphiMax = conf_.getParameter<double>("dphiMax");

  int dthetaBin = conf_.getParameter<int>("dthetaBin");
  double dthetaMin = conf_.getParameter<double>("dthetaMin");
  double dthetaMax = conf_.getParameter<double>("dthetaMax");

  int dptBin = conf_.getParameter<int>("dptBin");
  double dptMin = conf_.getParameter<double>("dptMin");
  double dptMax = conf_.getParameter<double>("dptMax");

  int dcurvBin = conf_.getParameter<int>("dcurvBin");
  double dcurvMin = conf_.getParameter<double>("dcurvMin");
  double dcurvMax = conf_.getParameter<double>("dcurvMax");

  int normBin = conf_.getParameter<int>("normBin");
  double normMin = conf_.getParameter<double>("normMin");
  double normMax = conf_.getParameter<double>("normMax");

  // declare histogram
  ddxyAbsoluteResiduals_tracker_ =
      ibooker.book1D("ddxyAbsoluteResiduals_tracker", "ddxyAbsoluteResiduals_tracker", ddxyBin, ddxyMin, ddxyMax);
  ddzAbsoluteResiduals_tracker_ =
      ibooker.book1D("ddzAbsoluteResiduals_tracker", "ddzAbsoluteResiduals_tracker", ddzBin, ddzMin, ddzMax);
  dphiAbsoluteResiduals_tracker_ =
      ibooker.book1D("dphiAbsoluteResiduals_tracker", "dphiAbsoluteResiduals_tracker", dphiBin, dphiMin, dphiMax);
  dthetaAbsoluteResiduals_tracker_ = ibooker.book1D(
      "dthetaAbsoluteResiduals_tracker", "dthetaAbsoluteResiduals_tracker", dthetaBin, dthetaMin, dthetaMax);
  dptAbsoluteResiduals_tracker_ =
      ibooker.book1D("dptAbsoluteResiduals_tracker", "dptAbsoluteResiduals_tracker", dptBin, dptMin, dptMax);
  dcurvAbsoluteResiduals_tracker_ =
      ibooker.book1D("dcurvAbsoluteResiduals_tracker", "dcurvAbsoluteResiduals_tracker", dcurvBin, dcurvMin, dcurvMax);

  ddxyNormalizedResiduals_tracker_ =
      ibooker.book1D("ddxyNormalizedResiduals_tracker", "ddxyNormalizedResiduals_tracker", normBin, normMin, normMax);
  ddzNormalizedResiduals_tracker_ =
      ibooker.book1D("ddzNormalizedResiduals_tracker", "ddzNormalizedResiduals_tracker", normBin, normMin, normMax);
  dphiNormalizedResiduals_tracker_ =
      ibooker.book1D("dphiNormalizedResiduals_tracker", "dphiNormalizedResiduals_tracker", normBin, normMin, normMax);
  dthetaNormalizedResiduals_tracker_ = ibooker.book1D(
      "dthetaNormalizedResiduals_tracker", "dthetaNormalizedResiduals_tracker", normBin, normMin, normMax);
  dptNormalizedResiduals_tracker_ =
      ibooker.book1D("dptNormalizedResiduals_tracker", "dptNormalizedResiduals_tracker", normBin, normMin, normMax);
  dcurvNormalizedResiduals_tracker_ =
      ibooker.book1D("dcurvNormalizedResiduals_tracker", "dcurvNormalizedResiduals_tracker", normBin, normMin, normMax);

  if (plotMuons_) {
    ddxyAbsoluteResiduals_global_ =
        ibooker.book1D("ddxyAbsoluteResiduals_global", "ddxyAbsoluteResiduals_global", ddxyBin, ddxyMin, ddxyMax);
    ddzAbsoluteResiduals_global_ =
        ibooker.book1D("ddzAbsoluteResiduals_global", "ddzAbsoluteResiduals_global", ddzBin, ddzMin, ddzMax);
    dphiAbsoluteResiduals_global_ =
        ibooker.book1D("dphiAbsoluteResiduals_global", "dphiAbsoluteResiduals_global", dphiBin, dphiMin, dphiMax);
    dthetaAbsoluteResiduals_global_ = ibooker.book1D(
        "dthetaAbsoluteResiduals_global", "dthetaAbsoluteResiduals_global", dthetaBin, dthetaMin, dthetaMax);
    dptAbsoluteResiduals_global_ =
        ibooker.book1D("dptAbsoluteResiduals_global", "dptAbsoluteResiduals_global", dptBin, dptMin, dptMax);
    dcurvAbsoluteResiduals_global_ =
        ibooker.book1D("dcurvAbsoluteResiduals_global", "dcurvAbsoluteResiduals_global", dcurvBin, dcurvMin, dcurvMax);

    ddxyNormalizedResiduals_global_ =
        ibooker.book1D("ddxyNormalizedResiduals_global", "ddxyNormalizedResiduals_global", normBin, normMin, normMax);
    ddzNormalizedResiduals_global_ =
        ibooker.book1D("ddzNormalizedResiduals_global", "ddzNormalizedResiduals_global", normBin, normMin, normMax);
    dphiNormalizedResiduals_global_ =
        ibooker.book1D("dphiNormalizedResiduals_global", "dphiNormalizedResiduals_global", normBin, normMin, normMax);
    dthetaNormalizedResiduals_global_ = ibooker.book1D(
        "dthetaNormalizedResiduals_global", "dthetaNormalizedResiduals_global", normBin, normMin, normMax);
    dptNormalizedResiduals_global_ =
        ibooker.book1D("dptNormalizedResiduals_global", "dptNormalizedResiduals_global", normBin, normMin, normMax);
    dcurvNormalizedResiduals_global_ =
        ibooker.book1D("dcurvNormalizedResiduals_global", "dcurvNormalizedResiduals_global", normBin, normMin, normMax);
  }

  ddxyAbsoluteResiduals_tracker_->setAxisTitle("(#delta d_{xy})/#sqrt{2} [#mum]");
  ddzAbsoluteResiduals_tracker_->setAxisTitle("(#delta d_{z})/#sqrt{2} [#mum]");
  dphiAbsoluteResiduals_tracker_->setAxisTitle("(#delta #phi)/#sqrt{2} [mrad]");
  dthetaAbsoluteResiduals_tracker_->setAxisTitle("(#delta #theta)/#sqrt{2} [mrad]");
  dptAbsoluteResiduals_tracker_->setAxisTitle("(#delta p_{T})/#sqrt{2} [GeV]");
  dcurvAbsoluteResiduals_tracker_->setAxisTitle("(#delta (1/p_{T}))/#sqrt{2} [GeV^{-1}]");

  ddxyNormalizedResiduals_tracker_->setAxisTitle("#delta d_{xy}/#sigma(d_{xy})");
  ddzNormalizedResiduals_tracker_->setAxisTitle("#delta d_{z}/#sigma(d_{z})");
  dphiNormalizedResiduals_tracker_->setAxisTitle("#delta #phi/#sigma(d_{#phi})");
  dthetaNormalizedResiduals_tracker_->setAxisTitle("#delta #theta/#sigma(d_{#theta})");
  dptNormalizedResiduals_tracker_->setAxisTitle("#delta p_{T}/#sigma(p_{T})");
  dcurvNormalizedResiduals_tracker_->setAxisTitle("#delta 1/p_{T}/#sigma(1/p_{T})");

  if (plotMuons_) {
    ddxyAbsoluteResiduals_global_->setAxisTitle("(#delta d_{xy})/#sqrt{2} [#mum]");
    ddzAbsoluteResiduals_global_->setAxisTitle("(#delta d_{z})/#sqrt{2} [#mum]");
    dphiAbsoluteResiduals_global_->setAxisTitle("(#delta #phi)/#sqrt{2} [mrad]");
    dthetaAbsoluteResiduals_global_->setAxisTitle("(#delta #theta)/#sqrt{2} [mrad]");
    dptAbsoluteResiduals_global_->setAxisTitle("(#delta p_{T})/#sqrt{2} [GeV]");
    dcurvAbsoluteResiduals_global_->setAxisTitle("(#delta (1/p_{T}))/#sqrt{2} [GeV^{-1}]");

    ddxyNormalizedResiduals_global_->setAxisTitle("#delta d_{xy}/#sigma(d_{xy})");
    ddzNormalizedResiduals_global_->setAxisTitle("#delta d_{z}/#sigma(d_{z})");
    dphiNormalizedResiduals_global_->setAxisTitle("#delta #phi/#sigma(d_{#phi})");
    dthetaNormalizedResiduals_global_->setAxisTitle("#delta #theta/#sigma(d_{#theta})");
    dptNormalizedResiduals_global_->setAxisTitle("#delta p_{T}/#sigma(p_{T})");
    dcurvNormalizedResiduals_global_->setAxisTitle("#delta 1/p_{T}/#sigma(1/p_{T})");
  }
}

//
// -- Analyse
//
void TrackSplittingMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  theMagField = &iSetup.getData(mfToken_);
  theGeometry = &iSetup.getData(tkGeomToken_);
  dtGeometry = &iSetup.getData(dtGeomToken_);
  cscGeometry = &iSetup.getData(cscGeomToken_);
  rpcGeometry = &iSetup.getData(rpcGeomToken_);

  edm::Handle<std::vector<reco::Track> > splitTracks = iEvent.getHandle(splitTracksToken_);
  if (!splitTracks.isValid())
    return;

  edm::Handle<std::vector<reco::Muon> > splitMuons;
  if (plotMuons_) {
    splitMuons = iEvent.getHandle(splitMuonsToken_);
  }

  if (splitTracks->size() == 2) {
    // check that there are 2 tracks in split track collection
    edm::LogInfo("TrackSplittingMonitor") << "Split Track size: " << splitTracks->size();

    // split tracks calculations
    reco::Track track1 = splitTracks->at(0);
    reco::Track track2 = splitTracks->at(1);

    // -------------------------- basic selection ---------------------------

    // hit counting
    // looping through the hits for track 1
    double nRechits1 = 0;
    double nRechitinBPIX1 = 0;
    for (auto const& iHit : track1.recHits()) {
      if (iHit->isValid()) {
        nRechits1++;
        int type = iHit->geographicalId().subdetId();
        if (type == int(PixelSubdetector::PixelBarrel)) {
          ++nRechitinBPIX1;
        }
      }
    }
    // looping through the hits for track 2
    double nRechits2 = 0;
    double nRechitinBPIX2 = 0;
    for (auto const& iHit : track2.recHits()) {
      if (iHit->isValid()) {
        nRechits2++;
        int type = iHit->geographicalId().subdetId();
        if (type == int(PixelSubdetector::PixelBarrel)) {
          ++nRechitinBPIX2;
        }
      }
    }

    // DCA of each track
    double d01 = track1.d0();
    double dz1 = track1.dz();
    double d02 = track2.d0();
    double dz2 = track2.dz();

    // pT of each track
    double pt1 = track1.pt();
    double pt2 = track2.pt();

    // chi2 of each track
    double norchi1 = track1.normalizedChi2();
    double norchi2 = track2.normalizedChi2();

    // basic selection
    // pixel hits and total hits
    if ((nRechitinBPIX1 >= pixelHitsPerLeg_) && (nRechitinBPIX1 >= pixelHitsPerLeg_) &&
        (nRechits1 >= totalHitsPerLeg_) && (nRechits2 >= totalHitsPerLeg_)) {
      // dca cut
      if (((std::abs(d01) < d0Cut_)) && (std::abs(d02) < d0Cut_) && (std::abs(dz1) < dzCut_) &&
          (std::abs(dz2) < dzCut_)) {
        // pt cut
        if ((pt1 + pt2) / 2 < ptCut_) {
          // chi2 cut
          if ((norchi1 < norchiCut_) && (norchi2 < norchiCut_)) {
            // passed all cuts...
            edm::LogInfo("TrackSplittingMonitor") << " Setected after all cuts ?";

            double ddxyVal = d01 - d02;
            double ddzVal = dz1 - dz2;
            double dphiVal = track1.phi() - track2.phi();
            double dthetaVal = track1.theta() - track2.theta();
            double dptVal = pt1 - pt2;
            double dcurvVal = (1 / pt1) - (1 / pt2);

            double d01ErrVal = track1.d0Error();
            double d02ErrVal = track2.d0Error();
            double dz1ErrVal = track1.dzError();
            double dz2ErrVal = track2.dzError();
            double phi1ErrVal = track1.phiError();
            double phi2ErrVal = track2.phiError();
            double theta1ErrVal = track1.thetaError();
            double theta2ErrVal = track2.thetaError();
            double pt1ErrVal = track1.ptError();
            double pt2ErrVal = track2.ptError();

            ddxyAbsoluteResiduals_tracker_->Fill(cmToUm * ddxyVal / sqrt2);
            ddzAbsoluteResiduals_tracker_->Fill(cmToUm * ddzVal / sqrt2);
            dphiAbsoluteResiduals_tracker_->Fill(radToUrad * dphiVal / sqrt2);
            dthetaAbsoluteResiduals_tracker_->Fill(radToUrad * dthetaVal / sqrt2);
            dptAbsoluteResiduals_tracker_->Fill(dptVal / sqrt2);
            dcurvAbsoluteResiduals_tracker_->Fill(dcurvVal / sqrt2);

            ddxyNormalizedResiduals_tracker_->Fill(ddxyVal / sqrt(d01ErrVal * d01ErrVal + d02ErrVal * d02ErrVal));
            ddzNormalizedResiduals_tracker_->Fill(ddzVal / sqrt(dz1ErrVal * dz1ErrVal + dz2ErrVal * dz2ErrVal));
            dphiNormalizedResiduals_tracker_->Fill(dphiVal / sqrt(phi1ErrVal * phi1ErrVal + phi2ErrVal * phi2ErrVal));
            dthetaNormalizedResiduals_tracker_->Fill(dthetaVal /
                                                     sqrt(theta1ErrVal * theta1ErrVal + theta2ErrVal * theta2ErrVal));
            dptNormalizedResiduals_tracker_->Fill(dptVal / sqrt(pt1ErrVal * pt1ErrVal + pt2ErrVal * pt2ErrVal));
            dcurvNormalizedResiduals_tracker_->Fill(
                dcurvVal / sqrt(pow(pt1ErrVal, 2) / pow(pt1, 4) + pow(pt2ErrVal, 2) / pow(pt2, 4)));

            // if do the same for split muons
            if (plotMuons_ && splitMuons.isValid()) {
              int gmCtr = 0;
              bool topGlobalMuonFlag = false;
              bool bottomGlobalMuonFlag = false;
              int topGlobalMuon = -1;
              int bottomGlobalMuon = -1;
              double topGlobalMuonNorchi2 = 1e10;
              double bottomGlobalMuonNorchi2 = 1e10;

              // check if usable split global muons
              for (std::vector<reco::Muon>::const_iterator gmI = splitMuons->begin(); gmI != splitMuons->end(); gmI++) {
                if (gmI->isTrackerMuon() && gmI->isStandAloneMuon() && gmI->isGlobalMuon()) {
                  reco::TrackRef trackerTrackRef1(splitTracks, 0);
                  reco::TrackRef trackerTrackRef2(splitTracks, 1);

                  if (gmI->innerTrack() == trackerTrackRef1) {
                    if (gmI->globalTrack()->normalizedChi2() < topGlobalMuonNorchi2) {
                      topGlobalMuonFlag = true;
                      topGlobalMuonNorchi2 = gmI->globalTrack()->normalizedChi2();
                      topGlobalMuon = gmCtr;
                    }
                  }
                  if (gmI->innerTrack() == trackerTrackRef2) {
                    if (gmI->globalTrack()->normalizedChi2() < bottomGlobalMuonNorchi2) {
                      bottomGlobalMuonFlag = true;
                      bottomGlobalMuonNorchi2 = gmI->globalTrack()->normalizedChi2();
                      bottomGlobalMuon = gmCtr;
                    }
                  }
                }
                gmCtr++;
              }

              if (bottomGlobalMuonFlag && topGlobalMuonFlag) {
                reco::Muon muonTop = splitMuons->at(topGlobalMuon);
                reco::Muon muonBottom = splitMuons->at(bottomGlobalMuon);

                reco::TrackRef glb1 = muonTop.globalTrack();
                reco::TrackRef glb2 = muonBottom.globalTrack();

                double ddxyValGlb = glb1->d0() - glb2->d0();
                double ddzValGlb = glb1->dz() - glb2->dz();
                double dphiValGlb = glb1->phi() - glb2->phi();
                double dthetaValGlb = glb1->theta() - glb2->theta();
                double dptValGlb = glb1->pt() - glb2->pt();
                double dcurvValGlb = (1 / glb1->pt()) - (1 / glb2->pt());

                double d01ErrValGlb = glb1->d0Error();
                double d02ErrValGlb = glb2->d0Error();
                double dz1ErrValGlb = glb1->dzError();
                double dz2ErrValGlb = glb2->dzError();
                double phi1ErrValGlb = glb1->phiError();
                double phi2ErrValGlb = glb2->phiError();
                double theta1ErrValGlb = glb1->thetaError();
                double theta2ErrValGlb = glb2->thetaError();
                double pt1ErrValGlb = glb1->ptError();
                double pt2ErrValGlb = glb2->ptError();

                ddxyAbsoluteResiduals_global_->Fill(cmToUm * ddxyValGlb / sqrt2);
                ddzAbsoluteResiduals_global_->Fill(cmToUm * ddzValGlb / sqrt2);
                dphiAbsoluteResiduals_global_->Fill(radToUrad * dphiValGlb / sqrt2);
                dthetaAbsoluteResiduals_global_->Fill(radToUrad * dthetaValGlb / sqrt2);
                dptAbsoluteResiduals_global_->Fill(dptValGlb / sqrt2);
                dcurvAbsoluteResiduals_global_->Fill(dcurvValGlb / sqrt2);

                ddxyNormalizedResiduals_global_->Fill(ddxyValGlb /
                                                      sqrt(d01ErrValGlb * d01ErrValGlb + d02ErrValGlb * d02ErrValGlb));
                ddxyNormalizedResiduals_global_->Fill(ddzValGlb /
                                                      sqrt(dz1ErrValGlb * dz1ErrValGlb + dz2ErrValGlb * dz2ErrValGlb));
                ddxyNormalizedResiduals_global_->Fill(
                    dphiValGlb / sqrt(phi1ErrValGlb * phi1ErrValGlb + phi2ErrValGlb * phi2ErrValGlb));
                ddxyNormalizedResiduals_global_->Fill(
                    dthetaValGlb / sqrt(theta1ErrValGlb * theta1ErrValGlb + theta2ErrValGlb * theta2ErrValGlb));
                ddxyNormalizedResiduals_global_->Fill(dptValGlb /
                                                      sqrt(pt1ErrValGlb * pt1ErrValGlb + pt2ErrValGlb * pt2ErrValGlb));
                ddxyNormalizedResiduals_global_->Fill(
                    dcurvValGlb / sqrt(pow(pt1ErrValGlb, 2) / pow(pt1, 4) + pow(pt2ErrValGlb, 2) / pow(pt2, 4)));
              }

            }  // end of split muons loop
          }
        }
      }
    }
  }
}

void TrackSplittingMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment(
      "Validates track parameters resolution by splitting cosmics tracks at the PCA and comparing the parameters of "
      "the two halves");
  desc.add<std::string>("FolderName", "TrackSplitMonitoring");
  desc.add<edm::InputTag>("splitTrackCollection", edm::InputTag("splittedTracksP5"));
  desc.add<edm::InputTag>("splitMuonCollection", edm::InputTag("splitMuons"));
  desc.add<bool>("ifPlotMuons", true);
  desc.add<int>("pixelHitsPerLeg", 1);
  desc.add<int>("totalHitsPerLeg", 6);
  desc.add<double>("d0Cut", 12.0);
  desc.add<double>("dzCut", 25.0);
  desc.add<double>("ptCut", 4.0);
  desc.add<double>("norchiCut", 100.0);
  desc.add<int>("ddxyBin", 100);
  desc.add<double>("ddxyMin", -200.0);
  desc.add<double>("ddxyMax", 200.0);
  desc.add<int>("ddzBin", 100);
  desc.add<double>("ddzMin", -400.0);
  desc.add<double>("ddzMax", 400.0);
  desc.add<int>("dphiBin", 100);
  desc.add<double>("dphiMin", -0.01);
  desc.add<double>("dphiMax", 0.01);
  desc.add<int>("dthetaBin", 100);
  desc.add<double>("dthetaMin", -0.01);
  desc.add<double>("dthetaMax", 0.01);
  desc.add<int>("dptBin", 100);
  desc.add<double>("dptMin", -5.0);
  desc.add<double>("dptMax", 5.0);
  desc.add<int>("dcurvBin", 100);
  desc.add<double>("dcurvMin", -0.005);
  desc.add<double>("dcurvMax", 0.005);
  desc.add<int>("normBin", 100);
  desc.add<double>("normMin", -5.);
  desc.add<double>("normMax", 5.);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(TrackSplittingMonitor);
