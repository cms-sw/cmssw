/*
 *  See header file for a description of this class.
 *
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DQMOffline/Alignment/interface/TkAlCaRecoMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

#include <string>
#include "TLorentzVector.h"

TkAlCaRecoMonitor::TkAlCaRecoMonitor(const edm::ParameterSet &iConfig)
    : tkGeomToken_(esConsumes()), mfToken_(esConsumes()) {
  conf_ = iConfig;
  trackProducer_ = consumes<reco::TrackCollection>(conf_.getParameter<edm::InputTag>("TrackProducer"));
  referenceTrackProducer_ =
      consumes<reco::TrackCollection>(conf_.getParameter<edm::InputTag>("ReferenceTrackProducer"));
  jetCollection_ = mayConsume<reco::CaloJetCollection>(conf_.getParameter<edm::InputTag>("CaloJetCollection"));
}

TkAlCaRecoMonitor::~TkAlCaRecoMonitor() {}

void TkAlCaRecoMonitor::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &, edm::EventSetup const &) {
  std::string histname;  // for naming the histograms according to algorithm used

  std::string AlgoName = conf_.getParameter<std::string>("AlgoName");
  std::string MEFolderName = conf_.getParameter<std::string>("FolderName");

  daughterMass_ = conf_.getParameter<double>("daughterMass");

  maxJetPt_ = conf_.getParameter<double>("maxJetPt");

  iBooker.setCurrentFolder(MEFolderName + "/TkAlignmentSpecific");
  fillInvariantMass_ = conf_.getParameter<bool>("fillInvariantMass");
  runsOnReco_ = conf_.getParameter<bool>("runsOnReco");
  useSignedR_ = conf_.getParameter<bool>("useSignedR");
  fillRawIdMap_ = conf_.getParameter<bool>("fillRawIdMap");

  //
  unsigned int MassBin = conf_.getParameter<unsigned int>("MassBin");
  double MassMin = conf_.getParameter<double>("MassMin");
  double MassMax = conf_.getParameter<double>("MassMax");

  if (fillInvariantMass_) {
    histname = "InvariantMass_";
    invariantMass_ = iBooker.book1D(histname + AlgoName, histname + AlgoName, MassBin, MassMin, MassMax);
    invariantMass_->setAxisTitle("invariant Mass / GeV");
  } else {
    invariantMass_ = nullptr;
  }

  unsigned int TrackPtPositiveBin = conf_.getParameter<unsigned int>("TrackPtBin");
  double TrackPtPositiveMin = conf_.getParameter<double>("TrackPtMin");
  double TrackPtPositiveMax = conf_.getParameter<double>("TrackPtMax");

  histname = "TrackPtPositive_";
  TrackPtPositive_ = iBooker.book1D(
      histname + AlgoName, histname + AlgoName, TrackPtPositiveBin, TrackPtPositiveMin, TrackPtPositiveMax);
  TrackPtPositive_->setAxisTitle("p_{T} of tracks charge > 0");

  unsigned int TrackPtNegativeBin = conf_.getParameter<unsigned int>("TrackPtBin");
  double TrackPtNegativeMin = conf_.getParameter<double>("TrackPtMin");
  double TrackPtNegativeMax = conf_.getParameter<double>("TrackPtMax");

  histname = "TrackPtNegative_";
  TrackPtNegative_ = iBooker.book1D(
      histname + AlgoName, histname + AlgoName, TrackPtNegativeBin, TrackPtNegativeMin, TrackPtNegativeMax);
  TrackPtNegative_->setAxisTitle("p_{T} of tracks charge < 0");

  histname = "TrackQuality_";
  TrackQuality_ = iBooker.book1D(
      histname + AlgoName, histname + AlgoName, reco::TrackBase::qualitySize, -0.5, reco::TrackBase::qualitySize - 0.5);
  TrackQuality_->setAxisTitle("quality");
  for (int i = 0; i < reco::TrackBase::qualitySize; ++i) {
    TrackQuality_->getTH1()->GetXaxis()->SetBinLabel(
        i + 1, reco::TrackBase::qualityName(reco::TrackBase::TrackQuality(i)).c_str());
  }

  unsigned int SumChargeBin = conf_.getParameter<unsigned int>("SumChargeBin");
  double SumChargeMin = conf_.getParameter<double>("SumChargeMin");
  double SumChargeMax = conf_.getParameter<double>("SumChargeMax");

  histname = "SumCharge_";
  sumCharge_ = iBooker.book1D(histname + AlgoName, histname + AlgoName, SumChargeBin, SumChargeMin, SumChargeMax);
  sumCharge_->setAxisTitle("#SigmaCharge");

  unsigned int TrackCurvatureBin = conf_.getParameter<unsigned int>("TrackCurvatureBin");
  double TrackCurvatureMin = conf_.getParameter<double>("TrackCurvatureMin");
  double TrackCurvatureMax = conf_.getParameter<double>("TrackCurvatureMax");

  histname = "TrackCurvature_";
  TrackCurvature_ =
      iBooker.book1D(histname + AlgoName, histname + AlgoName, TrackCurvatureBin, TrackCurvatureMin, TrackCurvatureMax);
  TrackCurvature_->setAxisTitle("#kappa track");

  if (runsOnReco_) {
    unsigned int JetPtBin = conf_.getParameter<unsigned int>("JetPtBin");
    double JetPtMin = conf_.getParameter<double>("JetPtMin");
    double JetPtMax = conf_.getParameter<double>("JetPtMax");

    histname = "JetPt_";
    jetPt_ = iBooker.book1D(histname + AlgoName, histname + AlgoName, JetPtBin, JetPtMin, JetPtMax);
    jetPt_->setAxisTitle("jet p_{T} / GeV");

    unsigned int MinJetDeltaRBin = conf_.getParameter<unsigned int>("MinJetDeltaRBin");
    double MinJetDeltaRMin = conf_.getParameter<double>("MinJetDeltaRMin");
    double MinJetDeltaRMax = conf_.getParameter<double>("MinJetDeltaRMax");

    histname = "MinJetDeltaR_";
    minJetDeltaR_ =
        iBooker.book1D(histname + AlgoName, histname + AlgoName, MinJetDeltaRBin, MinJetDeltaRMin, MinJetDeltaRMax);
    minJetDeltaR_->setAxisTitle("minimal Jet #DeltaR / rad");
  } else {
    jetPt_ = nullptr;
    minJetDeltaR_ = nullptr;
  }

  unsigned int MinTrackDeltaRBin = conf_.getParameter<unsigned int>("MinTrackDeltaRBin");
  double MinTrackDeltaRMin = conf_.getParameter<double>("MinTrackDeltaRMin");
  double MinTrackDeltaRMax = conf_.getParameter<double>("MinTrackDeltaRMax");

  histname = "MinTrackDeltaR_";
  minTrackDeltaR_ =
      iBooker.book1D(histname + AlgoName, histname + AlgoName, MinTrackDeltaRBin, MinTrackDeltaRMin, MinTrackDeltaRMax);
  minTrackDeltaR_->setAxisTitle("minimal Track #DeltaR / rad");

  unsigned int TrackEfficiencyBin = conf_.getParameter<unsigned int>("TrackEfficiencyBin");
  double TrackEfficiencyMin = conf_.getParameter<double>("TrackEfficiencyMin");
  double TrackEfficiencyMax = conf_.getParameter<double>("TrackEfficiencyMax");

  histname = "AlCaRecoTrackEfficiency_";
  AlCaRecoTrackEfficiency_ = iBooker.book1D(
      histname + AlgoName, histname + AlgoName, TrackEfficiencyBin, TrackEfficiencyMin, TrackEfficiencyMax);
  Labels l_tp, l_rtp;
  labelsForToken(referenceTrackProducer_, l_rtp);
  labelsForToken(trackProducer_, l_tp);
  AlCaRecoTrackEfficiency_->setAxisTitle("n(" + std::string(l_tp.module) + ") / n(" + std::string(l_rtp.module) + ")");

  int zBin = conf_.getParameter<unsigned int>("HitMapsZBin");  // 300
  double zMax = conf_.getParameter<double>("HitMapZMax");      // 300.0; //cm

  int rBin = conf_.getParameter<unsigned int>("HitMapsRBin");  // 120;
  double rMax = conf_.getParameter<double>("HitMapRMax");      // 120.0; //cm

  histname = "Hits_ZvsR_";
  double rMin = 0.0;
  if (useSignedR_)
    rMin = -rMax;

  Hits_ZvsR_ = iBooker.book2D(histname + AlgoName, histname + AlgoName, zBin, -zMax, zMax, rBin, rMin, rMax);

  histname = "Hits_XvsY_";
  Hits_XvsY_ = iBooker.book2D(histname + AlgoName, histname + AlgoName, rBin, -rMax, rMax, rBin, -rMax, rMax);

  if (fillRawIdMap_) {
    histname = "Hits_perDetId_";

    // leads to differences in axsis between samples??
    // int nModules = binByRawId_.size();
    // Hits_perDetId_ = iBooker.book1D(histname+AlgoName, histname+AlgoName,
    // nModules, static_cast<double>(nModules) -0.5,
    // static_cast<double>(nModules) -0.5);
    Hits_perDetId_ = iBooker.book1D(histname + AlgoName, histname + AlgoName, 16601, -0.5, 16600.5);
    Hits_perDetId_->setAxisTitle("rawId Bins");

    //// impossible takes too much memory :(
    //  std::stringstream binLabel;
    //  for( std::map<int,int>::iterator it = binByRawId_.begin(); it !=
    //  binByRawId_.end(); ++it ){
    //    binLabel.str() = "";
    //    binLabel << (*it).first;
    //    Hits_perDetId_->getTH1()->GetXaxis()->SetBinLabel( (*it).second +1,
    //    binLabel.str().c_str());
    //  }
  }
}
//
// -- Analyse
//
void TkAlCaRecoMonitor::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<reco::TrackCollection> trackCollection;
  iEvent.getByToken(trackProducer_, trackCollection);
  if (!trackCollection.isValid()) {
    edm::LogError("Alignment") << "invalid trackcollection encountered!";
    return;
  }

  edm::Handle<reco::TrackCollection> referenceTrackCollection;
  iEvent.getByToken(referenceTrackProducer_, referenceTrackCollection);
  if (!trackCollection.isValid()) {
    edm::LogError("Alignment") << "invalid reference track-collection encountered!";
    return;
  }

  const auto &geometry = iSetup.getHandle(tkGeomToken_);
  if (!geometry.isValid()) {
    edm::LogError("Alignment") << "invalid geometry found in event setup!";
  }

  const auto &magneticField = iSetup.getHandle(mfToken_);
  if (!magneticField.isValid()) {
    edm::LogError("Alignment") << "invalid magnetic field configuration encountered!";
    return;
  }

  edm::Handle<reco::CaloJetCollection> jets;
  if (runsOnReco_) {
    iEvent.getByToken(jetCollection_, jets);
    if (!jets.isValid()) {
      edm::LogError("Alignment") << "no jet collection found in event!";
    }
  }
  // fill only once - not yet in beginJob since no access to geometry
  if (fillRawIdMap_ && binByRawId_.empty())
    this->fillRawIdMap(*geometry);

  AlCaRecoTrackEfficiency_->Fill(static_cast<double>((*trackCollection).size()) / (*referenceTrackCollection).size());

  double sumOfCharges = 0;
  for (reco::TrackCollection::const_iterator track = (*trackCollection).begin(); track < (*trackCollection).end();
       ++track) {
    double dR = 0;
    if (runsOnReco_) {
      double minJetDeltaR = 10;  // some number > 2pi
      for (reco::CaloJetCollection::const_iterator itJet = jets->begin(); itJet != jets->end(); ++itJet) {
        jetPt_->Fill((*itJet).pt());
        dR = deltaR((*track), (*itJet));
        if ((*itJet).pt() > maxJetPt_ && dR < minJetDeltaR)
          minJetDeltaR = dR;

        // edm::LogInfo("Alignment") <<">  isolated: "<< isolated << " jetPt "<<
        // (*itJet).pt() <<" deltaR: "<< deltaR(*(*it),(*itJet)) ;
      }
      minJetDeltaR_->Fill(minJetDeltaR);
    }

    double minTrackDeltaR = 10;  // some number > 2pi
    for (reco::TrackCollection::const_iterator track2 = (*trackCollection).begin(); track2 < (*trackCollection).end();
         ++track2) {
      dR = deltaR((*track), (*track2));
      if (dR < minTrackDeltaR && dR > 1e-6)
        minTrackDeltaR = dR;
    }

    for (int i = 0; i < reco::TrackBase::qualitySize; ++i) {
      if ((*track).quality(reco::TrackBase::TrackQuality(i))) {
        TrackQuality_->Fill(i);
      }
    }

    GlobalPoint gPoint((*track).vx(), (*track).vy(), (*track).vz());
    double B = magneticField->inTesla(gPoint).z();
    double curv = -(*track).charge() * 0.002998 * B / (*track).pt();
    TrackCurvature_->Fill(curv);

    if ((*track).charge() > 0)
      TrackPtPositive_->Fill((*track).pt());
    if ((*track).charge() < 0)
      TrackPtNegative_->Fill((*track).pt());

    minTrackDeltaR_->Fill(minTrackDeltaR);
    fillHitmaps(*track, *geometry);
    sumOfCharges += (*track).charge();
  }

  sumCharge_->Fill(sumOfCharges);

  if (fillInvariantMass_) {
    if ((*trackCollection).size() == 2) {
      TLorentzVector track0(
          (*trackCollection).at(0).px(),
          (*trackCollection).at(0).py(),
          (*trackCollection).at(0).pz(),
          sqrt(((*trackCollection).at(0).p() * (*trackCollection).at(0).p()) + daughterMass_ * daughterMass_));
      TLorentzVector track1(
          (*trackCollection).at(1).px(),
          (*trackCollection).at(1).py(),
          (*trackCollection).at(1).pz(),
          sqrt(((*trackCollection).at(1).p() * (*trackCollection).at(1).p()) + daughterMass_ * daughterMass_));
      TLorentzVector mother = track0 + track1;

      invariantMass_->Fill(mother.M());
    } else {
      edm::LogInfo("Alignment") << "wrong number of tracks trackcollection encountered: " << (*trackCollection).size();
    }
  }
}

void TkAlCaRecoMonitor::fillHitmaps(const reco::Track &track, const TrackerGeometry &geometry) {
  for (trackingRecHit_iterator iHit = track.recHitsBegin(); iHit != track.recHitsEnd(); ++iHit) {
    if ((*iHit)->isValid()) {
      const TrackingRecHit *hit = (*iHit);
      const DetId geoId(hit->geographicalId());
      const GeomDet *gd = geometry.idToDet(geoId);
      // since 2_1_X local hit positions are transient. taking center of the hit
      // module for now. The alternative would be the coarse estimation or a
      // refit.
      // const GlobalPoint globP( gd->toGlobal( hit->localPosition() ) );
      const GlobalPoint globP(gd->toGlobal(Local3DPoint(0., 0., 0.)));
      double r = sqrt(globP.x() * globP.x() + globP.y() * globP.y());
      if (useSignedR_)
        r *= globP.y() / fabs(globP.y());

      Hits_ZvsR_->Fill(globP.z(), r);
      Hits_XvsY_->Fill(globP.x(), globP.y());

      if (fillRawIdMap_)
        Hits_perDetId_->Fill(binByRawId_[geoId.rawId()]);
    }
  }
}

void TkAlCaRecoMonitor::fillRawIdMap(const TrackerGeometry &geometry) {
  std::vector<int> sortedRawIds;
  for (std::vector<DetId>::const_iterator iDetId = geometry.detUnitIds().begin(); iDetId != geometry.detUnitIds().end();
       ++iDetId) {
    sortedRawIds.push_back((*iDetId).rawId());
  }
  std::sort(sortedRawIds.begin(), sortedRawIds.end());

  int i = 0;
  for (std::vector<int>::iterator iRawId = sortedRawIds.begin(); iRawId != sortedRawIds.end(); ++iRawId) {
    binByRawId_[*iRawId] = i;
    ++i;
  }
}

DEFINE_FWK_MODULE(TkAlCaRecoMonitor);
