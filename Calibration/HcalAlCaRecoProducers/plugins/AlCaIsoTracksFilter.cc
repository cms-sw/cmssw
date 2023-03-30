// system include files
#include <algorithm>
#include <atomic>
#include <memory>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"

#include "DataFormats/Common/interface/Handle.h"
//Tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
// RecHits
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
//Triggers
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

//#define EDM_ML_DEBUG
//
// class declaration
//

namespace alCaIsoTracksFilter {
  struct Counters {
    Counters() : nAll_(0), nGood_(0), nRange_(0), nHigh_(0) {}
    mutable std::atomic<unsigned int> nAll_, nGood_, nRange_, nHigh_;
  };
}  // namespace alCaIsoTracksFilter

class AlCaIsoTracksFilter : public edm::stream::EDFilter<edm::GlobalCache<alCaIsoTracksFilter::Counters>> {
public:
  explicit AlCaIsoTracksFilter(edm::ParameterSet const&, const alCaIsoTracksFilter::Counters* count);
  ~AlCaIsoTracksFilter() override = default;

  static std::unique_ptr<alCaIsoTracksFilter::Counters> initializeGlobalCache(edm::ParameterSet const& iConfig) {
    return std::make_unique<alCaIsoTracksFilter::Counters>();
  }

  bool filter(edm::Event&, edm::EventSetup const&) override;
  void endStream() override;
  static void globalEndJob(const alCaIsoTracksFilter::Counters* counters);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  HLTConfigProvider hltConfig_;
  const std::vector<std::string> trigNames_;
  const edm::InputTag labelGenTrack_, labelRecVtx_;
  const edm::InputTag labelEB_, labelEE_, labelHBHE_;
  const edm::InputTag triggerEvent_, theTriggerResultsLabel_;
  const std::string processName_;
  const double a_coneR_, a_mipR_, pTrackMin_, eEcalMax_;
  const double maxRestrictionP_, slopeRestrictionP_;
  const double eIsolate_;
  const double hitEthrEB_, hitEthrEE0_, hitEthrEE1_;
  const double hitEthrEE2_, hitEthrEE3_;
  const double hitEthrEELo_, hitEthrEEHi_;
  const double pTrackLow_, pTrackHigh_, pTrackH_;
  const int preScale_, preScaleH_;
  const std::string theTrackQuality_;
  const std::vector<int> debEvents_;
  const bool usePFThresh_;
  spr::trackSelectionParameters selectionParameter_;
  double a_charIsoR_;
  unsigned int nRun_, nAll_, nGood_, nRange_, nHigh_;
  edm::EDGetTokenT<trigger::TriggerEvent> tok_trigEvt_;
  edm::EDGetTokenT<edm::TriggerResults> tok_trigRes_;
  edm::EDGetTokenT<reco::TrackCollection> tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot> tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;
  edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> tok_ecalPFRecHitThresholds_;

  const EcalPFRecHitThresholds* eThresholds_;
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
AlCaIsoTracksFilter::AlCaIsoTracksFilter(const edm::ParameterSet& iConfig, const alCaIsoTracksFilter::Counters* count)
    : trigNames_(iConfig.getParameter<std::vector<std::string>>("triggers")),
      labelGenTrack_(iConfig.getParameter<edm::InputTag>("labelTrack")),
      labelRecVtx_(iConfig.getParameter<edm::InputTag>("labelVertex")),
      labelEB_(iConfig.getParameter<edm::InputTag>("labelEBRecHit")),
      labelEE_(iConfig.getParameter<edm::InputTag>("labelEERecHit")),
      labelHBHE_(iConfig.getParameter<edm::InputTag>("labelHBHERecHit")),
      triggerEvent_(iConfig.getParameter<edm::InputTag>("labelTriggerEvent")),
      theTriggerResultsLabel_(iConfig.getParameter<edm::InputTag>("labelTriggerResult")),
      processName_(iConfig.getParameter<std::string>("processName")),
      a_coneR_(iConfig.getParameter<double>("coneRadius")),
      a_mipR_(iConfig.getParameter<double>("coneRadiusMIP")),
      pTrackMin_(iConfig.getParameter<double>("minimumTrackP")),
      eEcalMax_(iConfig.getParameter<double>("maximumEcalEnergy")),
      maxRestrictionP_(iConfig.getParameter<double>("maxTrackP")),
      slopeRestrictionP_(iConfig.getParameter<double>("slopeTrackP")),
      eIsolate_(iConfig.getParameter<double>("isolationEnergy")),
      hitEthrEB_(iConfig.getParameter<double>("EBHitEnergyThreshold")),
      hitEthrEE0_(iConfig.getParameter<double>("EEHitEnergyThreshold0")),
      hitEthrEE1_(iConfig.getParameter<double>("EEHitEnergyThreshold1")),
      hitEthrEE2_(iConfig.getParameter<double>("EEHitEnergyThreshold2")),
      hitEthrEE3_(iConfig.getParameter<double>("EEHitEnergyThreshold3")),
      hitEthrEELo_(iConfig.getParameter<double>("EEHitEnergyThresholdLow")),
      hitEthrEEHi_(iConfig.getParameter<double>("EEHitEnergyThresholdHigh")),
      pTrackLow_(iConfig.getParameter<double>("momentumRangeLow")),
      pTrackHigh_(iConfig.getParameter<double>("momentumRangeHigh")),
      pTrackH_(iConfig.getParameter<double>("momentumHigh")),
      preScale_(iConfig.getParameter<int>("preScaleFactor")),
      preScaleH_(iConfig.getParameter<int>("preScaleHigh")),
      theTrackQuality_(iConfig.getParameter<std::string>("trackQuality")),
      debEvents_(iConfig.getParameter<std::vector<int>>("debugEvents")),
      usePFThresh_(iConfig.getParameter<bool>("usePFThreshold")),
      nRun_(0),
      nAll_(0),
      nGood_(0),
      nRange_(0),
      nHigh_(0) {
  //now do what ever initialization is needed
  const double isolationRadius(28.9);
  // Different isolation cuts are described in DN-2016/029
  // Tight cut uses 2 GeV; Loose cut uses 10 GeV
  // Eta dependent cut uses (maxRestrictionP_ * exp(|ieta|*log(2.5)/18))
  // with the factor for exponential slopeRestrictionP_ = log(2.5)/18
  // maxRestrictionP_ = 8 GeV as came from a study
  reco::TrackBase::TrackQuality trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality_);
  selectionParameter_.minPt = iConfig.getParameter<double>("minTrackPt");
  ;
  selectionParameter_.minQuality = trackQuality_;
  selectionParameter_.maxDxyPV = iConfig.getParameter<double>("maxDxyPV");
  selectionParameter_.maxDzPV = iConfig.getParameter<double>("maxDzPV");
  selectionParameter_.maxChi2 = iConfig.getParameter<double>("maxChi2");
  selectionParameter_.maxDpOverP = iConfig.getParameter<double>("maxDpOverP");
  selectionParameter_.minOuterHit = iConfig.getParameter<int>("minOuterHit");
  selectionParameter_.minLayerCrossed = iConfig.getParameter<int>("minLayerCrossed");
  selectionParameter_.maxInMiss = iConfig.getParameter<int>("maxInMiss");
  selectionParameter_.maxOutMiss = iConfig.getParameter<int>("maxOutMiss");
  a_charIsoR_ = a_coneR_ + isolationRadius;

  // define tokens for access
  tok_trigEvt_ = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes_ = consumes<edm::TriggerResults>(theTriggerResultsLabel_);
  tok_genTrack_ = consumes<reco::TrackCollection>(labelGenTrack_);
  tok_recVtx_ = consumes<reco::VertexCollection>(labelRecVtx_);
  tok_bs_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("labelBeamSpot"));

  tok_EB_ = consumes<EcalRecHitCollection>(labelEB_);
  tok_EE_ = consumes<EcalRecHitCollection>(labelEE_);
  tok_hbhe_ = consumes<HBHERecHitCollection>(labelHBHE_);

  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  tok_magField_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
  tok_ecalPFRecHitThresholds_ = esConsumes<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd>();

  edm::LogVerbatim("HcalIsoTrack") << "Parameters read from config file \n"
                                   << "\t minPt " << selectionParameter_.minPt << "\t theTrackQuality "
                                   << theTrackQuality_ << "\t minQuality " << selectionParameter_.minQuality
                                   << "\t maxDxyPV " << selectionParameter_.maxDxyPV << "\t maxDzPV "
                                   << selectionParameter_.maxDzPV << "\t maxChi2 " << selectionParameter_.maxChi2
                                   << "\t maxDpOverP " << selectionParameter_.maxDpOverP << "\t minOuterHit "
                                   << selectionParameter_.minOuterHit << "\t minLayerCrossed "
                                   << selectionParameter_.minLayerCrossed << "\t maxInMiss "
                                   << selectionParameter_.maxInMiss << "\t maxOutMiss "
                                   << selectionParameter_.maxOutMiss << "\n"
                                   << "\t a_coneR " << a_coneR_ << "\t a_charIsoR " << a_charIsoR_ << "\t a_mipR "
                                   << a_mipR_ << "\t maxRestrictionP_ " << maxRestrictionP_ << "\t slopeRestrictionP_ "
                                   << slopeRestrictionP_ << "\t eIsolate_ " << eIsolate_ << "\n"
                                   << "\t Precale factor " << preScale_ << "\t in momentum range " << pTrackLow_ << ":"
                                   << pTrackHigh_ << " and prescale factor " << preScaleH_ << " for p > " << pTrackH_
                                   << " Threshold flag used " << usePFThresh_ << " value for EB " << hitEthrEB_
                                   << " EE " << hitEthrEE0_ << ":" << hitEthrEE1_ << ":" << hitEthrEE2_ << ":"
                                   << hitEthrEE3_ << ":" << hitEthrEELo_ << ":" << hitEthrEEHi_ << " and "
                                   << debEvents_.size() << " events to be debugged";

  for (unsigned int k = 0; k < trigNames_.size(); ++k)
    edm::LogVerbatim("HcalIsoTrack") << "Trigger[" << k << "] " << trigNames_[k];
}  // AlCaIsoTracksFilter::AlCaIsoTracksFilter  constructor

//
// member functions
//

// ------------ method called on each new Event  ------------
bool AlCaIsoTracksFilter::filter(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  bool accept(false);
  ++nAll_;
#ifdef EDM_ML_DEBUG
  bool debug = (debEvents_.empty())
                   ? true
                   : (std::find(debEvents_.begin(), debEvents_.end(), iEvent.id().event()) != debEvents_.end());
  if (debug)
    edm::LogVerbatim("HcalIsoTrack") << "Run " << iEvent.id().run() << " Event " << iEvent.id().event()
                                     << " Luminosity " << iEvent.luminosityBlock() << " Bunch "
                                     << iEvent.bunchCrossing();
#endif

  // get Ecal Thresholds
  eThresholds_ = &iSetup.getData(tok_ecalPFRecHitThresholds_);

  //Step1: Find if the event passes one of the chosen triggers
  bool triggerSatisfied(false);
  if (trigNames_.empty()) {
    triggerSatisfied = true;
  } else {
    trigger::TriggerEvent triggerEvent;
    auto const& triggerEventHandle = iEvent.getHandle(tok_trigEvt_);
    if (!triggerEventHandle.isValid()) {
      edm::LogWarning("HcalIsoTrack") << "Error! Can't get the product " << triggerEvent_.label();
    } else {
      triggerEvent = *(triggerEventHandle.product());

      /////////////////////////////TriggerResults
      auto const& triggerResults = iEvent.getHandle(tok_trigRes_);
      if (triggerResults.isValid()) {
        std::vector<std::string> modules;
        const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
        const std::vector<std::string>& triggerNames_ = triggerNames.triggerNames();
        for (unsigned int iHLT = 0; iHLT < triggerResults->size(); iHLT++) {
          int hlt = triggerResults->accept(iHLT);
          for (unsigned int i = 0; i < trigNames_.size(); ++i) {
            if (triggerNames_[iHLT].find(trigNames_[i]) != std::string::npos) {
              if (hlt > 0)
                triggerSatisfied = true;
#ifdef EDM_ML_DEBUG
              if (debug)
                edm::LogVerbatim("HcalIsoTrack")
                    << triggerNames_[iHLT] << " has got HLT flag " << hlt << ":" << triggerSatisfied;
#endif
              if (triggerSatisfied)
                break;
            }
          }
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  if (debug)
    edm::LogVerbatim("HcalIsoTrack") << "AlCaIsoTracksFilter:: triggerSatisfied: " << triggerSatisfied;
#endif

  //Step2: Get geometry/B-field information
  if (triggerSatisfied) {
    //Get magnetic field
    const MagneticField* bField = &(iSetup.getData(tok_magField_));
    const CaloGeometry* geo = &(iSetup.getData(tok_geom_));

    //Also relevant information to extrapolate tracks to Hcal surface
    bool foundCollections(true);
    //Get track collection
    auto trkCollection = iEvent.getHandle(tok_genTrack_);
    if (!trkCollection.isValid()) {
      edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelGenTrack_;
      foundCollections = false;
    }

    //Define the best vertex and the beamspot
    auto const& recVtxs = iEvent.getHandle(tok_recVtx_);
    auto const& beamSpotH = iEvent.getHandle(tok_bs_);
    math::XYZPoint leadPV(0, 0, 0);
    if (!recVtxs->empty() && !((*recVtxs)[0].isFake())) {
      leadPV = math::XYZPoint((*recVtxs)[0].x(), (*recVtxs)[0].y(), (*recVtxs)[0].z());
    } else if (beamSpotH.isValid()) {
      leadPV = beamSpotH->position();
    }
#ifdef EDM_ML_DEBUG
    if (debug)
      edm::LogVerbatim("HcalIsoTrack") << "Primary Vertex " << leadPV;
#endif

    // RecHits
    auto barrelRecHitsHandle = iEvent.getHandle(tok_EB_);
    if (!barrelRecHitsHandle.isValid()) {
      edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelEB_;
      foundCollections = false;
    }
    auto endcapRecHitsHandle = iEvent.getHandle(tok_EE_);
    if (!endcapRecHitsHandle.isValid()) {
      edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelEE_;
      foundCollections = false;
    }
    auto hbhe = iEvent.getHandle(tok_hbhe_);
    if (!hbhe.isValid()) {
      edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelHBHE_;
      foundCollections = false;
    }
#ifdef EDM_ML_DEBUG
    if (debug)
      edm::LogVerbatim("HcalIsoTrack") << "AlCaIsoTracksFilter:: foundCollections: " << foundCollections;
#endif

    //Step3 propagate the tracks to calorimeter surface and find
    // candidates for isolated tracks
    if (foundCollections) {
      //Propagate tracks to calorimeter surface)
      std::vector<spr::propagatedTrackDirection> trkCaloDirections;
      spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_, trkCaloDirections, false);

      std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
#ifdef EDM_ML_DEBUG
      if (debug)
        edm::LogVerbatim("HcalIsoTrack") << "AlCaIsoTracksFilter:: Has " << trkCaloDirections.size()
                                         << " propagated tracks from a total of " << trkCollection->size();
#endif
      unsigned int nTracks(0), nselTracks(0), ntrin(0), ntrout(0), ntrH(0);
      for (trkDetItr = trkCaloDirections.begin(), nTracks = 0; trkDetItr != trkCaloDirections.end();
           trkDetItr++, nTracks++) {
        const reco::Track* pTrack = &(*(trkDetItr->trkItr));
        math::XYZTLorentzVector v4(pTrack->px(), pTrack->py(), pTrack->pz(), pTrack->p());
#ifdef EDM_ML_DEBUG
        if (debug)
          edm::LogVerbatim("HcalIsoTrack")
              << "This track : " << nTracks << " (pt|eta|phi|p) : " << pTrack->pt() << "|" << pTrack->eta() << "|"
              << pTrack->phi() << "|" << pTrack->p() << " OK HCAL " << trkDetItr->okHCAL;
#endif
        //Selection of good track
        int ieta(0);
        if (trkDetItr->okHCAL) {
          HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
          ieta = detId.ietaAbs();
        }
        bool qltyFlag = spr::goodTrack(pTrack, leadPV, selectionParameter_, false);
#ifdef EDM_ML_DEBUG
        if (debug)
          edm::LogVerbatim("HcalIsoTrack")
              << "qltyFlag|okECAL|okHCAL : " << qltyFlag << "|" << trkDetItr->okECAL << "|" << trkDetItr->okHCAL;
#endif
        if (qltyFlag && trkDetItr->okECAL && trkDetItr->okHCAL) {
          double t_p = pTrack->p();
          nselTracks++;
          int nNearTRKs(0);
          std::vector<DetId> eIds;
          std::vector<double> eHit;
#ifdef EDM_ML_DEBUG
          double eEcal =
#endif
              spr::eCone_ecal(geo,
                              barrelRecHitsHandle,
                              endcapRecHitsHandle,
                              trkDetItr->pointHCAL,
                              trkDetItr->pointECAL,
                              a_mipR_,
                              trkDetItr->directionECAL,
                              eIds,
                              eHit);
          double eMipDR(0);
          for (unsigned int k = 0; k < eIds.size(); ++k) {
            double eThr(hitEthrEB_);
            if (usePFThresh_) {
              eThr = static_cast<double>((*eThresholds_)[eIds[k]]);
            } else {
              const GlobalPoint& pos = geo->getPosition(eIds[k]);
              double eta = std::abs(pos.eta());
              if (eIds[k].subdetId() != EcalBarrel) {
                eThr = (((eta * hitEthrEE3_ + hitEthrEE2_) * eta + hitEthrEE1_) * eta + hitEthrEE0_);
                if (eThr < hitEthrEELo_)
                  eThr = hitEthrEELo_;
                else if (eThr > hitEthrEEHi_)
                  eThr = hitEthrEEHi_;
              }
            }
            if (eHit[k] > eThr)
              eMipDR += eHit[k];
          }
          double hmaxNearP = spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR_, nNearTRKs, false);
          double eIsolation = (maxRestrictionP_ * exp(slopeRestrictionP_ * ((double)(ieta))));
          if (eIsolation < eIsolate_)
            eIsolation = eIsolate_;
#ifdef EDM_ML_DEBUG
          std::string ctype =
              (t_p > pTrackMin_ && eMipDR < eEcalMax_ && hmaxNearP < eIsolation) ? " ***** ACCEPT *****" : "";
          if (debug)
            edm::LogVerbatim("HcalIsoTrack")
                << "This track : " << nTracks << " (pt|eta|phi|p) : " << pTrack->pt() << "|" << pTrack->eta() << "|"
                << pTrack->phi() << "|" << t_p << " e_MIP " << eMipDR << ":" << eEcal << " Chg Isolation " << hmaxNearP
                << ":" << eIsolation << ctype;
#endif
          if (t_p > pTrackMin_ && eMipDR < eEcalMax_ && hmaxNearP < eIsolation) {
            if (t_p > pTrackLow_ && t_p < pTrackHigh_)
              ntrin++;
            else if (t_p > pTrackH_)
              ntrH++;
            else
              ntrout++;
          }
        }
      }
      accept = (ntrout > 0);
      if (!accept && ntrin > 0) {
        ++nRange_;
        if (preScale_ <= 1)
          accept = true;
        else if (nRange_ % preScale_ == 1)
          accept = true;
      }
      if (!accept && ntrH > 0) {
        ++nHigh_;
        if (preScaleH_ <= 1)
          accept = true;
        else if (nHigh_ % preScaleH_ == 1)
          accept = true;
      }
#ifdef EDM_ML_DEBUG
      if (debug)
        edm::LogVerbatim("HcalIsoTrack") << "Summary Range " << ntrout << " Low " << ntrin << " High " << ntrH
                                         << " Accept " << accept;
#endif
    }
  }
  // Step 4:  Return the acceptance flag
  if (accept) {
    ++nGood_;
    edm::LogVerbatim("HcalIsoTrackX") << "Run " << iEvent.id().run() << " Event " << iEvent.id().event();
  }
  return accept;

}  // AlCaIsoTracksFilter::filter

// ------------ method called once each job just after ending the event loop  ------------
void AlCaIsoTracksFilter::endStream() {
  globalCache()->nAll_ += nAll_;
  globalCache()->nGood_ += nGood_;
  globalCache()->nRange_ += nRange_;
  globalCache()->nHigh_ += nHigh_;
}

void AlCaIsoTracksFilter::globalEndJob(const alCaIsoTracksFilter::Counters* count) {
  edm::LogVerbatim("HcalIsoTrack") << "Selects " << count->nGood_ << " in " << count->nAll_ << " events and with "
                                   << count->nRange_ << " events in the p-range" << count->nHigh_
                                   << " events with high p";
}

// ------------ method called when starting to processes a run  ------------
void AlCaIsoTracksFilter::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed(false);
  edm::LogVerbatim("HcalIsoTrack") << "Run[" << nRun_ << "] " << iRun.run() << " hltconfig.init "
                                   << hltConfig_.init(iRun, iSetup, processName_, changed);
}

// ------------ method called when ending the processing of a run  ------------
void AlCaIsoTracksFilter::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  ++nRun_;
  edm::LogVerbatim("HcalIsoTrack") << "endRun[" << nRun_ << "] " << iRun.run();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void AlCaIsoTracksFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("labelTrack", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("labelVertex", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("labelBeamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("labelEBRecHit", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("labelEERecHit", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  desc.add<edm::InputTag>("labelHBHERecHit", edm::InputTag("hbhereco"));
  desc.add<edm::InputTag>("labelTriggerEvent", edm::InputTag("hltTriggerSummaryAOD", "", "HLT"));
  desc.add<edm::InputTag>("labelTriggerResult", edm::InputTag("TriggerResults", "", "HLT"));
  std::vector<std::string> trigger;
  desc.add<std::vector<std::string>>("triggers", trigger);
  desc.add<std::string>("processName", "HLT");
  // following 10 parameters are parameters to select good tracks
  desc.add<std::string>("trackQuality", "highPurity");
  desc.add<double>("minTrackPt", 1.0);
  desc.add<double>("maxDxyPV", 10.0);
  desc.add<double>("maxDzPV", 100.0);
  desc.add<double>("maxChi2", 5.0);
  desc.add<double>("maxDpOverP", 0.1);
  desc.add<int>("minOuterHit", 4);
  desc.add<int>("minLayerCrossed", 8);
  desc.add<int>("maxInMiss", 2);
  desc.add<int>("maxOutMiss", 2);
  // Minimum momentum of selected isolated track and signal zone
  desc.add<double>("coneRadius", 34.98);
  desc.add<double>("minimumTrackP", 20.0);
  // signal zone in ECAL and MIP energy cutoff
  desc.add<double>("coneRadiusMIP", 14.0);
  desc.add<double>("maximumEcalEnergy", 100.0);
  // following 3 parameters are for isolation cuts and described in the code
  desc.add<double>("maxTrackP", 8.0);
  desc.add<double>("slopeTrackP", 0.05090504066);
  desc.add<double>("isolationEnergy", 10.0);
  // energy thershold for ECAL (from Egamma group)
  desc.add<double>("EBHitEnergyThreshold", 0.08);
  desc.add<double>("EEHitEnergyThreshold0", 0.30);
  desc.add<double>("EEHitEnergyThreshold1", 0.00);
  desc.add<double>("EEHitEnergyThreshold2", 0.00);
  desc.add<double>("EEHitEnergyThreshold3", 0.00);
  desc.add<double>("EEHitEnergyThresholdLow", 0.30);
  desc.add<double>("EEHitEnergyThresholdHigh", 0.30);
  // Prescale events only containing isolated tracks in the range
  desc.add<double>("momentumRangeLow", 20.0);
  desc.add<double>("momentumRangeHigh", 40.0);
  desc.add<int>("preScaleFactor", 10);
  desc.add<double>("momentumHigh", 60.0);
  desc.add<int>("preScaleHigh", 5);
  std::vector<int> events;
  desc.add<std::vector<int>>("debugEvents", events);
  desc.add<bool>("usePFThreshold", true);
  descriptions.add("alcaIsoTracksFilter", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlCaIsoTracksFilter);
