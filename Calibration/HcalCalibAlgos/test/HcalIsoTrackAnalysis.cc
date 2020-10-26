// system include files
#include <memory>
#include <string>
#include <vector>

// Root objects
#include "TH1D.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

//Tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
//Generator information
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//#define EDM_ML_DEBUG

class HcalIsoTrackAnalysis : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HcalIsoTrackAnalysis(edm::ParameterSet const&);
  ~HcalIsoTrackAnalysis() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

  edm::Service<TFileService> fs_;
  spr::trackSelectionParameters selectionParameter_;
  const std::string theTrackQuality_;
  const std::vector<double> maxDxyPV_, maxDzPV_, maxChi2_, maxDpOverP_;
  const std::vector<int> minOuterHit_, minLayerCrossed_;
  const std::vector<int> maxInMiss_, maxOutMiss_;
  const double a_coneR_, a_mipR_;
  const double pTrackLow_, pTrackHigh_;
  const int useRaw_, dataType_, etaMin_, etaMax_;
  const double hitEthrEB_, hitEthrEE0_, hitEthrEE1_;
  const double hitEthrEE2_, hitEthrEE3_;
  const double hitEthrEELo_, hitEthrEEHi_;
  const std::string labelGenTrack_, labelRecVtx_, labelEB_;
  const std::string labelEE_, labelHBHE_;
  double a_charIsoR_;

  edm::EDGetTokenT<reco::TrackCollection> tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot> tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<GenEventInfoProduct> tok_ew_;

  std::vector<TH1D*> h_eta_, h_eta0_, h_eta1_, h_rat0_, h_rat1_;
  TH1D *h_Dxy_, *h_Dz_, *h_Chi2_, *h_DpOverP_;
  TH1D *h_Layer_, *h_OutHit_, *h_InMiss_, *h_OutMiss_;
};

HcalIsoTrackAnalysis::HcalIsoTrackAnalysis(const edm::ParameterSet& iConfig)
    : theTrackQuality_(iConfig.getParameter<std::string>("trackQuality")),
      maxDxyPV_(iConfig.getParameter<std::vector<double>>("maxDxyPV")),
      maxDzPV_(iConfig.getParameter<std::vector<double>>("maxDzPV")),
      maxChi2_(iConfig.getParameter<std::vector<double>>("maxChi2")),
      maxDpOverP_(iConfig.getParameter<std::vector<double>>("maxDpOverP")),
      minOuterHit_(iConfig.getParameter<std::vector<int>>("minOuterHit")),
      minLayerCrossed_(iConfig.getParameter<std::vector<int>>("minLayerCrossed")),
      maxInMiss_(iConfig.getParameter<std::vector<int>>("maxInMiss")),
      maxOutMiss_(iConfig.getParameter<std::vector<int>>("maxOutMiss")),
      a_coneR_(iConfig.getParameter<double>("coneRadius")),
      a_mipR_(iConfig.getParameter<double>("coneRadiusMIP")),
      pTrackLow_(iConfig.getParameter<double>("momentumLow")),
      pTrackHigh_(iConfig.getParameter<double>("momentumHigh")),
      useRaw_(iConfig.getUntrackedParameter<int>("useRaw", 0)),
      dataType_(iConfig.getUntrackedParameter<int>("dataType", 0)),
      etaMin_(iConfig.getUntrackedParameter<int>("etaMin", -1)),
      etaMax_(iConfig.getUntrackedParameter<int>("etaMax", 10)),
      hitEthrEB_(iConfig.getParameter<double>("EBHitEnergyThreshold")),
      hitEthrEE0_(iConfig.getParameter<double>("EEHitEnergyThreshold0")),
      hitEthrEE1_(iConfig.getParameter<double>("EEHitEnergyThreshold1")),
      hitEthrEE2_(iConfig.getParameter<double>("EEHitEnergyThreshold2")),
      hitEthrEE3_(iConfig.getParameter<double>("EEHitEnergyThreshold3")),
      hitEthrEELo_(iConfig.getParameter<double>("EEHitEnergyThresholdLow")),
      hitEthrEEHi_(iConfig.getParameter<double>("EEHitEnergyThresholdHigh")),
      labelGenTrack_(iConfig.getParameter<std::string>("labelTrack")),
      labelRecVtx_(iConfig.getParameter<std::string>("labelVertex")),
      labelEB_(iConfig.getParameter<std::string>("labelEBRecHit")),
      labelEE_(iConfig.getParameter<std::string>("labelEERecHit")),
      labelHBHE_(iConfig.getParameter<std::string>("labelHBHERecHit")) {
  usesResource(TFileService::kSharedResource);

  //now do whatever initialization is needed
  const double isolationRadius(28.9);
  reco::TrackBase::TrackQuality trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality_);
  selectionParameter_.minPt = iConfig.getParameter<double>("minTrackPt");
  selectionParameter_.minQuality = trackQuality_;
  a_charIsoR_ = a_coneR_ + isolationRadius;
  // Different isolation cuts are described in DN-2016/029
  // Tight cut uses 2 GeV; Loose cut uses 10 GeV
  // Eta dependent cut uses (maxRestrictionP_ * exp(|ieta|*log(2.5)/18))
  // with the factor for exponential slopeRestrictionP_ = log(2.5)/18
  // maxRestrictionP_ = 8 GeV as came from a study
  std::string labelBS = iConfig.getParameter<std::string>("labelBeamSpot");

  // define tokens for access
  tok_bs_ = consumes<reco::BeamSpot>(labelBS);
  tok_ew_ = consumes<GenEventInfoProduct>(edm::InputTag("generator"));
  tok_genTrack_ = consumes<reco::TrackCollection>(labelGenTrack_);
  tok_recVtx_ = consumes<reco::VertexCollection>(labelRecVtx_);
  tok_EB_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", labelEB_));
  tok_EE_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", labelEE_));
  tok_hbhe_ = consumes<HBHERecHitCollection>(labelHBHE_);
  edm::LogVerbatim("HcalIsoTrack") << "Labels used " << labelBS << " " << labelRecVtx_ << " " << labelGenTrack_ << " "
                                   << edm::InputTag("ecalRecHit", labelEB_) << " "
                                   << edm::InputTag("ecalRecHit", labelEE_) << " " << labelHBHE_;

  edm::LogVerbatim("HcalIsoTrack") << "Parameters read from config file \n"
                                   << "\t minPt " << selectionParameter_.minPt << "\t theTrackQuality "
                                   << theTrackQuality_ << "\t a_coneR " << a_coneR_ << "\t a_charIsoR " << a_charIsoR_
                                   << "\t a_mipR " << a_mipR_ << "\n\t momentumLow_ " << pTrackLow_
                                   << "\t momentumHigh_ " << pTrackHigh_ << "\t useRaw_ " << useRaw_
                                   << "\t dataType_      " << dataType_ << "\t etaLimit " << etaMin_ << ":" << etaMax_
                                   << "\nThreshold for EB " << hitEthrEB_ << " EE " << hitEthrEE0_ << ":" << hitEthrEE1_
                                   << ":" << hitEthrEE2_ << ":" << hitEthrEE3_ << ":" << hitEthrEELo_ << ":"
                                   << hitEthrEEHi_;
}

void HcalIsoTrackAnalysis::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "Run " << iEvent.id().run() << " Event " << iEvent.id().event() << " type "
                                   << dataType_ << " Luminosity " << iEvent.luminosityBlock() << " Bunch "
                                   << iEvent.bunchCrossing();
#endif
  //Get magnetic field
  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  const MagneticField* bField = bFieldH.product();

  // get handles to calogeometry
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();

  bool okC(true);
  //Get track collection
  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);
  if (!trkCollection.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelGenTrack_;
    okC = false;
  }

  //event weight for FLAT sample
  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(tok_ew_, genEventInfo);
  double wt = ((genEventInfo.isValid()) ? genEventInfo->weight() : 1.0);

  //Define the best vertex and the beamspot
  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(tok_recVtx_, recVtxs);
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_bs_, beamSpotH);
  math::XYZPoint leadPV(0, 0, 0);
  bool goodPV(false);
  if (recVtxs.isValid() && !(recVtxs->empty())) {
    for (unsigned int k = 0; k < recVtxs->size(); ++k) {
      if (!((*recVtxs)[k].isFake()) && ((*recVtxs)[k].ndof() > 4)) {
        leadPV = math::XYZPoint((*recVtxs)[k].x(), (*recVtxs)[k].y(), (*recVtxs)[k].z());
        goodPV = true;
        break;
      }
    }
  }
  if (!goodPV && beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "Primary Vertex (" << goodPV << ") " << leadPV;
  if (beamSpotH.isValid()) {
    edm::LogVerbatim("HcalIsoTrack") << " Beam Spot " << beamSpotH->position();
  }
#endif
  // RecHits
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelEB_;
    okC = false;
  }
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);
  if (!endcapRecHitsHandle.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelEE_;
    okC = false;
  }
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_, hbhe);
  if (!hbhe.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelHBHE_;
    okC = false;
  }

  if (okC) {
    //Propagate tracks to calorimeter surface)
    std::vector<spr::propagatedTrackDirection> trkCaloDirections;
    spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_, trkCaloDirections, false);
    std::vector<spr::propagatedTrackID> trkCaloDets;
    spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_, trkCaloDets, false);

    //Loop over all tracks
    std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
    unsigned int nTracks(0);
    for (trkDetItr = trkCaloDirections.begin(), nTracks = 0; trkDetItr != trkCaloDirections.end();
         trkDetItr++, nTracks++) {
      const reco::Track* pTrack = &(*(trkDetItr->trkItr));
      double p = pTrack->p();
      if (p >= pTrackLow_ && p <= pTrackHigh_ && (trkDetItr->okHCAL)) {
        int ieta = (static_cast<HcalDetId>(trkDetItr->detIdHCAL)).ieta();

        ////////////////////////////////-Energy in ECAL-//////////////////////////
        std::vector<DetId> eIds;
        std::vector<double> eHit;
        double eMipDR = spr::eCone_ecal(geo,
                                        barrelRecHitsHandle,
                                        endcapRecHitsHandle,
                                        trkDetItr->pointHCAL,
                                        trkDetItr->pointECAL,
                                        a_mipR_,
                                        trkDetItr->directionECAL,
                                        eIds,
                                        eHit);
        double eEcal(0);
        for (unsigned int k = 0; k < eIds.size(); ++k) {
          const GlobalPoint& pos = geo->getPosition(eIds[k]);
          double eta = std::abs(pos.eta());
          double eThr(hitEthrEB_);
          if (eIds[k].subdetId() != EcalBarrel) {
            eThr = (((eta * hitEthrEE3_ + hitEthrEE2_) * eta + hitEthrEE1_) * eta + hitEthrEE0_);
            if (eThr < hitEthrEELo_)
              eThr = hitEthrEELo_;
            else if (eThr > hitEthrEEHi_)
              eThr = hitEthrEEHi_;
          }
          if (eHit[k] > eThr)
            eEcal += eHit[k];
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HcalIsoTrack") << "eMIP before and after: " << eMipDR << ":" << eEcal;
#endif

        ////////////////////////////////-Energy in HCAL-//////////////////////////
        int nRecHits(-999), nNearTRKs(0);
        std::vector<DetId> ids;
        std::vector<double> edet0;
        double eHcal = spr::eCone_hcal(geo,
                                       hbhe,
                                       trkDetItr->pointHCAL,
                                       trkDetItr->pointECAL,
                                       a_coneR_,
                                       trkDetItr->directionHCAL,
                                       nRecHits,
                                       ids,
                                       edet0,
                                       useRaw_);
        double ratio0 = eHcal / (p - eEcal);
        double ratio1 = eHcal / (p - eMipDR);
        double hmaxNearP = spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR_, nNearTRKs, false);
        static const double tightCut(2.0), looseCut(2.0);
        bool tight = (hmaxNearP < tightCut);
        bool loose = (hmaxNearP < looseCut);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HcalIsoTrack") << "eHcal and responses: " << eHcal << ":" << ratio0 << ":" << ratio1
                                         << " Isolation " << hmaxNearP << ":" << loose << ":" << tight;
#endif
        //Different criteria for selection of good tracks
        if (std::abs(ieta) > etaMin_ && std::abs(ieta) < etaMax_) {
          unsigned id(0);
          h_eta_[id]->Fill(ieta, wt);
          h_rat0_[id]->Fill(ratio0, wt);
          h_rat1_[id]->Fill(ratio1, wt);
          if (loose)
            h_eta0_[id]->Fill(ieta, wt);
          if (tight)
            h_eta1_[id]->Fill(ieta, wt);
          for (unsigned int k1 = 0; k1 < maxDxyPV_.size(); ++k1) {
            for (unsigned int k2 = 0; k2 < maxDzPV_.size(); ++k2) {
              for (unsigned int k3 = 0; k3 < maxChi2_.size(); ++k3) {
                for (unsigned int k4 = 0; k4 < maxDpOverP_.size(); ++k4) {
                  for (unsigned int k5 = 0; k5 < minOuterHit_.size(); ++k5) {
                    for (unsigned int k6 = 0; k6 < minLayerCrossed_.size(); ++k6) {
                      for (unsigned int k7 = 0; k7 < maxInMiss_.size(); ++k7) {
                        for (unsigned int k8 = 0; k8 < maxOutMiss_.size(); ++k8) {
                          ++id;
                          selectionParameter_.maxDxyPV = maxDxyPV_[k1];
                          selectionParameter_.maxDzPV = maxDzPV_[k2];
                          selectionParameter_.maxChi2 = maxChi2_[k3];
                          selectionParameter_.maxDpOverP = maxDpOverP_[k4];
                          selectionParameter_.minOuterHit = minOuterHit_[k5];
                          selectionParameter_.minLayerCrossed = minLayerCrossed_[k6];
                          selectionParameter_.maxInMiss = maxInMiss_[k7];
                          selectionParameter_.maxOutMiss = maxOutMiss_[k8];
                          if (spr::goodTrack(pTrack, leadPV, selectionParameter_, false)) {
                            h_eta_[id]->Fill(ieta, wt);
                            h_rat0_[id]->Fill(ratio0, wt);
                            h_rat1_[id]->Fill(ratio1, wt);
                            if (loose)
                              h_eta0_[id]->Fill(ieta, wt);
                            if (tight)
                              h_eta1_[id]->Fill(ieta, wt);
                            const reco::HitPattern& hitp = pTrack->hitPattern();
                            if ((k2 + k3 + k4 + k5 + k6 + k7 + k8 == 0) && (k1 + 1 == maxDxyPV_.size()))
                              h_Dxy_->Fill(pTrack->dxy(leadPV), wt);
                            if ((k1 + k3 + k4 + k5 + k6 + k7 + k8 == 0) && (k2 + 1 == maxDzPV_.size()))
                              h_Dz_->Fill(pTrack->dz(leadPV), wt);
                            if ((k1 + k2 + k4 + k5 + k6 + k7 + k8 == 0) && (k3 + 1 == maxChi2_.size()))
                              h_Chi2_->Fill(pTrack->normalizedChi2(), wt);
                            if ((k1 + k2 + k3 + k5 + k6 + k7 + k8 == 0) && (k4 + 1 == maxDpOverP_.size()))
                              h_DpOverP_->Fill(std::abs(pTrack->qoverpError() / pTrack->qoverp()), wt);
                            if ((k1 + k2 + k3 + k4 + k6 + k7 + k8 == 0) && (k5 + 1 == minOuterHit_.size()))
                              h_OutHit_->Fill(
                                  (hitp.stripTOBLayersWithMeasurement() + hitp.stripTECLayersWithMeasurement()), wt);
                            if ((k1 + k2 + k3 + k4 + k5 + k7 + k8 == 0) && (k6 + 1 == minLayerCrossed_.size()))
                              h_Layer_->Fill(hitp.trackerLayersWithMeasurement(), wt);
                            if ((k1 + k2 + k3 + k4 + k5 + k6 + k8 == 0) && (k7 + 1 == maxInMiss_.size()))
                              h_InMiss_->Fill(
                                  hitp.trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS), wt);
                            if ((k1 + k2 + k3 + k4 + k5 + k6 + k7 == 0) && (k8 + 1 == maxOutMiss_.size()))
                              h_OutMiss_->Fill(
                                  hitp.trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_OUTER_HITS), wt);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void HcalIsoTrackAnalysis::beginJob() {
  char name[100], title[200];
  h_eta_.emplace_back(fs_->make<TH1D>("eta", "Track i#eta (All)", 60, -30, 30));
  h_eta0_.emplace_back(fs_->make<TH1D>("eta", "Track i#eta (All Loose Isolation)", 60, -30, 30));
  h_eta1_.emplace_back(fs_->make<TH1D>("eta", "Track i#eta (All Tight Isolation)", 60, -30, 30));
  h_rat0_.emplace_back(fs_->make<TH1D>("rat0", "Response 0", 100, 0.0, 5.0));
  h_rat1_.emplace_back(fs_->make<TH1D>("rat1", "Response 1", 100, 0.0, 5.0));
  for (unsigned int k1 = 0; k1 < maxDxyPV_.size(); ++k1) {
    for (unsigned int k2 = 0; k2 < maxDzPV_.size(); ++k2) {
      for (unsigned int k3 = 0; k3 < maxChi2_.size(); ++k3) {
        for (unsigned int k4 = 0; k4 < maxDpOverP_.size(); ++k4) {
          for (unsigned int k5 = 0; k5 < minOuterHit_.size(); ++k5) {
            for (unsigned int k6 = 0; k6 < minLayerCrossed_.size(); ++k6) {
              for (unsigned int k7 = 0; k7 < maxInMiss_.size(); ++k7) {
                for (unsigned int k8 = 0; k8 < maxOutMiss_.size(); ++k8) {
                  sprintf(name, "eta%d%d%d%d%d%d%d%d", k1, k2, k3, k4, k5, k6, k7, k8);
                  sprintf(title,
                          "i#eta (d_{xy}=4.2%f, d_{z}=4.2%f, #chi^{2}=5.2%f, (#Delta p)/p=5.2%f, Hit_{out}=%d, "
                          "Layer=%d, Miss_{in}=%d, Miss_{out}=%d)",
                          maxDxyPV_[k1],
                          maxDzPV_[k2],
                          maxChi2_[k3],
                          maxDpOverP_[k4],
                          minOuterHit_[k5],
                          minLayerCrossed_[k6],
                          maxInMiss_[k7],
                          maxOutMiss_[k8]);
                  h_eta_.emplace_back(fs_->make<TH1D>(name, title, 60, -30, 30));
                  sprintf(name, "eta0%d%d%d%d%d%d%d%d", k1, k2, k3, k4, k5, k6, k7, k8);
                  sprintf(title,
                          "i#eta (d_{xy}=4.2%f, d_{z}=4.2%f, #chi^{2}=5.2%f, (#Delta p)/p=5.2%f, Hit_{out}=%d, "
                          "Layer=%d, Miss_{in}=%d, Miss_{out}=%d, loose isolation)",
                          maxDxyPV_[k1],
                          maxDzPV_[k2],
                          maxChi2_[k3],
                          maxDpOverP_[k4],
                          minOuterHit_[k5],
                          minLayerCrossed_[k6],
                          maxInMiss_[k7],
                          maxOutMiss_[k8]);
                  h_eta0_.emplace_back(fs_->make<TH1D>(name, title, 60, -30, 30));
                  sprintf(name, "eta1%d%d%d%d%d%d%d%d", k1, k2, k3, k4, k5, k6, k7, k8);
                  sprintf(title,
                          "i#eta (d_{xy}=4.2%f, d_{z}=4.2%f, #chi^{2}=5.2%f, (#Delta p)/p=5.2%f, Hit_{out}=%d, "
                          "Layer=%d, Miss_{in}=%d, Miss_{out}=%d, tight isolation)",
                          maxDxyPV_[k1],
                          maxDzPV_[k2],
                          maxChi2_[k3],
                          maxDpOverP_[k4],
                          minOuterHit_[k5],
                          minLayerCrossed_[k6],
                          maxInMiss_[k7],
                          maxOutMiss_[k8]);
                  h_eta1_.emplace_back(fs_->make<TH1D>(name, title, 60, -30, 30));
                  sprintf(name, "rat0%d%d%d%d%d%d%d%d", k1, k2, k3, k4, k5, k6, k7, k8);
                  sprintf(title,
                          "Response 0 (d_{xy}=4.2%f, d_{z}=4.2%f, #chi^{2}=5.2%f, (#Delta p)/p=5.2%f, Hit_{out}=%d, "
                          "Layer=%d, Miss_{in}=%d, Miss_{out}=%d)",
                          maxDxyPV_[k1],
                          maxDzPV_[k2],
                          maxChi2_[k3],
                          maxDpOverP_[k4],
                          minOuterHit_[k5],
                          minLayerCrossed_[k6],
                          maxInMiss_[k7],
                          maxOutMiss_[k8]);
                  h_rat0_.emplace_back(fs_->make<TH1D>(name, title, 100, 0.0, 5.0));
                  sprintf(name, "rat1%d%d%d%d%d%d%d%d", k1, k2, k3, k4, k5, k6, k7, k8);
                  sprintf(title,
                          "Response 1 (d_{xy}=4.2%f, d_{z}=4.2%f, #chi^{2}=5.2%f, (#Delta p)/p=5.2%f, Hit_{out}=%d, "
                          "Layer=%d, Miss_{in}=%d, Miss_{out}=%d)",
                          maxDxyPV_[k1],
                          maxDzPV_[k2],
                          maxChi2_[k3],
                          maxDpOverP_[k4],
                          minOuterHit_[k5],
                          minLayerCrossed_[k6],
                          maxInMiss_[k7],
                          maxOutMiss_[k8]);
                  h_rat1_.emplace_back(fs_->make<TH1D>(name, title, 100, 0.0, 5.0));
                }
              }
            }
          }
        }
      }
    }
  }
  h_Dxy_ = fs_->make<TH1D>("Dxy", "d_{xy}", 100, 0.0, 1.0);
  h_Dz_ = fs_->make<TH1D>("Dz", "d_{z}", 100, 0.0, 1.0);
  h_Chi2_ = fs_->make<TH1D>("Chi2", "#chi^{2}", 100, 0.0, 20.0);
  h_DpOverP_ = fs_->make<TH1D>("DpOverP", "#frac{#Delta p}{p}", 100, 0.0, 1.0);
  h_Layer_ = fs_->make<TH1D>("Layer", "Layers Crossed", 50, 0.0, 50.0);
  h_OutHit_ = fs_->make<TH1D>("OutHit", "Outer Layers Hit", 20, 0.0, 20.0);
  h_InMiss_ = fs_->make<TH1D>("InMiss", "Missed Inner Hits", 20, 0.0, 20.0);
  h_OutMiss_ = fs_->make<TH1D>("OutMiss", "Missed Outer Hits", 20, 0.0, 20.0);
}

void HcalIsoTrackAnalysis::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // following 10 parameters are parameters to select good tracks
  desc.add<std::string>("trackQuality", "highPurity");
  desc.add<double>("minTrackPt", 1.0);
  std::vector<double> maxdxy = {0.02, 0.01, 0.05, 0.10};
  std::vector<double> maxdz = {0.02, 0.01, 0.04, 0.50};
  std::vector<double> maxchi2 = {5.0, 2.0, 10.0, 20.0};
  std::vector<double> maxdpoverp = {0.1, 0.02, 0.05, 0.4};
  std::vector<int> minouterhit = {4, 2, 1, 0};
  std::vector<int> minlayercrossed = {8, 4, 2, 0};
  std::vector<int> maxinmiss = {0, 1, 2, 4};
  std::vector<int> maxoutmiss = {0, 1, 2, 4};
  desc.add<std::vector<double>>("maxDxyPV", maxdxy);
  desc.add<std::vector<double>>("maxDzPV", maxdz);
  desc.add<std::vector<double>>("maxChi2", maxchi2);
  desc.add<std::vector<double>>("maxDpOverP", maxdpoverp);
  desc.add<std::vector<int>>("minOuterHit", minouterhit);
  desc.add<std::vector<int>>("minLayerCrossed", minlayercrossed);
  desc.add<std::vector<int>>("maxInMiss", maxinmiss);
  desc.add<std::vector<int>>("maxOutMiss", maxoutmiss);
  // Signal zone in HCAL and ECAL
  desc.add<double>("coneRadius", 34.98);
  desc.add<double>("coneRadiusMIP", 14.0);
  // energy thershold for ECAL (from Egamma group)
  desc.add<double>("EBHitEnergyThreshold", 0.08);
  desc.add<double>("EEHitEnergyThreshold0", 0.30);
  desc.add<double>("EEHitEnergyThreshold1", 0.00);
  desc.add<double>("EEHitEnergyThreshold2", 0.00);
  desc.add<double>("EEHitEnergyThreshold3", 0.00);
  desc.add<double>("EEHitEnergyThresholdLow", 0.30);
  desc.add<double>("EEHitEnergyThresholdHigh", 0.30);
  // prescale factors
  desc.add<double>("momentumLow", 40.0);
  desc.add<double>("momentumHigh", 60.0);
  // various labels for collections used in the code
  desc.add<std::string>("labelTrack", "generalTracks");
  desc.add<std::string>("labelVertex", "offlinePrimaryVertices");
  desc.add<std::string>("labelEBRecHit", "EcalRecHitsEB");
  desc.add<std::string>("labelEERecHit", "EcalRecHitsEE");
  desc.add<std::string>("labelHBHERecHit", "hbhereco");
  desc.add<std::string>("labelBeamSpot", "offlineBeamSpot");
  //  Various flags used for selecting tracks, choice of energy Method2/0
  //  Data type 0/1 for single jet trigger or others
  desc.addUntracked<int>("useRaw", 0);
  desc.addUntracked<int>("dataType", 0);
  desc.addUntracked<int>("etaMin", -1);
  desc.addUntracked<int>("etaMax", 10);
  descriptions.add("HcalIsoTrackAnalysis", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalIsoTrackAnalysis);
