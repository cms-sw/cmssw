#ifndef HEAVYIONSANALYSIS_EGMANALYSIS_GGHINTUPLIZER_H
#define HEAVYIONSANALYSIS_EGMANALYSIS_GGHINTUPLIZER_H

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/HIPhotonIsolation.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
//#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "CommonTools/Egamma/interface/ConversionTools.h"
#include "CommonTools/Egamma/interface/EffectiveAreas.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "HeavyIonsAnalysis/EGMAnalysis/interface/pfIsoCalculator.h"

#include <TTree.h>

class ggHiNtuplizer : public edm::one::EDAnalyzer<> {
public:
  ggHiNtuplizer(const edm::ParameterSet&);
  ~ggHiNtuplizer() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void fillGenPileupInfo(const edm::Event&);
  void fillGenParticles(const edm::Event&);
  void fillSC(const edm::Event&);
  void fillElectrons(const edm::Event&, const edm::EventSetup&, reco::Vertex&);
  void fillPhotons(const edm::Event&, const edm::EventSetup&, reco::Vertex&);
  void fillMuons(const edm::Event&, const edm::EventSetup&, reco::Vertex&);

  template <typename T, typename T2>
  void fillGenCandidates(const edm::Handle<std::vector<T>>& handle,
                         const edm::Handle<edm::View<T2>>& signalPartHandle,
                         bool hasSignalPart = false);

  // Et and pT sums
  template <typename T, typename T2>
  float getGenCalIso(const edm::Handle<edm::View<T>>& handle, T2 thisPart, float dR2Max, bool removeMu, bool removeNu);
  bool getGenCalIsoPass(
      const reco::Candidate* p, const reco::Candidate* thisPart, float dR2Max, bool removeMu, bool removeNu);
  template <typename T, typename T2>
  float getGenTrkIso(const edm::Handle<edm::View<T>>& handle, T2 thisPart, float dR2Max);
  bool getGenTrkIsoPass(const reco::Candidate* p, const reco::Candidate* thisPart, float dR2Max);

  template <typename T>
  inline bool isGenParticle(const T&) const {
    return std::is_base_of<reco::GenParticle, T>::value;
  }
  template <typename T>
  inline bool isPackedGenParticle(const T&) const {
    return std::is_base_of<pat::PackedGenParticle, T>::value;
  }

  // switches
  bool doGenParticles_;
  bool doElectrons_;
  bool doPhotons_;
  bool doMuons_;

  bool doEffectiveAreas_;
  EffectiveAreas effectiveAreas_;

  bool isParticleGun_;
  bool useValMapIso_;
  bool doPfIso_;

  bool doPhoEReg_;
  bool doRecHitsEB_;
  bool doRecHitsEE_;
  bool doSuperClusters_;

  // handles to collections of objects
  edm::EDGetTokenT<std::vector<PileupSummaryInfo>> pileupToken_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> genParticlesToken_;
  edm::EDGetTokenT<std::vector<pat::PackedGenParticle>> packedGenParticlesToken_;
  edm::EDGetTokenT<edm::View<pat::PackedGenParticle>> signalPackedGenParticlesToken_;
  bool doPackedGenParticle_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> vertexToken_;
  edm::EDGetTokenT<double> rhoToken_;
  edm::EDGetTokenT<edm::View<pat::Electron>> electronsToken_;
  edm::EDGetTokenT<edm::View<pat::Photon>> photonsToken_;
  edm::EDGetTokenT<edm::View<pat::Muon>> muonsToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<reco::ConversionCollection> conversionsToken_;
  edm::EDGetTokenT<edm::View<pat::PackedCandidate>> packedPFCandToken_;
  edm::EDGetTokenT<edm::View<reco::PFCandidate>> recoPFCandToken_;
  bool isPackedPFCandidate_;

  const TransientTrackBuilder* tb;

  edm::EDGetTokenT<EcalRecHitCollection> recHitsEB_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitsEE_;

  edm::EDGetTokenT<std::vector<reco::SuperCluster>> scToken_;

  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbESToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;

  const CaloGeometry* geo;

  TTree* tree_;

  UInt_t run_;
  ULong64_t event_;
  UInt_t lumis_;

  float muonPtMin_;
  float rho_;

  // PileupSummaryInfo
  int nPUInfo_;
  std::vector<int> nPU_;
  std::vector<int> puBX_;
  std::vector<float> puTrue_;

  // reco::GenParticle
  int nMC_;

  std::vector<float> mcVtx_x_;
  std::vector<float> mcVtx_y_;
  std::vector<float> mcVtx_z_;

  std::vector<int> mcPID_;
  std::vector<int> mcStatus_;
  std::vector<float> mcPt_;
  std::vector<float> mcEta_;
  std::vector<float> mcPhi_;
  std::vector<float> muL1Eta_;
  std::vector<float> muL1Phi_;
  std::vector<float> mcE_;
  std::vector<float> mcEt_;
  std::vector<float> mcMass_;

  std::vector<int> mcParentage_;
  std::vector<int> mcMomPID_;
  std::vector<int> mcMomKey_;
  std::vector<float> mcMomPt_;
  std::vector<float> mcMomEta_;
  std::vector<float> mcMomPhi_;
  std::vector<float> mcMomMass_;
  std::vector<int> mcGMomPID_;
  std::vector<int> mcGMomKey_;
  std::vector<int> mcGGMomPID_;
  std::vector<int> mcGGMomKey_;

  std::vector<int> mcIndex_;
  std::vector<int> mcSube_;

  std::vector<float> mcCalIsoDR03_;
  std::vector<float> mcCalIsoDR04_;
  std::vector<float> mcTrkIsoDR03_;
  std::vector<float> mcTrkIsoDR04_;

  /* supercluster info */
  int nSC_;
  std::vector<float> scE_;
  std::vector<float> scRawE_;
  std::vector<float> scEta_;
  std::vector<float> scPhi_;

  // electrons
  int nEle_;

  std::vector<float> eleD0_;
  std::vector<float> eleDz_;
  std::vector<float> eleD0Err_;
  std::vector<float> eleDzErr_;
  std::vector<float> eleTrkPt_;
  std::vector<float> eleTrkEta_;
  std::vector<float> eleTrkPhi_;
  std::vector<int> eleTrkCharge_;
  std::vector<float> eleTrkPtErr_;
  std::vector<float> eleTrkChi2_;
  std::vector<float> eleTrkNdof_;
  std::vector<float> eleTrkNormalizedChi2_;
  std::vector<int> eleTrkValidHits_;
  std::vector<int> eleTrkLayers_;
  std::vector<int> eleMissHits_;
  std::vector<float> eleIP3D_;
  std::vector<float> eleIP3DErr_;

  std::vector<float> elePt_;
  std::vector<float> eleRawPt_;
  std::vector<float> eleEta_;
  std::vector<float> elePhi_;
  std::vector<int> eleCharge_;
  std::vector<float> eleEn_;

  std::vector<float> eleSCEn_;
  std::vector<float> eleESEn_;
  std::vector<float> eleSCEta_;
  std::vector<float> eleSCPhi_;
  std::vector<float> eleSCRawEn_;
  std::vector<float> eleSCEtaWidth_;
  std::vector<float> eleSCPhiWidth_;
  std::vector<int> eleSCClustersSize_;
  std::vector<float> eleSeedEn_;
  std::vector<float> eleSeedEta_;
  std::vector<float> eleSeedPhi_;

  std::vector<float> eleHoverE_;
  std::vector<float> eleHoverEBc_;
  std::vector<float> eleRawHoverE_;
  std::vector<float> eleRawHoverEBc_;
  std::vector<float> eleEoverP_;
  std::vector<float> eleEoverPInv_;
  std::vector<float> eleEcalE_;
  std::vector<float> eleRawEcalE_;
  std::vector<float> elePAtVtx_;
  std::vector<float> elePAtSC_;
  std::vector<float> elePAtCluster_;
  std::vector<float> elePAtSeed_;
  std::vector<float> eledEtaAtVtx_;
  std::vector<float> eledPhiAtVtx_;
  std::vector<float> eledEtaSeedAtVtx_;
  std::vector<float> eleSigmaIEtaIEta_;
  std::vector<float> eleSigmaIPhiIPhi_;
  std::vector<float> eleBrem_;

  std::vector<int> eleConvVeto_;

  std::vector<float> eleR9_;
  std::vector<float> eleE3x3_;
  std::vector<float> eleE5x5_;
  std::vector<float> eleR9Full5x5_;
  std::vector<float> eleE3x3Full5x5_;
  std::vector<float> eleE5x5Full5x5_;
  std::vector<float> eleSigmaIEtaIEta_2012_;

  std::vector<float> elePFChIso_;
  std::vector<float> elePFPhoIso_;
  std::vector<float> elePFNeuIso_;
  std::vector<float> elePFPUIso_;

  std::vector<float> elePFRelIsoWithEA_;
  std::vector<float> elePFRelIsoWithDBeta_;
  std::vector<float> eleEffAreaTimesRho_;

  std::vector<float> elePFChIso03_;
  std::vector<float> elePFPhoIso03_;
  std::vector<float> elePFNeuIso03_;
  std::vector<float> elePFChIso04_;
  std::vector<float> elePFPhoIso04_;
  std::vector<float> elePFNeuIso04_;

  std::vector<float> eleSeedCryEta_;
  std::vector<float> eleSeedCryPhi_;
  std::vector<float> eleSeedCryIeta_;
  std::vector<float> eleSeedCryIphi_;

  std::vector<float> eleMVAIso_, eleMVAId_;
  std::map<std::string, std::vector<int>> eleMVAIsoWP_{{{"WP95",{}}, {"WP90",{}}, {"WP85",{}}, {"WP80",{}}}};
  std::map<std::string, std::vector<int>> eleMVAIdWP_ {{{"WP95",{}}, {"WP90",{}}, {"WP85",{}}, {"WP80",{}}}};
  std::map<std::string, std::vector<int>> eleCutIdWP_ {{{"WP95",{}}, {"WP90",{}}, {"WP80",{}}, {"WP70",{}}}};

  std::vector<int> ele_genMatchedIndex_;

  // photons
  int nPho_;

  std::vector<float> phoE_;
  std::vector<float> phoEt_;
  std::vector<float> phoEta_;
  std::vector<float> phoPhi_;

  std::vector<float> phoEcorrStdEcal_;
  std::vector<float> phoEcorrPhoEcal_;
  std::vector<float> phoEcorrRegr1_;
  std::vector<float> phoEcorrRegr2_;
  std::vector<float> phoEcorrErrStdEcal_;
  std::vector<float> phoEcorrErrPhoEcal_;
  std::vector<float> phoEcorrErrRegr1_;
  std::vector<float> phoEcorrErrRegr2_;

  std::vector<float> phoSCE_;
  std::vector<float> phoSCRawE_;
  std::vector<float> phoSCEta_;
  std::vector<float> phoSCPhi_;
  std::vector<float> phoSCEtaWidth_;
  std::vector<float> phoSCPhiWidth_;
  std::vector<float> phoSCBrem_;
  std::vector<int> phoSCnHits_;
  std::vector<uint32_t> phoSCflags_;
  std::vector<int> phoSCinClean_;
  std::vector<int> phoSCinUnClean_;
  std::vector<int> phoSCnBC_;
  std::vector<float> phoESEn_;

  std::vector<int> phoIsPFPhoton_;
  std::vector<int> phoIsStandardPhoton_;
  std::vector<int> phoHasPixelSeed_;
  std::vector<int> phoHasConversionTracks_;
  std::vector<float> phoHadTowerOverEm_;
  std::vector<float> phoHoverE_;
  std::vector<int> phoHoverEValid_;
  std::vector<float> phoSigmaIEtaIEta_;
  std::vector<float> phoR9_;
  std::vector<float> phoE1x5_;
  std::vector<float> phoE2x5_;
  std::vector<float> phoE3x3_;
  std::vector<float> phoE5x5_;
  std::vector<float> phoMaxEnergyXtal_;
  std::vector<float> phoSigmaEtaEta_;

  std::vector<float> phoSigmaIEtaIEta_2012_;
  std::vector<float> phoR9_2012_;
  std::vector<float> phoE1x5_2012_;
  std::vector<float> phoE2x5_2012_;
  std::vector<float> phoE3x3_2012_;
  std::vector<float> phoE5x5_2012_;
  std::vector<float> phoMaxEnergyXtal_2012_;
  std::vector<float> phoSigmaEtaEta_2012_;

  std::vector<float> phoHadTowerOverEm1_;
  std::vector<float> phoHadTowerOverEm2_;
  std::vector<float> phoHoverE1_;
  std::vector<float> phoHoverE2_;

  std::vector<float> phoSigmaIEtaIPhi_;
  std::vector<float> phoSigmaIPhiIPhi_;
  std::vector<float> phoR1x5_;
  std::vector<float> phoR2x5_;
  std::vector<float> phoE2nd_;
  std::vector<float> phoETop_;
  std::vector<float> phoEBottom_;
  std::vector<float> phoELeft_;
  std::vector<float> phoERight_;
  std::vector<float> phoE1x3_;
  std::vector<float> phoE2x2_;
  std::vector<float> phoE2x5Max_;
  std::vector<float> phoE2x5Top_;
  std::vector<float> phoE2x5Bottom_;
  std::vector<float> phoE2x5Left_;
  std::vector<float> phoE2x5Right_;
  //std::vector<float>  phoSMMajor_;   // TODO: enable when they become available in future releases
  //std::vector<float>  phoSMMinor_;   // TODO: enable when they become available in future releases
  //std::vector<float>  phoSMAlpha_;   // TODO: enable when they become available in future releases

  std::vector<float> phoSigmaIEtaIPhi_2012_;
  std::vector<float> phoSigmaIPhiIPhi_2012_;
  std::vector<float> phoR1x5_2012_;
  std::vector<float> phoR2x5_2012_;
  std::vector<float> phoE2nd_2012_;
  std::vector<float> phoETop_2012_;
  std::vector<float> phoEBottom_2012_;
  std::vector<float> phoELeft_2012_;
  std::vector<float> phoERight_2012_;
  std::vector<float> phoE1x3_2012_;
  std::vector<float> phoE2x2_2012_;
  std::vector<float> phoE2x5Max_2012_;
  std::vector<float> phoE2x5Top_2012_;
  std::vector<float> phoE2x5Bottom_2012_;
  std::vector<float> phoE2x5Left_2012_;
  std::vector<float> phoE2x5Right_2012_;
  //std::vector<float>  phoSMMajor_2012_;   // TODO: enable when they become available in future releases
  //std::vector<float>  phoSMMinor_2012_;   // TODO: enable when they become available in future releases
  //std::vector<float>  phoSMAlpha_2012_;   // TODO: enable when they become available in future releases

  std::vector<float> phoBC1E_;
  std::vector<float> phoBC1Ecorr_;
  std::vector<float> phoBC1Eta_;
  std::vector<float> phoBC1Phi_;
  std::vector<int> phoBC1size_;
  std::vector<uint32_t> phoBC1flags_;
  std::vector<int> phoBC1inClean_;
  std::vector<int> phoBC1inUnClean_;
  std::vector<uint32_t> phoBC1rawID_;

  std::vector<float> pho_ecalClusterIsoR1_;
  std::vector<float> pho_ecalClusterIsoR2_;
  std::vector<float> pho_ecalClusterIsoR3_;
  std::vector<float> pho_ecalClusterIsoR4_;
  std::vector<float> pho_ecalClusterIsoR5_;
  std::vector<float> pho_hcalRechitIsoR1_;
  std::vector<float> pho_hcalRechitIsoR2_;
  std::vector<float> pho_hcalRechitIsoR3_;
  std::vector<float> pho_hcalRechitIsoR4_;
  std::vector<float> pho_hcalRechitIsoR5_;
  std::vector<float> pho_trackIsoR1PtCut20_;
  std::vector<float> pho_trackIsoR2PtCut20_;
  std::vector<float> pho_trackIsoR3PtCut20_;
  std::vector<float> pho_trackIsoR4PtCut20_;
  std::vector<float> pho_trackIsoR5PtCut20_;
  std::vector<float> pho_swissCrx_;
  std::vector<float> pho_seedTime_;

  std::vector<int> pho_genMatchedIndex_;

  // rechit info
  int nRH_;
  std::vector<uint32_t> rhRawId_;
  std::vector<int> rhieta_;
  std::vector<int> rhiphi_;
  std::vector<int> rhix_;
  std::vector<int> rhiy_;
  std::vector<float> rhE_;
  std::vector<float> rhEt_;
  std::vector<float> rhEta_;
  std::vector<float> rhPhi_;
  std::vector<float> rhChi2_;
  std::vector<float> rhEerror_;
  std::vector<uint32_t> rhFlags_;
  std::vector<int> rhPhoIdx_;  // index of the photon this rechit belongs to
  std::vector<int> rhBCIdx_;   // index of this rechit's BC in the SC

  // photon pf isolation stuff
  std::vector<float> pfcIso1_;
  std::vector<float> pfcIso2_;
  std::vector<float> pfcIso3_;
  std::vector<float> pfcIso4_;
  std::vector<float> pfcIso5_;

  std::vector<float> pfpIso1_;
  std::vector<float> pfpIso2_;
  std::vector<float> pfpIso3_;
  std::vector<float> pfpIso4_;
  std::vector<float> pfpIso5_;

  std::vector<float> pfnIso1_;
  std::vector<float> pfnIso2_;
  std::vector<float> pfnIso3_;
  std::vector<float> pfnIso4_;
  std::vector<float> pfnIso5_;

  std::vector<float> pfpIso1subSC_;
  std::vector<float> pfpIso2subSC_;
  std::vector<float> pfpIso3subSC_;
  std::vector<float> pfpIso4subSC_;
  std::vector<float> pfpIso5subSC_;

  // photon pf isolation UE-subtracted
  std::vector<float> pfcIso1subUE_;
  std::vector<float> pfcIso2subUE_;
  std::vector<float> pfcIso3subUE_;
  std::vector<float> pfcIso4subUE_;
  std::vector<float> pfcIso5subUE_;

  std::vector<float> pfpIso1subUE_;
  std::vector<float> pfpIso2subUE_;
  std::vector<float> pfpIso3subUE_;
  std::vector<float> pfpIso4subUE_;
  std::vector<float> pfpIso5subUE_;

  std::vector<float> pfnIso1subUE_;
  std::vector<float> pfnIso2subUE_;
  std::vector<float> pfnIso3subUE_;
  std::vector<float> pfnIso4subUE_;
  std::vector<float> pfnIso5subUE_;

  std::vector<float> pfpIso1subSCsubUE_;
  std::vector<float> pfpIso2subSCsubUE_;
  std::vector<float> pfpIso3subSCsubUE_;
  std::vector<float> pfpIso4subSCsubUE_;
  std::vector<float> pfpIso5subSCsubUE_;

  // photon pf isolation with pT cut and UE-subtracted
  std::vector<float> pfcIso1pTgt1p0subUE_;
  std::vector<float> pfcIso2pTgt1p0subUE_;
  std::vector<float> pfcIso3pTgt1p0subUE_;
  std::vector<float> pfcIso4pTgt1p0subUE_;
  std::vector<float> pfcIso5pTgt1p0subUE_;

  std::vector<float> pfcIso1pTgt2p0subUE_;
  std::vector<float> pfcIso2pTgt2p0subUE_;
  std::vector<float> pfcIso3pTgt2p0subUE_;
  std::vector<float> pfcIso4pTgt2p0subUE_;
  std::vector<float> pfcIso5pTgt2p0subUE_;

  // remove if deemed useless
  std::vector<float> pfcIso1pTgt3p0subUE_;
  std::vector<float> pfcIso2pTgt3p0subUE_;
  std::vector<float> pfcIso3pTgt3p0subUE_;
  std::vector<float> pfcIso4pTgt3p0subUE_;
  std::vector<float> pfcIso5pTgt3p0subUE_;
  // remove if deemed useless - END

  // photon pf isolation UE-subtracted and cone excluded
  std::vector<float> pfcIso2subUEec_;
  std::vector<float> pfcIso3subUEec_;
  std::vector<float> pfcIso4subUEec_;

  std::vector<float> pfpIso2subUEec_;
  std::vector<float> pfpIso3subUEec_;
  std::vector<float> pfpIso4subUEec_;

  std::vector<float> pfnIso2subUEec_;
  std::vector<float> pfnIso3subUEec_;
  std::vector<float> pfnIso4subUEec_;

  std::vector<float> pfcIso2pTgt2p0subUEec_;
  std::vector<float> pfcIso3pTgt2p0subUEec_;
  std::vector<float> pfcIso4pTgt2p0subUEec_;

  // muons
  int nMu_;

  std::vector<float> muPt_;
  std::vector<float> muEta_;
  std::vector<float> muPhi_;
  std::vector<int> muCharge_;
  std::vector<int> muType_;
  std::vector<int> muIsGood_;

  std::vector<int> muIsGlobal_;
  std::vector<int> muIsTracker_;
  std::vector<int> muIsPF_;
  std::vector<int> muIsSTA_;

  std::vector<float> muD0_;
  std::vector<float> muDz_;
  std::vector<float> muIP3D_;
  std::vector<float> muD0Err_;
  std::vector<float> muDzErr_;
  std::vector<float> muIP3DErr_;
  std::vector<float> muChi2NDF_;
  std::vector<float> muInnerD0_;
  std::vector<float> muInnerDz_;

  std::vector<float> muInnerD0Err_;
  std::vector<float> muInnerDzErr_;
  std::vector<float> muInnerPt_;
  std::vector<float> muInnerPtErr_;
  std::vector<float> muInnerEta_;

  std::vector<int> muTrkLayers_;
  std::vector<int> muPixelLayers_;
  std::vector<int> muPixelHits_;
  std::vector<int> muMuonHits_;
  std::vector<int> muTrkQuality_;
  std::vector<int> muStations_;
  std::vector<float> muIsoTrk_;
  std::vector<float> muPFChIso_;
  std::vector<float> muPFPhoIso_;
  std::vector<float> muPFNeuIso_;
  std::vector<float> muPFPUIso_;

  std::vector<uint64_t> muSelectors_;
  std::vector<int> muIDSoft_;
  std::vector<int> muIDLoose_;
  std::vector<int> muIDMedium_;
  std::vector<int> muIDMediumPrompt_;
  std::vector<int> muIDTight_;
  std::vector<int> muIDGlobalHighPt_;
  std::vector<int> muIDTrkHighPt_;
  std::vector<int> muIDInTime_;

  std::vector<float> muMVAIso_;
  std::map<std::string, std::vector<int>> muMVAIsoWP_{{{"WP95",{}}, {"WP90",{}}, {"WP85",{}}, {"WP80",{}}}};

  std::vector<int> mu_genMatchedIndex_;
};

#endif /* HEAVYIONSANALYSIS_EGMANALYSIS_GGHINTUPLIZER_H */
