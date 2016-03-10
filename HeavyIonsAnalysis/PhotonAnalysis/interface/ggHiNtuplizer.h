#ifndef ggHiNtuplizer_h
#define ggHiNtuplizer_h

#include "TTree.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/HIPhotonIsolation.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

using namespace std;

class ggHiNtuplizer : public edm::EDAnalyzer {

 public:

   ggHiNtuplizer(const edm::ParameterSet&);
   virtual ~ggHiNtuplizer() {};

 private:

   virtual void analyze(const edm::Event&, const edm::EventSetup&);

   void fillGenParticles (const edm::Event&);
   void fillGenPileupInfo(const edm::Event&);
   void fillElectrons    (const edm::Event&, const edm::EventSetup&, math::XYZPoint& pv);
   void fillPhotons      (const edm::Event&, const edm::EventSetup&, math::XYZPoint& pv);
   void fillMuons        (const edm::Event&, const edm::EventSetup&, math::XYZPoint& pv);

   // Et and pT sums
   float getGenCalIso(edm::Handle<vector<reco::GenParticle> >&, reco::GenParticleCollection::const_iterator, float dRMax, bool removeMu, bool removeNu);
   float getGenTrkIso(edm::Handle<vector<reco::GenParticle> >&, reco::GenParticleCollection::const_iterator, float dRMax);

   // switches
   bool doGenParticles_;
   bool runOnParticleGun_;
   bool useValMapIso_;
   bool doPfIso_;
   bool doVsIso_; // also requires above boolean to make sense
   bool doVID_;

   // handles to collections of objects
   edm::EDGetTokenT<vector<PileupSummaryInfo> >    genPileupCollection_;
   edm::EDGetTokenT<vector<reco::GenParticle> >    genParticlesCollection_;
   edm::EDGetTokenT<edm::View<reco::GsfElectron> > gsfElectronsCollection_;
   edm::EDGetTokenT<edm::ValueMap<bool> > eleVetoIdMapToken_;
   edm::EDGetTokenT<edm::ValueMap<bool> > eleLooseIdMapToken_;
   edm::EDGetTokenT<edm::ValueMap<bool> > eleMediumIdMapToken_;
   edm::EDGetTokenT<edm::ValueMap<bool> > eleTightIdMapToken_;
   edm::EDGetTokenT<edm::View<reco::Photon> >      recoPhotonsCollection_;
   edm::EDGetTokenT<edm::ValueMap<reco::HIPhotonIsolation> > recoPhotonsHiIso_;
   edm::EDGetTokenT<edm::View<reco::Muon> >        recoMuonsCollection_;
   edm::EDGetTokenT<vector<reco::Vertex> >         vtxCollection_;
   edm::EDGetTokenT<double> rhoToken_;
   edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
   edm::EDGetTokenT<reco::ConversionCollection> conversionsToken_;
   edm::EDGetTokenT<edm::View<reco::PFCandidate> >    pfCollection_;
   edm::EDGetTokenT<edm::ValueMap<reco::VoronoiBackground> > voronoiBkgCalo_;
   edm::EDGetTokenT<edm::ValueMap<reco::VoronoiBackground> > voronoiBkgPF_;

   EffectiveAreas effectiveAreas_;

   TTree*         tree_;

   // variables associated with tree branches
   UInt_t          run_;
   ULong64_t       event_;
   UInt_t          lumis_;
   Bool_t         isData_;

   // PileupSummaryInfo
   Int_t          nPUInfo_;
   vector<int>    nPU_;
   vector<int>    puBX_;
   vector<float>  puTrue_;

   // reco::GenParticle
   Int_t          nMC_;
   vector<int>    mcPID_;
   vector<int>    mcStatus_;
   vector<float>  mcVtx_x_;
   vector<float>  mcVtx_y_;
   vector<float>  mcVtx_z_;
   vector<float>  mcPt_;
   vector<float>  mcEta_;
   vector<float>  mcPhi_;
   vector<float>  mcE_;
   vector<float>  mcEt_;
   vector<float>  mcMass_;
   vector<int>    mcParentage_;
   vector<int>    mcMomPID_;
   vector<float>  mcMomPt_;
   vector<float>  mcMomEta_;
   vector<float>  mcMomPhi_;
   vector<float>  mcMomMass_;
   vector<int>    mcGMomPID_;
   vector<int>    mcIndex_;
   vector<float>  mcCalIsoDR03_;
   vector<float>  mcCalIsoDR04_;
   vector<float>  mcTrkIsoDR03_;
   vector<float>  mcTrkIsoDR04_;

   // reco::GsfElectron
   Int_t          nEle_;
   vector<int>    eleCharge_;
   vector<int>    eleChargeConsistent_;
   vector<int>    eleSCPixCharge_;
   vector<int>    eleCtfCharge_;
   vector<float>  eleEn_;
   vector<float>  eleD0_;
   vector<float>  eleDz_;
   vector<float>  eleD0Err_;
   vector<float>  eleDzErr_;
   vector<float>  eleTrkPt_;
   vector<float>  eleTrkEta_;
   vector<float>  eleTrkPhi_;
   vector<int>    eleTrkCharge_;
   vector<float>  eleTrkChi2_;
   vector<float>  eleTrkNdof_;
   vector<float>  eleTrkNormalizedChi2_;
   vector<int>    eleTrkValidHits_;
   vector<int>    eleTrkLayers_;
   vector<float>  elePt_;
   vector<float>  eleEta_;
   vector<float>  elePhi_;
   vector<float>  eleSCEn_;
   vector<float>  eleESEn_;
   vector<float>  eleSCEta_;
   vector<float>  eleSCPhi_;
   vector<float>  eleSCRawEn_;
   vector<float>  eleSCEtaWidth_;
   vector<float>  eleSCPhiWidth_;
   vector<float>  eleHoverE_;
   vector<float>  eleHoverEBc_;
   vector<float>  eleEoverP_;
   vector<float>  eleEoverPInv_;
   vector<float>  eleBrem_;
   vector<float>  eledEtaAtVtx_;
   vector<float>  eledPhiAtVtx_;
   vector<float>  eleSigmaIEtaIEta_;
   vector<float>  eleSigmaIEtaIEta_2012_;
   vector<float>  eleSigmaIPhiIPhi_;
// vector<int>    eleConvVeto_;     // TODO: not available in reco::
   vector<int>    eleMissHits_;
   vector<float>  eleESEffSigmaRR_;
   vector<float>  elePFChIso_;
   vector<float>  elePFPhoIso_;
   vector<float>  elePFNeuIso_;
   vector<float>  elePFPUIso_;
   vector<float>  elePFChIso03_;
   vector<float>  elePFPhoIso03_;
   vector<float>  elePFNeuIso03_;
   vector<float>  elePFChIso04_;
   vector<float>  elePFPhoIso04_;
   vector<float>  elePFNeuIso04_;
   vector<float>  eleR9_;
   vector<float>  eleE3x3_;
   vector<float>  eleE5x5_;
   vector<float>  eleR9Full5x5_;
   vector<float>  eleE3x3Full5x5_;
   vector<float>  eleE5x5Full5x5_;
   vector<int>	  NClusters_;
   vector<int>    NEcalClusters_;
   vector<float>  eleSeedEn_;
   vector<float>  eleSeedEta_;
   vector<float>  eleSeedPhi_;
   vector<float>  eleSeedCryEta_;
   vector<float>  eleSeedCryPhi_;
   vector<float>  eleSeedCryIeta_;
   vector<float>  eleSeedCryIphi_;
   vector<float>  eleBC1E_;
   vector<float>  eleBC1Eta_;
   vector<float>  eleBC2E_;
   vector<float>  eleBC2Eta_;
   vector<int>    eleIDVeto_; //50nsV1 is depreacated; updated in 76X
   vector<int>    eleIDLoose_;
   vector<int>    eleIDMedium_;
   vector<int>    eleIDTight_;
   vector<int>    elepassConversionVeto_;
   vector<float>    eleEffAreaTimesRho_;

   // reco::Photon
   Int_t          nPho_;
   vector<float>  phoE_;
   vector<float>  phoEt_;
   vector<float>  phoEta_;
   vector<float>  phoPhi_;
   vector<float>  phoSCE_;
   vector<float>  phoSCRawE_;
   vector<float>  phoESEn_;
   vector<float>  phoSCEta_;
   vector<float>  phoSCPhi_;
   vector<float>  phoSCEtaWidth_;
   vector<float>  phoSCPhiWidth_;
   vector<float>  phoSCBrem_;
   vector<int>    phohasPixelSeed_;
// vector<int>    phoEleVeto_;         // TODO: not available in reco::
   vector<float>  phoR9_;
   vector<float>  phoHadTowerOverEm_;
   vector<float>  phoHoverE_;
   vector<float>  phoSigmaIEtaIEta_;
// vector<float>  phoSigmaIEtaIPhi_;   // TODO: not available in reco::
// vector<float>  phoSigmaIPhiIPhi_;   // TODO: not available in reco::
   vector<float>  phoE1x3_;
   vector<float>  phoE2x2_;
   vector<float>  phoE3x3_;
   vector<float>  phoE2x5Max_;
   vector<float>  phoE1x5_;
   vector<float>  phoE2x5_;
   vector<float>  phoE5x5_;
   vector<float>  phoMaxEnergyXtal_;
   vector<float>  phoSigmaEtaEta_;
   vector<float>  phoR1x5_;
   vector<float>  phoR2x5_;
   vector<float>  phoESEffSigmaRR_;
   vector<float>  phoSigmaIEtaIEta_2012_;
   vector<float>  phoSigmaIEtaIPhi_2012_;
   vector<float>  phoSigmaIPhiIPhi_2012_;
   vector<float>  phoE1x3_2012_;
   vector<float>  phoE2x2_2012_;
   vector<float>  phoE3x3_2012_;
   vector<float>  phoE2x5Max_2012_;
   vector<float>  phoE5x5_2012_;
   vector<float>  phoBC1E_;
   vector<float>  phoBC1Eta_;
   vector<float>  phoBC2E_;
   vector<float>  phoBC2Eta_;
   vector<float>  pho_ecalClusterIsoR2_;
   vector<float>  pho_ecalClusterIsoR3_;
   vector<float>  pho_ecalClusterIsoR4_;
   vector<float>  pho_ecalClusterIsoR5_;
   vector<float>  pho_hcalRechitIsoR1_;
   vector<float>  pho_hcalRechitIsoR2_;
   vector<float>  pho_hcalRechitIsoR3_;
   vector<float>  pho_hcalRechitIsoR4_;
   vector<float>  pho_hcalRechitIsoR5_;
   vector<float>  pho_trackIsoR1PtCut20_;
   vector<float>  pho_trackIsoR2PtCut20_;
   vector<float>  pho_trackIsoR3PtCut20_;
   vector<float>  pho_trackIsoR4PtCut20_;
   vector<float>  pho_trackIsoR5PtCut20_;
   vector<float>  pho_swissCrx_;
   vector<float>  pho_seedTime_;
   vector<int>    pho_genMatchedIndex_;

   //photon pf isolation stuff
   vector<float> pfcIso1;
   vector<float> pfcIso2;
   vector<float> pfcIso3;
   vector<float> pfcIso4;
   vector<float> pfcIso5;

   vector<float> pfpIso1;
   vector<float> pfpIso2;
   vector<float> pfpIso3;
   vector<float> pfpIso4;
   vector<float> pfpIso5;

   vector<float> pfnIso1;
   vector<float> pfnIso2;
   vector<float> pfnIso3;
   vector<float> pfnIso4;
   vector<float> pfnIso5;

   vector<float> pfsumIso1;
   vector<float> pfsumIso2;
   vector<float> pfsumIso3;
   vector<float> pfsumIso4;
   vector<float> pfsumIso5;

   vector<float> pfcVsIso1;
   vector<float> pfcVsIso2;
   vector<float> pfcVsIso3;
   vector<float> pfcVsIso4;
   vector<float> pfcVsIso5;
   vector<float> pfcVsIso1th1;
   vector<float> pfcVsIso2th1;
   vector<float> pfcVsIso3th1;
   vector<float> pfcVsIso4th1;
   vector<float> pfcVsIso5th1;
   vector<float> pfcVsIso1th2;
   vector<float> pfcVsIso2th2;
   vector<float> pfcVsIso3th2;
   vector<float> pfcVsIso4th2;
   vector<float> pfcVsIso5th2;

   vector<float> pfnVsIso1;
   vector<float> pfnVsIso2;
   vector<float> pfnVsIso3;
   vector<float> pfnVsIso4;
   vector<float> pfnVsIso5;
   vector<float> pfnVsIso1th1;
   vector<float> pfnVsIso2th1;
   vector<float> pfnVsIso3th1;
   vector<float> pfnVsIso4th1;
   vector<float> pfnVsIso5th1;
   vector<float> pfnVsIso1th2;
   vector<float> pfnVsIso2th2;
   vector<float> pfnVsIso3th2;
   vector<float> pfnVsIso4th2;
   vector<float> pfnVsIso5th2;


   vector<float> pfpVsIso1;
   vector<float> pfpVsIso2;
   vector<float> pfpVsIso3;
   vector<float> pfpVsIso4;
   vector<float> pfpVsIso5;
   vector<float> pfpVsIso1th1;
   vector<float> pfpVsIso2th1;
   vector<float> pfpVsIso3th1;
   vector<float> pfpVsIso4th1;
   vector<float> pfpVsIso5th1;
   vector<float> pfpVsIso1th2;
   vector<float> pfpVsIso2th2;
   vector<float> pfpVsIso3th2;
   vector<float> pfpVsIso4th2;
   vector<float> pfpVsIso5th2;


   vector<float> pfsumVsIso1;
   vector<float> pfsumVsIso2;
   vector<float> pfsumVsIso3;
   vector<float> pfsumVsIso4;
   vector<float> pfsumVsIso5;
   vector<float> pfsumVsIso1th1;
   vector<float> pfsumVsIso2th1;
   vector<float> pfsumVsIso3th1;
   vector<float> pfsumVsIso4th1;
   vector<float> pfsumVsIso5th1;
   vector<float> pfsumVsIso1th2;
   vector<float> pfsumVsIso2th2;
   vector<float> pfsumVsIso3th2;
   vector<float> pfsumVsIso4th2;
   vector<float> pfsumVsIso5th2;


   vector<float> pfVsSubIso1;
   vector<float> pfVsSubIso2;
   vector<float> pfVsSubIso3;
   vector<float> pfVsSubIso4;
   vector<float> pfVsSubIso5;

   vector<float> towerIso1;
   vector<float> towerIso2;
   vector<float> towerIso3;
   vector<float> towerIso4;
   vector<float> towerIso5;
   vector<float> towerVsIso1;
   vector<float> towerVsIso2;
   vector<float> towerVsIso3;
   vector<float> towerVsIso4;
   vector<float> towerVsIso5;
   vector<float> towerVsSubIso1;
   vector<float> towerVsSubIso2;
   vector<float> towerVsSubIso3;
   vector<float> towerVsSubIso4;
   vector<float> towerVsSubIso5;


   // reco::Muon
   Int_t          nMu_;
   vector<float>  muPt_;
   vector<float>  muEta_;
   vector<float>  muPhi_;
   vector<int>    muCharge_;
   vector<int>    muType_;
   vector<int>    muIsGood_;
   vector<float>  muD0_;
   vector<float>  muDz_;
   vector<float>  muChi2NDF_;
   vector<float>  muInnerD0_;
   vector<float>  muInnerDz_;
   vector<int>    muTrkLayers_;
   vector<int>    muPixelLayers_;
   vector<int>    muPixelHits_;
   vector<int>    muMuonHits_;
   vector<int>    muTrkQuality_;
   vector<int>    muStations_;
   vector<float>  muIsoTrk_;
   vector<float>  muPFChIso_;
   vector<float>  muPFPhoIso_;
   vector<float>  muPFNeuIso_;
   vector<float>  muPFPUIso_;
};

#endif
