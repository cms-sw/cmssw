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
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
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
   void fillPhotons      (const edm::Event&, const edm::EventSetup&);
   void fillMuons        (const edm::Event&, const edm::EventSetup&, math::XYZPoint& pv);

   // Et and pT sums
   float getGenCalIso(edm::Handle<vector<reco::GenParticle> >&, reco::GenParticleCollection::const_iterator, float dRMax, bool removeMu, bool removeNu);
   float getGenTrkIso(edm::Handle<vector<reco::GenParticle> >&, reco::GenParticleCollection::const_iterator, float dRMax);

   // switches
   bool doGenParticles_;
   bool runOnParticleGun_;

   // handles to collections of objects
   edm::EDGetTokenT<vector<PileupSummaryInfo> >    genPileupCollection_;
   edm::EDGetTokenT<vector<reco::GenParticle> >    genParticlesCollection_;
   edm::EDGetTokenT<edm::View<reco::GsfElectron> > gsfElectronsCollection_;
   edm::EDGetTokenT<edm::View<reco::Photon> >      recoPhotonsCollection_;
   edm::EDGetTokenT<edm::View<reco::Muon> >        recoMuonsCollection_;
   edm::EDGetTokenT<EcalRecHitCollection>          ebRecHitCollection_;
   edm::EDGetTokenT<EcalRecHitCollection>          eeRecHitCollection_;
   edm::EDGetTokenT<vector<reco::Vertex> >         vtxCollection_;

   TTree*         tree_;

   // variables associated with tree branches
   Int_t          run_;
   Long64_t       event_;
   Int_t          lumis_;
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
   vector<float>  eleEn_;
   vector<float>  eleD0_;
   vector<float>  eleDz_;
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
   vector<float>  eleBC1E_;
   vector<float>  eleBC1Eta_;
   vector<float>  eleBC2E_;
   vector<float>  eleBC2Eta_;

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
   vector<float>  phoHoverE_;
// vector<float>  phoSigmaIEtaIEta_;   // TODO: not available in reco::
// vector<float>  phoSigmaIEtaIPhi_;   // TODO: not available in reco::
// vector<float>  phoSigmaIPhiIPhi_;   // TODO: not available in reco::
   vector<float>  phoE1x3_;
   vector<float>  phoE2x2_;
   vector<float>  phoE2x5Max_;
   vector<float>  phoE5x5_;
   vector<float>  phoESEffSigmaRR_;
   vector<float>  phoSigmaIEtaIEta_2012_;
   vector<float>  phoSigmaIEtaIPhi_2012_;
   vector<float>  phoSigmaIPhiIPhi_2012_;
   vector<float>  phoE1x3_2012_;
   vector<float>  phoE2x2_2012_;
   vector<float>  phoE2x5Max_2012_;
   vector<float>  phoE5x5_2012_;
   vector<float>  phoBC1E_;
   vector<float>  phoBC1Eta_;
   vector<float>  phoBC2E_;
   vector<float>  phoBC2Eta_;

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
