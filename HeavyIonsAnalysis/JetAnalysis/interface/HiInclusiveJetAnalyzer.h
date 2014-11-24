#ifndef MNguyen_HiInclusiveJetAnalyzer_inclusiveJetAnalyzer_
#define MNguyen_HiInclusiveJetAnalyzer_inclusiveJetAnalyzer_

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "RecoHI/HiCentralityAlgos/interface/CentralityProvider.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
//

/**\class HiInclusiveJetAnalyzer

   \author Matt Nguyen
   \date   November 2010
*/




class HiInclusiveJetAnalyzer : public edm::EDAnalyzer {
public:

  explicit HiInclusiveJetAnalyzer(const edm::ParameterSet&);

  ~HiInclusiveJetAnalyzer();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginRun(const edm::Run & r, const edm::EventSetup & c);

  virtual void beginJob();

  void fillL1Bits(const edm::Event &iEvent);

  void fillHLTBits(const edm::Event &iEvent);

  template <typename TYPE>
    void getProduct(const std::string name, edm::Handle<TYPE> &prod,
		    const edm::Event &event) const;
  template <typename TYPE>
    bool getProductSafe(const std::string name, edm::Handle<TYPE> &prod,
			const edm::Event &event) const;


private:

  int getPFJetMuon(const pat::Jet& pfJet, const reco::PFCandidateCollection *pfCandidateColl);

  double getPtRel(const reco::PFCandidate lep, const pat::Jet& jet );

  void saveDaughters( const reco::GenParticle & gen);
  void saveDaughters( const reco::Candidate & gen);
  double getEt(math::XYZPoint pos, double energy);
  math::XYZPoint getPosition(const DetId &id, reco::Vertex::Point vtx = reco::Vertex::Point(0,0,0));
  int TaggedJet(reco::Jet calojet, edm::Handle<reco::JetTagCollection > jetTags );

  edm::InputTag   jetTag_, vtxTag_, genjetTag_, eventInfoTag_, L1gtReadout_, pfCandidateLabel_, trackTag_, matchTag_;
  edm::InputTag HcalRecHitHFSrc_;
  edm::InputTag HcalRecHitHBHESrc_;
  edm::InputTag EBSrc_;
  edm::InputTag EESrc_;
  edm::InputTag genParticleSrc_;

  std::vector<float> usedStringPts;

  /// verbose ?
  bool verbose_;
  bool doMatch_;
  bool useCentrality_;
  bool useVtx_;
  bool useJEC_;
  bool usePat_;
  bool isMC_;
  bool fillGenJets_;
  bool doTrigger_;
  bool useQuality_;
  std::string trackQuality_;

  bool doSubEvent_;
  double genPtMin_;
  bool doLifeTimeTagging_;
  bool doLifeTimeTaggingExtras_;
  bool saveBfragments_;
  bool skipCorrections_;

  bool doHiJetID_;
  bool doStandardJetID_;

  double rParam;
  double hardPtMin_;
  double jetPtMin_;

  TTree *t;
  edm::Service<TFileService> fs1;

  CentralityProvider * centrality_;
  const CaloGeometry *geo;

  std::string                   hltResName_;         //HLT trigger results name
  std::vector<std::string>      hltProcNames_;       //HLT process name(s)
  std::vector<std::string>      hltTrgNames_;        //HLT trigger name(s)

  std::vector<int>              hltTrgBits_;         //HLT trigger bit(s)
  std::vector<bool>             hltTrgDeci_;         //HLT trigger descision(s)
  std::vector<std::string>      hltTrgUsedNames_;    //HLT used trigger name(s)
  std::string                   hltUsedResName_;     //used HLT trigger results name

  std::string bTagJetName_;
  std::string ImpactParameterTagInfos_;
  std::string TrackCountingHighEffBJetTags_;
  std::string NegativeTrackCountingHighEffJetTags_;
  std::string TrackCountingHighPurBJetTags_;
  std::string NegativeTrackCountingHighPur_;
  std::string JetProbabilityBJetTags_;
  std::string PositiveOnlyJetProbabilityJetTags_;
  std::string NegativeOnlyJetProbabilityJetTags_;
  std::string JetBProbabilityBJetTags_;
  std::string NegativeOnlyJetBProbabilityJetTags_;
  std::string PositiveOnlyJetBProbabilityJetTags_;
  std::string SecondaryVertexTagInfos_;
  std::string SecondaryVertexNegativeTagInfos_;
  std::string SimpleSecondaryVertexHighEffBJetTags_;
  std::string SimpleSecondaryVertexNegativeHighEffBJetTags_;
  std::string SimpleSecondaryVertexHighPurBJetTags_;
  std::string SimpleSecondaryVertexNegativeHighPurBJetTags_;
  std::string CombinedSecondaryVertexBJetTags_;
  std::string CombinedSecondaryVertexNegativeBJetTags_;
  std::string CombinedSecondaryVertexPositiveBJetTags_;
  std::string NegativeSoftMuonByPtBJetTags_;
  std::string PositiveSoftMuonByPtBJetTags_;

  static const int MAXJETS = 500;
  static const int MAXTRACKS = 5000;
  static const int MAXHLTBITS = 5000;
  static const int MAXBFRAG = 500;

  struct JRA{

    int nref;
    int run;
    int evt;
    int lumi;
    int bin;
    float vx, vy, vz;
    float b;
    float hf;

    float rawpt[MAXJETS];
    float jtpt[MAXJETS];
    float jteta[MAXJETS];
    float jtphi[MAXJETS];
    float jty[MAXJETS];
    float jtpu[MAXJETS];
    float jtm[MAXJETS];

    float trackMax[MAXJETS];
    float trackSum[MAXJETS];
    int trackN[MAXJETS];

    float chargedMax[MAXJETS];
    float chargedSum[MAXJETS];
    int chargedN[MAXJETS];

    float photonMax[MAXJETS];
    float photonSum[MAXJETS];
    int photonN[MAXJETS];

    float trackHardSum[MAXJETS];
    float chargedHardSum[MAXJETS];
    float photonHardSum[MAXJETS];

    int trackHardN[MAXJETS];
    int chargedHardN[MAXJETS];
    int photonHardN[MAXJETS];

    float neutralMax[MAXJETS];
    float neutralSum[MAXJETS];
    int neutralN[MAXJETS];

    float eMax[MAXJETS];
    float eSum[MAXJETS];
    int eN[MAXJETS];

    float muMax[MAXJETS];
    float muSum[MAXJETS];
    int muN[MAXJETS];

    float genChargedSum[MAXJETS];
    float genHardSum[MAXJETS];
    float signalChargedSum[MAXJETS];
    float signalHardSum[MAXJETS];

    float hcalSum[MAXJETS];
    float ecalSum[MAXJETS];


    float fHPD[MAXJETS];
    float fRBX[MAXJETS];
    int n90[MAXJETS];
    float fSubDet1[MAXJETS];
    float fSubDet2[MAXJETS];
    float fSubDet3[MAXJETS];
    float fSubDet4[MAXJETS];
    float restrictedEMF[MAXJETS];
    int nHCAL[MAXJETS];
    int nECAL[MAXJETS];
    float apprHPD[MAXJETS];
    float apprRBX[MAXJETS];

    //    int n90[MAXJETS];
    int n2RPC[MAXJETS];
    int n3RPC[MAXJETS];
    int nRPC[MAXJETS];

    float fEB[MAXJETS];
    float fEE[MAXJETS];
    float fHB[MAXJETS];
    float fHE[MAXJETS];
    float fHO[MAXJETS];
    float fLong[MAXJETS];
    float fShort[MAXJETS];
    float fLS[MAXJETS];
    float fHFOOT[MAXJETS];


    int subid[MAXJETS];

    float matchedPt[MAXJETS];
    float matchedRawPt[MAXJETS];
    float matchedR[MAXJETS];
    float matchedPu[MAXJETS];

    float discr_csvMva[MAXJETS];
    float discr_csvSimple[MAXJETS];
    float discr_muByIp3[MAXJETS];
    float discr_muByPt[MAXJETS];
    float discr_prob[MAXJETS];
    float discr_probb[MAXJETS];
    float discr_tcHighEff[MAXJETS];
    float discr_tcHighPur[MAXJETS];
    float discr_ssvHighEff[MAXJETS];
    float discr_ssvHighPur[MAXJETS];

    float ndiscr_ssvHighEff[MAXJETS];
    float ndiscr_ssvHighPur[MAXJETS];
    float ndiscr_csvSimple[MAXJETS];
    float ndiscr_muByPt[MAXJETS];
    float ndiscr_prob[MAXJETS];
    float ndiscr_probb[MAXJETS];
    float ndiscr_tcHighEff[MAXJETS];
    float ndiscr_tcHighPur[MAXJETS];

    float pdiscr_csvSimple[MAXJETS];
    float pdiscr_prob[MAXJETS];
    float pdiscr_probb[MAXJETS];

    int nsvtx[MAXJETS];
    int svtxntrk[MAXJETS];
    float svtxdl[MAXJETS];
    float svtxdls[MAXJETS];
    float svtxm[MAXJETS];
    float svtxpt[MAXJETS];
    float svtxnormchi2[MAXJETS];

    int nIPtrk[MAXJETS];
    int nselIPtrk[MAXJETS];

    int nIP;
    int ipJetIndex[MAXTRACKS];
    float ipPt[MAXTRACKS];
    float ipEta[MAXTRACKS];
    float ipDxy[MAXTRACKS];
    float ipDz[MAXTRACKS];
    float ipChi2[MAXTRACKS];
    int ipNHit[MAXTRACKS];
    int ipNHitPixel[MAXTRACKS];
    int ipNHitStrip[MAXTRACKS];
    bool ipIsHitL1[MAXTRACKS];
    float ipProb0[MAXTRACKS];
    float ipProb1[MAXTRACKS];
    float ip2d[MAXTRACKS];
    float ip2dSig[MAXTRACKS];
    float ip3d[MAXTRACKS];
    float ip3dSig[MAXTRACKS];
    float ipDist2Jet[MAXTRACKS];
    float ipDist2JetSig[MAXTRACKS];
    float ipClosest2Jet[MAXTRACKS];

    float mue[MAXJETS];
    float mupt[MAXJETS];
    float mueta[MAXJETS];
    float muphi[MAXJETS];
    float mudr[MAXJETS];
    float muptrel[MAXJETS];
    int muchg[MAXJETS];

    float discr_fr01[MAXJETS];

    float refpt[MAXJETS];
    float refeta[MAXJETS];
    float refphi[MAXJETS];
    float refy[MAXJETS];
    float refdphijt[MAXJETS];
    float refdrjt[MAXJETS];
    float refparton_pt[MAXJETS];
    int refparton_flavor[MAXJETS];
    int refparton_flavorForB[MAXJETS];

    float pthat;
    int beamId1, beamId2;
    int ngen;
    int genmatchindex[MAXJETS];
    float genpt[MAXJETS];
    float geneta[MAXJETS];
    float genphi[MAXJETS];
    float geny[MAXJETS];
    float gendphijt[MAXJETS];
    float gendrjt[MAXJETS];
    int gensubid[MAXJETS];

    // hlt
    int nHLTBit;
    bool hltBit[MAXHLTBITS];

    // l1
    int nL1TBit;
    bool l1TBit[MAXHLTBITS];
    int nL1ABit;
    bool l1ABit[MAXHLTBITS];

    int bMult;
    int bJetIndex[MAXBFRAG];
    int bStatus[MAXBFRAG];
    int bPdg[MAXBFRAG];
    int bChg[MAXBFRAG];
    float bVx[MAXBFRAG];
    float bVy[MAXBFRAG];
    float bVz[MAXBFRAG];
    float bPt[MAXBFRAG];
    float bEta[MAXBFRAG];
    float bPhi[MAXBFRAG];


  };

  JRA jets_;

};

#endif
