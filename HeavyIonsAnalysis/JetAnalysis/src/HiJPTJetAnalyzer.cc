/*
  Based on the jet response analyzer
  Modified by Matt Nguyen, November 2010

*/

#include "HeavyIonsAnalysis/JetAnalysis/interface/HiJPTJetAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <Math/DistFunc.h>
#include "TMath.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "L1Trigger/GlobalTrigger/plugins/L1GlobalTrigger.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

using namespace std;
using namespace edm;
using namespace reco;

HiJPTJetAnalyzer::HiJPTJetAnalyzer(const edm::ParameterSet& iConfig) :
  geo(0)
{

   doMatch_ = iConfig.getUntrackedParameter<bool>("matchJets",false);
  jetTag_ = iConfig.getParameter<InputTag>("jetTag");
  matchTag_ = iConfig.getUntrackedParameter<InputTag>("matchTag",jetTag_);

  vtxTag_ = iConfig.getUntrackedParameter<edm::InputTag>("vtxTag",edm::InputTag("hiSelectedVertex"));
  trackTag_ = iConfig.getParameter<InputTag>("trackTag");
  useQuality_ = iConfig.getUntrackedParameter<bool>("useQuality",1);
  trackQuality_ = iConfig.getUntrackedParameter<string>("trackQuality","highPurity");

  isMC_ = iConfig.getUntrackedParameter<bool>("isMC",false);
  fillGenJets_ = iConfig.getUntrackedParameter<bool>("fillGenJets",false);

  doTrigger_ = iConfig.getUntrackedParameter<bool>("doTrigger",false);
  doHiJetID_ = iConfig.getUntrackedParameter<bool>("doHiJetID",false);
  doStandardJetID_ = iConfig.getUntrackedParameter<bool>("doStandardJetID",false);

  rParam = iConfig.getParameter<double>("rParam");
  hardPtMin_ = iConfig.getUntrackedParameter<double>("hardPtMin",4);
  jetPtMin_ = iConfig.getUntrackedParameter<double>("jetPtMin",0);

  if(isMC_){
    genjetTag_ = iConfig.getParameter<InputTag>("genjetTag");
    eventInfoTag_ = iConfig.getParameter<InputTag>("eventInfoTag");
  }
  verbose_ = iConfig.getUntrackedParameter<bool>("verbose",false);

  useVtx_ = iConfig.getUntrackedParameter<bool>("useVtx",false);
  useJEC_ = iConfig.getUntrackedParameter<bool>("useJEC",true);
  usePat_ = iConfig.getUntrackedParameter<bool>("usePAT",true);

  doLifeTimeTagging_ = iConfig.getUntrackedParameter<bool>("doLifeTimeTagging",false);
  doLifeTimeTaggingExtras_ = iConfig.getUntrackedParameter<bool>("doLifeTimeTaggingExtras",true);
  saveBfragments_  = iConfig.getUntrackedParameter<bool>("saveBfragments",false);
  skipCorrections_  = iConfig.getUntrackedParameter<bool>("skipCorrections",false);

  pfCandidateLabel_ = iConfig.getUntrackedParameter<edm::InputTag>("pfCandidateLabel",edm::InputTag("particleFlowTmp"));

  EBSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("EBRecHitSrc",edm::InputTag("ecalRecHit","EcalRecHitsEB"));
  EESrc_ = iConfig.getUntrackedParameter<edm::InputTag>("EERecHitSrc",edm::InputTag("ecalRecHit","EcalRecHitsEE"));
  HcalRecHitHFSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("hcalHFRecHitSrc",edm::InputTag("hfreco"));
  HcalRecHitHBHESrc_ = iConfig.getUntrackedParameter<edm::InputTag>("hcalHBHERecHitSrc",edm::InputTag("hbhereco"));

  genParticleSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("genParticles",edm::InputTag("hiGenParticles"));

  if(doTrigger_){
    L1gtReadout_ = iConfig.getParameter<edm::InputTag>("L1gtReadout");
    hltResName_ = iConfig.getUntrackedParameter<string>("hltTrgResults","TriggerResults::HLT");


    if (iConfig.exists("hltTrgNames"))
      hltTrgNames_ = iConfig.getUntrackedParameter<vector<string> >("hltTrgNames");

    if (iConfig.exists("hltProcNames"))
      hltProcNames_ = iConfig.getUntrackedParameter<vector<string> >("hltProcNames");
    else {
      hltProcNames_.push_back("FU");
      hltProcNames_.push_back("HLT");
    }
  }

  //  cout<<" jet collection : "<<jetTag_<<endl;
  doSubEvent_ = 0;

  if(isMC_){
    //     cout<<" genjet collection : "<<genjetTag_<<endl;
     genPtMin_ = iConfig.getUntrackedParameter<double>("genPtMin",0);
     doSubEvent_ = iConfig.getUntrackedParameter<bool>("doSubEvent",1);
  }


}



HiJPTJetAnalyzer::~HiJPTJetAnalyzer() { }



void
HiJPTJetAnalyzer::beginRun(const edm::Run& run,
			      const edm::EventSetup & es) {}

void
HiJPTJetAnalyzer::beginJob() {

  //string jetTagName = jetTag_.label()+"_tree";
  string jetTagTitle = jetTag_.label()+" Jet Analysis Tree";
  t = fs1->make<TTree>("t",jetTagTitle.c_str());

  //  TTree* t= new TTree("t","Jet Response Analyzer");
  //t->Branch("run",&jets_.run,"run/I");
  t->Branch("evt",&jets_.evt,"evt/I");
  //t->Branch("lumi",&jets_.lumi,"lumi/I");
  t->Branch("b",&jets_.b,"b/F");
  if (useVtx_) {
     t->Branch("vx",&jets_.vx,"vx/F");
     t->Branch("vy",&jets_.vy,"vy/F");
     t->Branch("vz",&jets_.vz,"vz/F");
  }

  t->Branch("nref",&jets_.nref,"nref/I");
  t->Branch("rawpt",jets_.rawpt,"rawpt[nref]/F");
  if(!skipCorrections_) t->Branch("jtpt",jets_.jtpt,"jtpt[nref]/F");
  t->Branch("jteta",jets_.jteta,"jteta[nref]/F");
  t->Branch("jty",jets_.jty,"jty[nref]/F");
  t->Branch("jtphi",jets_.jtphi,"jtphi[nref]/F");
  t->Branch("jtpu",jets_.jtpu,"jtpu[nref]/F");
  t->Branch("jtm",jets_.jtm,"jtm[nref]/F");


// Tracks, associated with jpt jets
   t->Branch("pion_bgn",jets_.pion_bgn, "pion_bgn[nref]/I");
   t->Branch("pion_end",jets_.pion_end, "pion_end[nref]/I");
   t->Branch("muon_bgn", jets_.muon_bgn, "muon_bgn[nref]/I");
   t->Branch("muon_end",  jets_.muon_end, "muon_end[nref]/I");
   t->Branch("elecs_bgn",  jets_.elecs_bgn, "elecs_bgn[nref]/I");
   t->Branch("elecs_end",  jets_.elecs_end, "elecs_end[nref]/I");
   t->Branch("ntrack",&jets_.ntrack,"ntrack/I");
   t->Branch("jpttrackpt", jets_.jpttrackpt, "jpttrackpt[ntrack]/F");
   t->Branch("jpttrackphi", jets_.jpttrackphi, "jpttrackphi[ntrack]/F");
   t->Branch("jpttracketa", jets_.jpttracketa, "jpttracketa[ntrack]/F");


  // jet ID information, jet composition
  if(doHiJetID_){
  t->Branch("discr_fr01", jets_.discr_fr01,"discr_fr01[nref]/F");

  t->Branch("trackMax", jets_.trackMax,"trackMax[nref]/F");
  t->Branch("trackSum", jets_.trackSum,"trackSum[nref]/F");
  t->Branch("trackN", jets_.trackN,"trackN[nref]/I");
  t->Branch("trackHardSum", jets_.trackHardSum,"trackHardSum[nref]/F");
  t->Branch("trackHardN", jets_.trackHardN,"trackHardN[nref]/I");

  t->Branch("chargedMax", jets_.chargedMax,"chargedMax[nref]/F");
  t->Branch("chargedSum", jets_.chargedSum,"chargedSum[nref]/F");
  t->Branch("chargedN", jets_.chargedN,"chargedN[nref]/I");
  t->Branch("chargedHardSum", jets_.chargedHardSum,"chargedHardSum[nref]/F");
  t->Branch("chargedHardN", jets_.chargedHardN,"chargedHardN[nref]/I");

  t->Branch("photonMax", jets_.photonMax,"photonMax[nref]/F");
  t->Branch("photonSum", jets_.photonSum,"photonSum[nref]/F");
  t->Branch("photonN", jets_.photonN,"photonN[nref]/I");
  t->Branch("photonHardSum", jets_.photonHardSum,"photonHardSum[nref]/F");
  t->Branch("photonHardN", jets_.photonHardN,"photonHardN[nref]/I");

  t->Branch("neutralMax", jets_.neutralMax,"neutralMax[nref]/F");
  t->Branch("neutralSum", jets_.neutralSum,"neutralSum[nref]/F");
  t->Branch("neutralN", jets_.neutralN,"neutralN[nref]/I");

  t->Branch("hcalSum", jets_.hcalSum,"hcalSum[nref]/F");
  t->Branch("ecalSum", jets_.ecalSum,"ecalSum[nref]/F");

  t->Branch("eMax", jets_.eMax,"eMax[nref]/F");
  t->Branch("eSum", jets_.eSum,"eSum[nref]/F");
  t->Branch("eN", jets_.eN,"eN[nref]/I");

  t->Branch("muMax", jets_.muMax,"muMax[nref]/F");
  t->Branch("muSum", jets_.muSum,"muSum[nref]/F");
  t->Branch("muN", jets_.muN,"muN[nref]/I");
  }

  if(doStandardJetID_){
  t->Branch("fHPD",jets_.fHPD,"fHPD[nref]/F");
  t->Branch("fRBX",jets_.fRBX,"fRBX[nref]/F");
  t->Branch("n90",jets_.n90,"n90[nref]/I");
  t->Branch("fSubDet1",jets_.fSubDet1,"fSubDet1[nref]/F");
  t->Branch("fSubDet2",jets_.fSubDet2,"fSubDet2[nref]/F");
  t->Branch("fSubDet3",jets_.fSubDet3,"fSubDet3[nref]/F");
  t->Branch("fSubDet4",jets_.fSubDet4,"fSubDet4[nref]/F");
  t->Branch("restrictedEMF",jets_.restrictedEMF,"restrictedEMF[nref]/F");
  t->Branch("nHCAL",jets_.nHCAL,"nHCAL[nref]/I");
  t->Branch("nECAL",jets_.nECAL,"nECAL[nref]/I");
  t->Branch("apprHPD",jets_.apprHPD,"apprHPD[nref]/F");
  t->Branch("apprRBX",jets_.apprRBX,"apprRBX[nref]/F");

  //  t->Branch("hitsInN90",jets_.n90,"hitsInN90[nref]");
  t->Branch("n2RPC",jets_.n2RPC,"n2RPC[nref]/I");
  t->Branch("n3RPC",jets_.n3RPC,"n3RPC[nref]/I");
  t->Branch("nRPC",jets_.nRPC,"nRPC[nref]/I");

  t->Branch("fEB",jets_.fEB,"fEB[nref]/F");
  t->Branch("fEE",jets_.fEE,"fEE[nref]/F");
  t->Branch("fHB",jets_.fHB,"fHB[nref]/F");
  t->Branch("fHE",jets_.fHE,"fHE[nref]/F");
  t->Branch("fHO",jets_.fHO,"fHO[nref]/F");
  t->Branch("fLong",jets_.fLong,"fLong[nref]/F");
  t->Branch("fShort",jets_.fShort,"fShort[nref]/F");
  t->Branch("fLS",jets_.fLS,"fLS[nref]/F");
  t->Branch("fHFOOT",jets_.fHFOOT,"fHFOOT[nref]/F");
  }

  // Jet ID
  if(doMatch_){
  if(!skipCorrections_) t->Branch("matchedPt", jets_.matchedPt,"matchedPt[nref]/F");
  t->Branch("matchedRawPt", jets_.matchedRawPt,"matchedRawPt[nref]/F");
  t->Branch("matchedPu", jets_.matchedPu,"matchedPu[nref]/F");
  t->Branch("matchedR", jets_.matchedR,"matchedR[nref]/F");
  }

  // b-jet discriminators
  if (doLifeTimeTagging_) {

    t->Branch("discr_csvMva",jets_.discr_csvMva,"discr_csvMva[nref]/F");
    t->Branch("discr_csvSimple",jets_.discr_csvSimple,"discr_csvSimple[nref]/F");
    t->Branch("discr_muByIp3",jets_.discr_muByIp3,"discr_muByIp3[nref]/F");
    t->Branch("discr_muByPt",jets_.discr_muByPt,"discr_muByPt[nref]/F");
    t->Branch("discr_prob",jets_.discr_prob,"discr_prob[nref]/F");
    t->Branch("discr_probb",jets_.discr_probb,"discr_probb[nref]/F");
    t->Branch("discr_tcHighEff",jets_.discr_tcHighEff,"discr_tcHighEff[nref]/F");
    t->Branch("discr_tcHighPur",jets_.discr_tcHighPur,"discr_tcHighPur[nref]/F");

    t->Branch("nsvtx",    jets_.nsvtx,    "nsvtx[nref]/I");
    t->Branch("svtxntrk", jets_.svtxntrk, "svtxntrk[nref]/I");
    t->Branch("svtxdl",   jets_.svtxdl,   "svtxdl[nref]/F");
    t->Branch("svtxdls",  jets_.svtxdls,  "svtxdls[nref]/F");
    t->Branch("svtxm",    jets_.svtxm,    "svtxm[nref]/F");
    t->Branch("svtxpt",   jets_.svtxpt,   "svtxpt[nref]/F");

    t->Branch("nIPtrk",jets_.nIPtrk,"nIPtrk[nref]/I");
    t->Branch("nselIPtrk",jets_.nselIPtrk,"nselIPtrk[nref]/I");

    if (doLifeTimeTaggingExtras_) {
      t->Branch("nIP",&jets_.nIP,"nIP/I");
      t->Branch("ipJetIndex",jets_.ipJetIndex,"ipJetIndex[nIP]/I");
      t->Branch("ipPt",jets_.ipPt,"ipPt[nIP]/F");
      t->Branch("ipProb0",jets_.ipProb0,"ipProb0[nIP]/F");
      t->Branch("ipProb1",jets_.ipProb1,"ipProb1[nIP]/F");
      t->Branch("ip2d",jets_.ip2d,"ip2d[nIP]/F");
      t->Branch("ip2dSig",jets_.ip2dSig,"ip2dSig[nIP]/F");
      t->Branch("ip3d",jets_.ip3d,"ip3d[nIP]/F");
      t->Branch("ip3dSig",jets_.ip3dSig,"ip3dSig[nIP]/F");
      t->Branch("ipDist2Jet",jets_.ipDist2Jet,"ipDist2Jet[nIP]/F");
      t->Branch("ipDist2JetSig",jets_.ipDist2JetSig,"ipDist2JetSig[nIP]/F");
      t->Branch("ipClosest2Jet",jets_.ipClosest2Jet,"ipClosest2Jet[nIP]/F");

    }

    t->Branch("mue",     jets_.mue,     "mue[nref]/F");
    t->Branch("mupt",    jets_.mupt,    "mupt[nref]/F");
    t->Branch("mueta",   jets_.mueta,   "mueta[nref]/F");
    t->Branch("muphi",   jets_.muphi,   "muphi[nref]/F");
    t->Branch("mudr",    jets_.mudr,    "mudr[nref]/F");
    t->Branch("muptrel", jets_.muptrel, "muptrel[nref]/F");
    t->Branch("muchg",   jets_.muchg,   "muchg[nref]/I");
  }


  if(isMC_){
    t->Branch("beamId1",&jets_.beamId1,"beamId1/I");
    t->Branch("beamId2",&jets_.beamId2,"beamId2/I");

    t->Branch("pthat",&jets_.pthat,"pthat/F");

    // Only matched gen jets
    t->Branch("refpt",jets_.refpt,"refpt[nref]/F");
    t->Branch("refeta",jets_.refeta,"refeta[nref]/F");
    t->Branch("refy",jets_.refy,"refy[nref]/F");
    t->Branch("refphi",jets_.refphi,"refphi[nref]/F");
    t->Branch("refdphijt",jets_.refdphijt,"refdphijt[nref]/F");
    t->Branch("refdrjt",jets_.refdrjt,"refdrjt[nref]/F");
    // matched parton
    t->Branch("refparton_pt",jets_.refparton_pt,"refparton_pt[nref]/F");
    t->Branch("refparton_flavor",jets_.refparton_flavor,"refparton_flavor[nref]/I");
    t->Branch("refparton_flavorForB",jets_.refparton_flavorForB,"refparton_flavorForB[nref]/I");

    t->Branch("genChargedSum", jets_.genChargedSum,"genChargedSum[nref]/F");
    t->Branch("genHardSum", jets_.genHardSum,"genHardSum[nref]/F");
    t->Branch("signalChargedSum", jets_.signalChargedSum,"signalChargedSum[nref]/F");
    t->Branch("signalHardSum", jets_.signalHardSum,"signalHardSum[nref]/F");

    if(doSubEvent_){
      t->Branch("subid",jets_.subid,"subid[nref]/I");
    }

    if(fillGenJets_){
       // For all gen jets, matched or unmatched
       t->Branch("ngen",&jets_.ngen,"ngen/I");
       t->Branch("genmatchindex",jets_.genmatchindex,"genmatchindex[ngen]/I");
       t->Branch("genpt",jets_.genpt,"genpt[ngen]/F");
       t->Branch("geneta",jets_.geneta,"geneta[ngen]/F");
       t->Branch("geny",jets_.geny,"geny[ngen]/F");
       t->Branch("genphi",jets_.genphi,"genphi[ngen]/F");
       t->Branch("gendphijt",jets_.gendphijt,"gendphijt[ngen]/F");
       t->Branch("gendrjt",jets_.gendrjt,"gendrjt[ngen]/F");

       if(doSubEvent_){
	  t->Branch("gensubid",jets_.gensubid,"gensubid[ngen]/I");
       }
    }

    if(saveBfragments_  ) {
      t->Branch("bMult",&jets_.bMult,"bMult/I");
      t->Branch("bJetIndex",jets_.bJetIndex,"bJetIndex[bMult]/I");
      t->Branch("bStatus",jets_.bStatus,"bStatus[bMult]/I");
      t->Branch("bVx",jets_.bVx,"bVx[bMult]/F");
      t->Branch("bVy",jets_.bVy,"bVy[bMult]/F");
      t->Branch("bVz",jets_.bVz,"bVz[bMult]/F");
      t->Branch("bPt",jets_.bPt,"bPt[bMult]/F");
      t->Branch("bEta",jets_.bEta,"bEta[bMult]/F");
      t->Branch("bPhi",jets_.bPhi,"bPhi[bMult]/F");
      t->Branch("bPdg",jets_.bPdg,"bPdg[bMult]/I");
      t->Branch("bChg",jets_.bChg,"bChg[bMult]/I");
    }

  }
  /*
  if(!isMC_){
    t->Branch("nL1TBit",&jets_.nL1TBit,"nL1TBit/I");
    t->Branch("l1TBit",jets_.l1TBit,"l1TBit[nL1TBit]/O");

    t->Branch("nL1ABit",&jets_.nL1ABit,"nL1ABit/I");
    t->Branch("l1ABit",jets_.l1ABit,"l1ABit[nL1ABit]/O");

    t->Branch("nHLTBit",&jets_.nHLTBit,"nHLTBit/I");
    t->Branch("hltBit",jets_.hltBit,"hltBit[nHLTBit]/O");

  }
  */
  TH1D::SetDefaultSumw2();


}


void
HiJPTJetAnalyzer::analyze(const Event& iEvent,
			     const EventSetup& iSetup) {

  int event = iEvent.id().event();
  int run = iEvent.id().run();
  int lumi = iEvent.id().luminosityBlock();

  jets_.run = run;
  jets_.evt = event;
  jets_.lumi = lumi;

  LogDebug("HiJPTJetAnalyzer")<<"START event: "<<event<<" in run "<<run<<endl;

 int bin = -1;
  double hf = 0.;
  double b = 999.;


  if(doHiJetID_ && !geo){
    edm::ESHandle<CaloGeometry> pGeo;
    iSetup.get<CaloGeometryRecord>().get(pGeo);
    geo = pGeo.product();
  }

   // loop the events

   jets_.bin = bin;
   jets_.hf = hf;

   reco::Vertex::Point vtx(0,0,0);
   if (useVtx_) {
     edm::Handle<vector<reco::Vertex> >vertex;
     iEvent.getByLabel(vtxTag_, vertex);

     if(vertex->size()>0) {
       jets_.vx=vertex->begin()->x();
       jets_.vy=vertex->begin()->y();
       jets_.vz=vertex->begin()->z();
       vtx = vertex->begin()->position();
     }
   }

   edm::Handle<pat::JetCollection> patjets;
   if(usePat_)iEvent.getByLabel(jetTag_, patjets);

   edm::Handle<pat::JetCollection> patmatchedjets;
   iEvent.getByLabel(matchTag_, patmatchedjets);

   edm::Handle<reco::JetView> matchedjets;
   iEvent.getByLabel(matchTag_, matchedjets);

   edm::Handle<reco::JetView> jets;
   iEvent.getByLabel(jetTag_, jets);

   edm::Handle<reco::PFCandidateCollection> pfCandidates;
   iEvent.getByLabel(pfCandidateLabel_,pfCandidates);

   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(trackTag_,tracks);

   edm::Handle<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > ebHits;
   edm::Handle<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > eeHits;

   edm::Handle<HFRecHitCollection> hfHits;
   edm::Handle<HBHERecHitCollection> hbheHits;

   iEvent.getByLabel(HcalRecHitHBHESrc_,hbheHits);
   iEvent.getByLabel(HcalRecHitHFSrc_,hfHits);
   iEvent.getByLabel(EBSrc_,ebHits);
   iEvent.getByLabel(EESrc_,eeHits);

   edm::Handle<reco::GenParticleCollection> genparts;
   iEvent.getByLabel(genParticleSrc_,genparts);


   // FILL JRA TREE
   jets_.b = b;
   jets_.nref = 0;


   if(doTrigger_){
     fillL1Bits(iEvent);
     fillHLTBits(iEvent);
   }

   for(unsigned int j = 0; j < jets->size(); ++j){
     const reco::Jet& jet = (*jets)[j];

     if(jet.pt() < jetPtMin_) continue;
     if (useJEC_ && usePat_){
       jets_.rawpt[jets_.nref]=(*patjets)[j].correctedJet("Uncorrected").pt();
	}

     if(doLifeTimeTagging_){

       if(jetTag_.label()=="icPu5patJets"){
	 jets_.discr_csvMva[jets_.nref]    = (*patjets)[j].bDiscriminator("icPu5CombinedSecondaryVertexMVABJetTags");
	 jets_.discr_csvSimple[jets_.nref] = (*patjets)[j].bDiscriminator("icPu5CombinedSecondaryVertexBJetTags");
	 jets_.discr_muByIp3[jets_.nref]   = (*patjets)[j].bDiscriminator("icPu5SoftMuonByIP3dBJetTags");
	 jets_.discr_muByPt[jets_.nref]    = (*patjets)[j].bDiscriminator("icPu5SoftMuonByPtBJetTags");
	 jets_.discr_prob[jets_.nref]      = (*patjets)[j].bDiscriminator("icPu5JetBProbabilityBJetTags");
	 jets_.discr_probb[jets_.nref]     = (*patjets)[j].bDiscriminator("icPu5JetProbabilityBJetTags");
	 jets_.discr_tcHighEff[jets_.nref] = (*patjets)[j].bDiscriminator("icPu5TrackCountingHighEffBJetTags");
	 jets_.discr_tcHighPur[jets_.nref] = (*patjets)[j].bDiscriminator("icPu5TrackCountingHighPurBJetTags");
       }
       else if(jetTag_.label()=="akPu3PFpatJets"){
	 jets_.discr_csvMva[jets_.nref]    = (*patjets)[j].bDiscriminator("akPu3PFCombinedSecondaryVertexMVABJetTags");
	 jets_.discr_csvSimple[jets_.nref] = (*patjets)[j].bDiscriminator("akPu3PFCombinedSecondaryVertexBJetTags");
	 jets_.discr_muByIp3[jets_.nref]   = (*patjets)[j].bDiscriminator("akPu3PFSoftMuonByIP3dBJetTags");
	 jets_.discr_muByPt[jets_.nref]    = (*patjets)[j].bDiscriminator("akPu3PFSoftMuonByPtBJetTags");
	 jets_.discr_prob[jets_.nref]      = (*patjets)[j].bDiscriminator("akPu3PFJetBProbabilityBJetTags");
	 jets_.discr_probb[jets_.nref]     = (*patjets)[j].bDiscriminator("akPu3PFJetProbabilityBJetTags");
	 jets_.discr_tcHighEff[jets_.nref] = (*patjets)[j].bDiscriminator("akPu3PFTrackCountingHighEffBJetTags");
	 jets_.discr_tcHighPur[jets_.nref] = (*patjets)[j].bDiscriminator("akPu3PFTrackCountingHighPurBJetTags");
       }
       else{
	 //	 cout<<" b-tagging variables not filled for this collection, turn of doLifeTimeTagging "<<endl;
       }

       const reco::SecondaryVertexTagInfo& tagInfoSV=*(*patjets)[j].tagInfoSecondaryVertex();

       jets_.nsvtx[jets_.nref]     = tagInfoSV.nVertices();

       if (tagInfoSV.nVertices()>0) {
	 jets_.svtxntrk[jets_.nref]  = tagInfoSV.nVertexTracks(0);
	 // this is the 3d flight distance, for 2-D use (0,true)
	 Measurement1D m1D = tagInfoSV.flightDistance(0);
	 jets_.svtxdl[jets_.nref]    = m1D.value();
	 jets_.svtxdls[jets_.nref]   = m1D.significance();

	 const reco::Vertex& svtx = tagInfoSV.secondaryVertex(0);
         //cout<<" SV:  vx: "<<svtx.x()<<" vy "<<svtx.y()<<" vz "<<svtx.z()<<endl;
         //cout<<" PV:  vx: "<<jet.vx()<<" vy "<<jet.vy()<<" vz "<<jet.vz()<<endl;
	 jets_.svtxm[jets_.nref]    = svtx.p4().mass();
	 jets_.svtxpt[jets_.nref]   = svtx.p4().pt();
	 //cout<<" chi2 "<<svtx.chi2()<<" ndof "<<svtx.ndof()<<endl;
       }

       const reco::TrackIPTagInfo& tagInfoIP=*(*patjets)[j].tagInfoTrackIP();

       jets_.nIPtrk[jets_.nref] = tagInfoIP.tracks().size();
       jets_.nselIPtrk[jets_.nref] = tagInfoIP.selectedTracks().size();

       if (doLifeTimeTaggingExtras_) {

	 TrackRefVector selTracks=tagInfoIP.selectedTracks();

	 GlobalPoint pv(tagInfoIP.primaryVertex()->position().x(),tagInfoIP.primaryVertex()->position().y(),tagInfoIP.primaryVertex()->position().z());

	 for(int it=0;it<jets_.nselIPtrk[jets_.nref] ;it++)
	   {
	     jets_.ipJetIndex[jets_.nIP + it]= jets_.nref;
	     reco::btag::TrackIPData data = tagInfoIP.impactParameterData()[it];
	     jets_.ipPt[jets_.nIP + it] = selTracks[it]->pt();
	     jets_.ipProb0[jets_.nIP + it] = tagInfoIP.probabilities(0)[it];
	     jets_.ipProb1[jets_.nIP + it] = tagInfoIP.probabilities(1)[it];
	     jets_.ip2d[jets_.nIP + it] = data.ip2d.value();
	     jets_.ip2dSig[jets_.nIP + it] = data.ip2d.significance();
	     jets_.ip3d[jets_.nIP + it] = data.ip3d.value();
	     jets_.ip3dSig[jets_.nIP + it] = data.ip3d.significance();
	     jets_.ipDist2Jet[jets_.nIP + it] = data.distanceToJetAxis.value();
	     jets_.ipDist2JetSig[jets_.nIP + it] = data.distanceToJetAxis.significance();
	     jets_.ipClosest2Jet[jets_.nIP + it] = (data.closestToJetAxis - pv).mag();	//decay length
	   }

	 jets_.nIP += jets_.nselIPtrk[jets_.nref];

       }

       const reco::PFCandidateCollection *pfCandidateColl = &(*pfCandidates);
       int pfMuonIndex = getPFJetMuon(jet, pfCandidateColl);
       if(pfMuonIndex >=0){
	 const reco::PFCandidate muon = pfCandidateColl->at(pfMuonIndex);
	 jets_.mupt[jets_.nref]    =  muon.pt();
	 jets_.mueta[jets_.nref]   =  muon.eta();
	 jets_.muphi[jets_.nref]   =  muon.phi();
	 jets_.mue[jets_.nref]     =  muon.energy();
	 jets_.mudr[jets_.nref]    =  reco::deltaR(jet,muon);
	 jets_.muptrel[jets_.nref] =  getPtRel(muon, jet);
	 jets_.muchg[jets_.nref]   =  muon.charge();
       }else{
	 jets_.mupt[jets_.nref]    =  0.0;
	 jets_.mueta[jets_.nref]   =  0.0;
	 jets_.muphi[jets_.nref]   =  0.0;
	 jets_.mue[jets_.nref]     =  0.0;
	 jets_.mudr[jets_.nref]    =  9.9;
	 jets_.muptrel[jets_.nref] =  0.0;
	 jets_.muchg[jets_.nref]   = 0;
       }

     }

     if(doHiJetID_){
     // Jet ID variables

     jets_.muMax[jets_.nref] = 0;
     jets_.muSum[jets_.nref] = 0;
     jets_.muN[jets_.nref] = 0;

     jets_.eMax[jets_.nref] = 0;
     jets_.eSum[jets_.nref] = 0;
     jets_.eN[jets_.nref] = 0;

     jets_.neutralMax[jets_.nref] = 0;
     jets_.neutralSum[jets_.nref] = 0;
     jets_.neutralN[jets_.nref] = 0;

     jets_.photonMax[jets_.nref] = 0;
     jets_.photonSum[jets_.nref] = 0;
     jets_.photonN[jets_.nref] = 0;
     jets_.photonHardSum[jets_.nref] = 0;
     jets_.photonHardN[jets_.nref] = 0;

     jets_.chargedMax[jets_.nref] = 0;
     jets_.chargedSum[jets_.nref] = 0;
     jets_.chargedN[jets_.nref] = 0;
     jets_.chargedHardSum[jets_.nref] = 0;
     jets_.chargedHardN[jets_.nref] = 0;

     jets_.trackMax[jets_.nref] = 0;
     jets_.trackSum[jets_.nref] = 0;
     jets_.trackN[jets_.nref] = 0;
     jets_.trackHardSum[jets_.nref] = 0;
     jets_.trackHardN[jets_.nref] = 0;

     jets_.hcalSum[jets_.nref] = 0;
     jets_.ecalSum[jets_.nref] = 0;

     jets_.genChargedSum[jets_.nref] = 0;
     jets_.genHardSum[jets_.nref] = 0;

     jets_.signalChargedSum[jets_.nref] = 0;
     jets_.signalHardSum[jets_.nref] = 0;

     jets_.subid[jets_.nref] = -1;

     for(unsigned int icand = 0; icand < tracks->size(); ++icand){
	const reco::Track& track = (*tracks)[icand];
	if(useQuality_ ){
	   bool goodtrack = track.quality(reco::TrackBase::qualityByName(trackQuality_));
	   if(!goodtrack) continue;
	}

	double dr = deltaR(jet,track);
	if(dr < rParam){
	   double ptcand = track.pt();
	   jets_.trackSum[jets_.nref] += ptcand;
	   jets_.trackN[jets_.nref] += 1;

	   if(ptcand > hardPtMin_){
	      jets_.trackHardSum[jets_.nref] += ptcand;
	      jets_.trackHardN[jets_.nref] += 1;

	   }
	   if(ptcand > jets_.trackMax[jets_.nref]) jets_.trackMax[jets_.nref] = ptcand;

	}
     }

     for(unsigned int icand = 0; icand < pfCandidates->size(); ++icand){
        const reco::PFCandidate& track = (*pfCandidates)[icand];
        double dr = deltaR(jet,track);
        if(dr < rParam){
           double ptcand = track.pt();
	   int pfid = track.particleId();

	   switch(pfid){

	   case 1:
              jets_.chargedSum[jets_.nref] += ptcand;
              jets_.chargedN[jets_.nref] += 1;
              if(ptcand > hardPtMin_){
                 jets_.chargedHardSum[jets_.nref] += ptcand;
                 jets_.chargedHardN[jets_.nref] += 1;
              }
              if(ptcand > jets_.chargedMax[jets_.nref]) jets_.chargedMax[jets_.nref] = ptcand;
	      break;

	   case 2:
              jets_.eSum[jets_.nref] += ptcand;
              jets_.eN[jets_.nref] += 1;
              if(ptcand > jets_.eMax[jets_.nref]) jets_.eMax[jets_.nref] = ptcand;
              break;

	   case 3:
              jets_.muSum[jets_.nref] += ptcand;
              jets_.muN[jets_.nref] += 1;
              if(ptcand > jets_.muMax[jets_.nref]) jets_.muMax[jets_.nref] = ptcand;
              break;

	   case 4:
              jets_.photonSum[jets_.nref] += ptcand;
              jets_.photonN[jets_.nref] += 1;
	      if(ptcand > hardPtMin_){
		 jets_.photonHardSum[jets_.nref] += ptcand;
		 jets_.photonHardN[jets_.nref] += 1;
	      }
              if(ptcand > jets_.photonMax[jets_.nref]) jets_.photonMax[jets_.nref] = ptcand;
              break;

	   case 5:
              jets_.neutralSum[jets_.nref] += ptcand;
              jets_.neutralN[jets_.nref] += 1;
              if(ptcand > jets_.neutralMax[jets_.nref]) jets_.neutralMax[jets_.nref] = ptcand;
              break;

	   default:
	     break;

	   }
	}
     }

     // Calorimeter fractions

     for(unsigned int i = 0; i < hbheHits->size(); ++i){
       const HBHERecHit & hit= (*hbheHits)[i];
       math::XYZPoint pos = getPosition(hit.id(),vtx);
       double dr = deltaR(jet.eta(),jet.phi(),pos.eta(),pos.phi());
       if(dr < rParam){
	 jets_.hcalSum[jets_.nref] += getEt(pos,hit.energy());
       }
     }

     for(unsigned int i = 0; i < hfHits->size(); ++i){
       const HFRecHit & hit= (*hfHits)[i];
       math::XYZPoint pos = getPosition(hit.id(),vtx);
       double dr = deltaR(jet.eta(),jet.phi(),pos.eta(),pos.phi());
       if(dr < rParam){
         jets_.hcalSum[jets_.nref] += getEt(pos,hit.energy());
       }
     }


     for(unsigned int i = 0; i < ebHits->size(); ++i){
       const EcalRecHit & hit= (*ebHits)[i];
       math::XYZPoint pos = getPosition(hit.id(),vtx);
       double dr = deltaR(jet.eta(),jet.phi(),pos.eta(),pos.phi());
       if(dr < rParam){
         jets_.ecalSum[jets_.nref] += getEt(pos,hit.energy());
       }
     }

     for(unsigned int i = 0; i < eeHits->size(); ++i){
       const EcalRecHit & hit= (*eeHits)[i];
       math::XYZPoint pos = getPosition(hit.id(),vtx);
       double dr = deltaR(jet.eta(),jet.phi(),pos.eta(),pos.phi());
       if(dr < rParam){
         jets_.ecalSum[jets_.nref] += getEt(pos,hit.energy());
       }
     }

     }
     // Jet ID for CaloJets



     if(doMatch_){

     // Alternative reconstruction matching (PF for calo, calo for PF)

     double drMin = 100;
     for(unsigned int imatch = 0 ; imatch < matchedjets->size(); ++imatch){
	const reco::Jet& mjet = (*matchedjets)[imatch];

	double dr = deltaR(jet,mjet);
	if(dr < drMin){
	   jets_.matchedPt[jets_.nref] = mjet.pt();
	   if(usePat_){
	     const pat::Jet& mpatjet = (*patmatchedjets)[imatch];
	     jets_.matchedRawPt[jets_.nref] = mpatjet.correctedJet("Uncorrected").pt();
             jets_.matchedPu[jets_.nref] = mpatjet.pileup();
	   }
           jets_.matchedR[jets_.nref] = dr;
	   drMin = dr;
	}
     }

     }
     //     if(etrk.quality(reco::TrackBase::qualityByName(qualityString_))) pev_.trkQual[pev_.nTrk]=1;


     if(doHiJetID_){

	/////////////////////////////////////////////////////////////////
	// Jet core pt^2 discriminant for fake jets
	// Edited by Yue Shi Lai <ylai@mit.edu>

	// Initial value is 0
	jets_.discr_fr01[jets_.nref] = 0;
	// Start with no directional adaption, i.e. the fake rejection
	// axis is the jet axis
	float pseudorapidity_adapt = jets_.jteta[jets_.nref];
	float azimuth_adapt = jets_.jtphi[jets_.nref];

	// Unadapted discriminant with adaption search
	for (size_t iteration = 0; iteration < 2; iteration++) {
		float pseudorapidity_adapt_new = pseudorapidity_adapt;
		float azimuth_adapt_new = azimuth_adapt;
		float max_weighted_perp = 0;
		float perp_square_sum = 0;

		for (size_t index_pf_candidate = 0;
			 index_pf_candidate < pfCandidates->size();
			 index_pf_candidate++) {
			const reco::PFCandidate &p =
				(*pfCandidates)[index_pf_candidate];

			switch (p.particleId()) {
			  //case 1:	// Charged hadron
			  //case 3:	// Muon
			case 4:	// Photon
				{
					const float dpseudorapidity =
						p.eta() - pseudorapidity_adapt;
					const float dazimuth =
						reco::deltaPhi(p.phi(), azimuth_adapt);
					// The Gaussian scale factor is 0.5 / (0.1 * 0.1)
					// = 50
					const float angular_weight =
						exp(-50.0F * (dpseudorapidity * dpseudorapidity +
									  dazimuth * dazimuth));
					const float weighted_perp =
						angular_weight * p.pt() * p.pt();
					const float weighted_perp_square =
						weighted_perp * p.pt();

					perp_square_sum += weighted_perp_square;
					if (weighted_perp >= max_weighted_perp) {
						pseudorapidity_adapt_new = p.eta();
						azimuth_adapt_new = p.phi();
						max_weighted_perp = weighted_perp;
					}
				}
			default:
			  break;
			}
		}
		// Update the fake rejection value
		jets_.discr_fr01[jets_.nref] = std::max(
			jets_.discr_fr01[jets_.nref], perp_square_sum);
		// Update the directional adaption
		pseudorapidity_adapt = pseudorapidity_adapt_new;
		azimuth_adapt = azimuth_adapt_new;
	}
     }

     jets_.jtpt[jets_.nref] = jet.pt();
     jets_.jteta[jets_.nref] = jet.eta();
     jets_.jtphi[jets_.nref] = jet.phi();
     jets_.jty[jets_.nref] = jet.eta();
     jets_.jtpu[jets_.nref] = jet.pileup();
     jets_.jtm[jets_.nref] = jet.mass();

     // Fill JPT specific using dynamic_cast


     jets_.ntrack = 0;

     const reco::JPTJet *jptjet = dynamic_cast<const reco::JPTJet*>((*patjets)[j].originalObject());

     jets_.pion_bgn[jets_.nref]=jets_.ntrack;
     const reco::TrackRefVector& pioninin = (*jptjet).getPionsInVertexInCalo();
     for(reco::TrackRefVector::const_iterator it = pioninin.begin(); it != pioninin.end(); it++) {
//	std::cout<<" pion Track in in "<<(*it)->p()<<" "<<(*it)->pt()<<" "<<(*it)->eta()<<" "<<(*it)->phi()<<std::endl;
        jets_.jpttrackpt[jets_.ntrack] = (*it)->pt();
        jets_.jpttrackphi[jets_.ntrack] = (*it)->phi();
        jets_.jpttracketa[jets_.ntrack] = (*it)->eta();
	jets_.ntrack++;
	}

     const reco::TrackRefVector& pioninout = (*jptjet).getPionsInVertexOutCalo();
     for(reco::TrackRefVector::const_iterator it = pioninout.begin(); it != pioninout.end(); it++) {
  //      std::cout<<" pion Track in out "<<(*it)->p()<<" "<<(*it)->pt()<<" "<<(*it)->eta()<<" "<<(*it)->phi()<<std::endl;
        jets_.jpttrackpt[jets_.ntrack] = (*it)->pt();
        jets_.jpttrackphi[jets_.ntrack] = (*it)->phi();
        jets_.jpttracketa[jets_.ntrack] = (*it)->eta();
	jets_.ntrack++;
        }

     jets_.pion_end[jets_.nref]=jets_.ntrack;

     jets_.muon_bgn[jets_.nref]=jets_.ntrack;

     const reco::TrackRefVector& muoninin = (*jptjet).getMuonsInVertexInCalo();
     for(reco::TrackRefVector::const_iterator it = muoninin.begin(); it != muoninin.end(); it++) {
//        std::cout<<" muon Track in in "<<(*it)->p()<<" "<<(*it)->pt()<<" "<<(*it)->eta()<<" "<<(*it)->phi()<<std::endl;
        jets_.jpttrackpt[jets_.ntrack] = (*it)->pt();
        jets_.jpttrackphi[jets_.ntrack] = (*it)->phi();
        jets_.jpttracketa[jets_.ntrack] = (*it)->eta();
        jets_.ntrack++;
        }

     const reco::TrackRefVector& muoninout = (*jptjet).getMuonsInVertexOutCalo();
     for(reco::TrackRefVector::const_iterator it = muoninout.begin(); it != muoninout.end(); it++) {
//        std::cout<<" muon Track in out "<<(*it)->p()<<" "<<(*it)->pt()<<" "<<(*it)->eta()<<" "<<(*it)->phi()<<std::endl;
        jets_.jpttrackpt[jets_.ntrack] = (*it)->pt();
        jets_.jpttrackphi[jets_.ntrack] = (*it)->phi();
        jets_.jpttracketa[jets_.ntrack] = (*it)->eta();
        jets_.ntrack++;
        }

     jets_.muon_end[jets_.nref]=jets_.ntrack;

     jets_.elecs_bgn[jets_.nref]=jets_.ntrack;

     const reco::TrackRefVector& elecsinin = (*jptjet).getElecsInVertexInCalo();
     for(reco::TrackRefVector::const_iterator it = elecsinin.begin(); it != elecsinin.end(); it++) {
//        std::cout<<" elec Track in in "<<(*it)->p()<<" "<<(*it)->pt()<<" "<<(*it)->eta()<<" "<<(*it)->phi()<<std::endl;
        jets_.jpttrackpt[jets_.ntrack] = (*it)->pt();
        jets_.jpttrackphi[jets_.ntrack] = (*it)->phi();
        jets_.jpttracketa[jets_.ntrack] = (*it)->eta();
        jets_.ntrack++;
        }

     const reco::TrackRefVector& elecsinout = (*jptjet).getElecsInVertexOutCalo();
     for(reco::TrackRefVector::const_iterator it = elecsinout.begin(); it != elecsinout.end(); it++) {
//        std::cout<<" elec Track in out "<<(*it)->p()<<" "<<(*it)->pt()<<" "<<(*it)->eta()<<" "<<(*it)->phi()<<std::endl;
        jets_.jpttrackpt[jets_.ntrack] = (*it)->pt();
        jets_.jpttrackphi[jets_.ntrack] = (*it)->phi();
        jets_.jpttracketa[jets_.ntrack] = (*it)->eta();
        jets_.ntrack++;
        }

     jets_.elecs_end[jets_.nref]=jets_.ntrack;



     if(usePat_){

	if(doStandardJetID_){
       jets_.fHPD[jets_.nref] = (*patjets)[j].jetID().fHPD;
       jets_.fRBX[jets_.nref] = (*patjets)[j].jetID().fRBX;
       jets_.n90[jets_.nref] = (*patjets)[j].n90();

       jets_.fSubDet1[jets_.nref] = (*patjets)[j].jetID().fSubDetector1;
       jets_.fSubDet2[jets_.nref] = (*patjets)[j].jetID().fSubDetector2;
       jets_.fSubDet3[jets_.nref] = (*patjets)[j].jetID().fSubDetector3;
       jets_.fSubDet4[jets_.nref] = (*patjets)[j].jetID().fSubDetector4;
       jets_.restrictedEMF[jets_.nref] = (*patjets)[j].jetID().restrictedEMF;
       jets_.nHCAL[jets_.nref] = (*patjets)[j].jetID().nHCALTowers;
       jets_.nECAL[jets_.nref] = (*patjets)[j].jetID().nECALTowers;
       jets_.apprHPD[jets_.nref] = (*patjets)[j].jetID().approximatefHPD;
       jets_.apprRBX[jets_.nref] = (*patjets)[j].jetID().approximatefRBX;

       //       jets_.n90[jets_.nref] = (*patjets)[j].jetID().hitsInN90;
       jets_.n2RPC[jets_.nref] = (*patjets)[j].jetID().numberOfHits2RPC;
       jets_.n3RPC[jets_.nref] = (*patjets)[j].jetID().numberOfHits3RPC;
       jets_.nRPC[jets_.nref] = (*patjets)[j].jetID().numberOfHitsRPC;

       jets_.fEB[jets_.nref] = (*patjets)[j].jetID().fEB;
       jets_.fEE[jets_.nref] = (*patjets)[j].jetID().fEE;
       jets_.fHB[jets_.nref] = (*patjets)[j].jetID().fHB;
       jets_.fHE[jets_.nref] = (*patjets)[j].jetID().fHE;
       jets_.fHO[jets_.nref] = (*patjets)[j].jetID().fHO;
       jets_.fLong[jets_.nref] = (*patjets)[j].jetID().fLong;
       jets_.fShort[jets_.nref] = (*patjets)[j].jetID().fShort;
       jets_.fLS[jets_.nref] = (*patjets)[j].jetID().fLS;
       jets_.fHFOOT[jets_.nref] = (*patjets)[j].jetID().fHFOOT;
	}

          }

     if(isMC_){

       for(UInt_t i = 0; i < genparts->size(); ++i){
         const reco::GenParticle& p = (*genparts)[i];
         if (p.status()!=1) continue;
         if (p.charge()==0) continue;
         double dr = deltaR(jet,p);
         if(dr < rParam){
           double ppt = p.pt();
           jets_.genChargedSum[jets_.nref] += ppt;
           if(ppt > hardPtMin_) jets_.genHardSum[jets_.nref] += ppt;
           if(p.collisionId() == 0){
             jets_.signalChargedSum[jets_.nref] += ppt;
             if(ppt > hardPtMin_) jets_.signalHardSum[jets_.nref] += ppt;
           }

         }
       }

     }

     if(isMC_ && usePat_){


       const reco::GenJet * genjet = (*patjets)[j].genJet();

       if(genjet){
	 jets_.refpt[jets_.nref] = genjet->pt();
	 jets_.refeta[jets_.nref] = genjet->eta();
	 jets_.refphi[jets_.nref] = genjet->phi();
	 jets_.refy[jets_.nref] = genjet->eta();
	 jets_.refdphijt[jets_.nref] = reco::deltaPhi(jet.phi(), genjet->phi());
	 jets_.refdrjt[jets_.nref] = reco::deltaR(jet.eta(),jet.phi(),genjet->eta(),genjet->phi());

	 if(doSubEvent_){
	   const GenParticle* gencon = genjet->getGenConstituent(0);
	   jets_.subid[jets_.nref] = gencon->collisionId();
         }

       }else{
	 jets_.refpt[jets_.nref] = -999.;
	 jets_.refeta[jets_.nref] = -999.;
	 jets_.refphi[jets_.nref] = -999.;
	 jets_.refy[jets_.nref] = -999.;
	 jets_.refdphijt[jets_.nref] = -999.;
	 jets_.refdrjt[jets_.nref] = -999.;
       }

       jets_.refparton_flavorForB[jets_.nref] = (*patjets)[j].partonFlavour();

       // matched partons
       const reco::GenParticle & parton = *(*patjets)[j].genParton();

       if((*patjets)[j].genParton()){
	 jets_.refparton_pt[jets_.nref] = parton.pt();
	 jets_.refparton_flavor[jets_.nref] = parton.pdgId();

	 if(saveBfragments_ && abs(jets_.refparton_flavorForB[jets_.nref])==5){

	   usedStringPts.clear();

	   // uncomment this if you want to know the ugly truth about parton matching -matt
	   //if(jet.pt() > 50 &&abs(parton.pdgId())!=5 && parton.pdgId()!=21)
	   // cout<<" Identified as a b, but doesn't match b or gluon, id = "<<parton.pdgId()<<endl;

	   jets_.bJetIndex[jets_.bMult] = jets_.nref;
	   jets_.bStatus[jets_.bMult] = parton.status();
	   jets_.bVx[jets_.bMult] = parton.vx();
	   jets_.bVy[jets_.bMult] = parton.vy();
	   jets_.bVz[jets_.bMult] = parton.vz();
	   jets_.bPt[jets_.bMult] = parton.pt();
	   jets_.bEta[jets_.bMult] = parton.eta();
	   jets_.bPhi[jets_.bMult] = parton.phi();
	   jets_.bPdg[jets_.bMult] = parton.pdgId();
	   jets_.bChg[jets_.bMult] = parton.charge();
	   jets_.bMult++;
	   saveDaughters(parton);
	 }


       } else {
	 jets_.refparton_pt[jets_.nref] = -999;
	 jets_.refparton_flavor[jets_.nref] = -999;
       }


     }

     jets_.nref++;


   }


   if(isMC_){

     edm::Handle<HepMCProduct> hepMCProduct;
     iEvent.getByLabel(eventInfoTag_,hepMCProduct);
     const HepMC::GenEvent* MCEvt = hepMCProduct->GetEvent();

	std::pair<HepMC::GenParticle*,HepMC::GenParticle*> beamParticles = MCEvt->beam_particles();
	if(beamParticles.first != 0)jets_.beamId1 = beamParticles.first->pdg_id();
	if(beamParticles.second != 0)jets_.beamId2 = beamParticles.second->pdg_id();

     edm::Handle<GenEventInfoProduct> hEventInfo;
     iEvent.getByLabel(eventInfoTag_,hEventInfo);
     //jets_.pthat = hEventInfo->binningValues()[0];

     // binning values and qscale appear to be equivalent, but binning values not always present
     jets_.pthat = hEventInfo->qScale();

     edm::Handle<vector<reco::GenJet> >genjets;
     iEvent.getByLabel(genjetTag_, genjets);

     jets_.ngen = 0;
     for(unsigned int igen = 0 ; igen < genjets->size(); ++igen){
       const reco::GenJet & genjet = (*genjets)[igen];

       float genjet_pt = genjet.pt();

       // threshold to reduce size of output in minbias PbPb
       if(genjet_pt>genPtMin_){


	 jets_.genpt[jets_.ngen] = genjet_pt;
	 jets_.geneta[jets_.ngen] = genjet.eta();
	 jets_.genphi[jets_.ngen] = genjet.phi();
	 jets_.geny[jets_.ngen] = genjet.eta();

	 if(doSubEvent_){
	    const GenParticle* gencon = genjet.getGenConstituent(0);
	    jets_.gensubid[jets_.ngen] = gencon->collisionId();
	 }

	 // find matching patJet if there is one

	 jets_.gendrjt[jets_.ngen] = -1.0;
	 jets_.genmatchindex[jets_.ngen] = -1;

	 for(int ijet = 0 ; ijet < jets_.nref; ++ijet){
	     // poor man's matching, someone fix please
	   if(fabs(genjet.pt()-jets_.refpt[ijet])<0.00001 &&
	      fabs(genjet.eta()-jets_.refeta[ijet])<0.00001){

	     jets_.genmatchindex[jets_.ngen] = (int)ijet;
	       jets_.gendphijt[jets_.ngen] = reco::deltaPhi(jets_.refphi[ijet],genjet.phi());
	       jets_.gendrjt[jets_.ngen] = sqrt(pow(jets_.gendphijt[jets_.ngen],2)+pow(fabs(genjet.eta()-jets_.refeta[ijet]),2));

	       break;
	     }
	   }
	 }

	 jets_.ngen++;
     }

   }





   t->Fill();
   memset(&jets_,0,sizeof jets_);

}




//--------------------------------------------------------------------------------------------------
void HiJPTJetAnalyzer::fillL1Bits(const edm::Event &iEvent)
{
  edm::Handle< L1GlobalTriggerReadoutRecord >  L1GlobalTrigger;

  iEvent.getByLabel(L1gtReadout_, L1GlobalTrigger);
  const TechnicalTriggerWord&  technicalTriggerWordBeforeMask = L1GlobalTrigger->technicalTriggerWord();

  for (int i=0; i<64;i++)
    {
      jets_.l1TBit[i] = technicalTriggerWordBeforeMask.at(i);
    }
  jets_.nL1TBit = 64;

  int ntrigs = L1GlobalTrigger->decisionWord().size();
  jets_.nL1ABit = ntrigs;

  for (int i=0; i != ntrigs; i++) {
    bool accept = L1GlobalTrigger->decisionWord()[i];
    //jets_.l1ABit[i] = (accept == true)? 1:0;
    if(accept== true){
      jets_.l1ABit[i] = 1;
    }
    else{
      jets_.l1ABit[i] = 0;
    }

  }
}

//--------------------------------------------------------------------------------------------------
void HiJPTJetAnalyzer::fillHLTBits(const edm::Event &iEvent)
{
  // Fill HLT trigger bits.
  Handle<TriggerResults> triggerResultsHLT;
  getProduct(hltResName_, triggerResultsHLT, iEvent);

  const TriggerResults *hltResults = triggerResultsHLT.product();
  const TriggerNames & triggerNames = iEvent.triggerNames(*hltResults);

  jets_.nHLTBit = hltTrgNames_.size();

  for(size_t i=0;i<hltTrgNames_.size();i++){

    for(size_t j=0;j<triggerNames.size();++j) {

      if(triggerNames.triggerName(j) == hltTrgNames_[i]){

	//cout <<"hltTrgNames_(i) "<<hltTrgNames_[i]<<endl;
	//cout <<"triggerName(j) "<<triggerNames.triggerName(j)<<endl;
	//cout<<" result "<<triggerResultsHLT->accept(j)<<endl;
	jets_.hltBit[i] = triggerResultsHLT->accept(j);
      }

    }
  }
}

//--------------------------------------------------------------------------------------------------
template <typename TYPE>
inline void HiJPTJetAnalyzer::getProduct(const std::string name, edm::Handle<TYPE> &prod,
					 const edm::Event &event) const
{
  // Try to access data collection from EDM file. We check if we really get just one
  // product with the given name. If not we throw an exception.

  event.getByLabel(edm::InputTag(name),prod);
  if (!prod.isValid())
    throw edm::Exception(edm::errors::Configuration, "HiJPTJetAnalyzer::GetProduct()\n")
      << "Collection with label '" << name << "' is not valid" <<  std::endl;
}

//--------------------------------------------------------------------------------------------------
template <typename TYPE>
inline bool HiJPTJetAnalyzer::getProductSafe(const std::string name, edm::Handle<TYPE> &prod,
					     const edm::Event &event) const
{
  // Try to safely access data collection from EDM file. We check if we really get just one
  // product with the given name. If not, we return false.

  if (name.size()==0)
    return false;

  try {
    event.getByLabel(edm::InputTag(name),prod);
    if (!prod.isValid())
      return false;
  } catch (...) {
    return false;
  }
  return true;
}


int
HiJPTJetAnalyzer::getPFJetMuon(const pat::Jet& pfJet, const reco::PFCandidateCollection *pfCandidateColl)
{

  int pfMuonIndex = -1;
  float ptMax = 0.;


  for(unsigned icand=0;icand<pfCandidateColl->size(); icand++) {
    const reco::PFCandidate pfCandidate = pfCandidateColl->at(icand);

    int id = pfCandidate.particleId();
    if(abs(id) != 3) continue;

    if(reco::deltaR(pfJet,pfCandidate)>0.5) continue;

    double pt =  pfCandidate.pt();
    if(pt>ptMax){
      ptMax = pt;
      pfMuonIndex = (int) icand;
    }
  }

  return pfMuonIndex;

}


double
HiJPTJetAnalyzer::getPtRel(const reco::PFCandidate lep, const pat::Jet& jet )
{

  float lj_x = jet.p4().px();
  float lj_y = jet.p4().py();
  float lj_z = jet.p4().pz();

  // absolute values squared
  float lj2  = lj_x*lj_x+lj_y*lj_y+lj_z*lj_z;
  float lep2 = lep.px()*lep.px()+lep.py()*lep.py()+lep.pz()*lep.pz();

  // projection vec(mu) to lepjet axis
  float lepXlj = lep.px()*lj_x+lep.py()*lj_y+lep.pz()*lj_z;

  // absolute value squared and normalized
  float pLrel2 = lepXlj*lepXlj/lj2;

  // lep2 = pTrel2 + pLrel2
  float pTrel2 = lep2-pLrel2;

  return (pTrel2 > 0) ? std::sqrt(pTrel2) : 0.0;
}

// Recursive function, but this version gets called only the first time
void
HiJPTJetAnalyzer::saveDaughters(const reco::GenParticle &gen){

  for(unsigned i=0;i<gen.numberOfDaughters();i++){
    const reco::Candidate & daughter = *gen.daughter(i);
    double daughterPt = daughter.pt();
    if(daughterPt<1.) continue;
    double daughterEta = daughter.eta();
    if(fabs(daughterEta)>3.) continue;
    int daughterPdgId = daughter.pdgId();
    int daughterStatus = daughter.status();
    // Special case when b->b+string, both b and string contain all daughters, so only take the string
    if(gen.pdgId()==daughterPdgId && gen.status()==3 && daughterStatus==2) continue;

    // cheesy way of finding strings which were already used
    if(daughter.pdgId()==92){
      for(unsigned ist=0;ist<usedStringPts.size();ist++){
	if(fabs(daughter.pt() - usedStringPts[ist]) < 0.0001) return;
      }
      usedStringPts.push_back(daughter.pt());
    }
    jets_.bJetIndex[jets_.bMult] = jets_.nref;
    jets_.bStatus[jets_.bMult] = daughterStatus;
    jets_.bVx[jets_.bMult] = daughter.vx();
    jets_.bVy[jets_.bMult] = daughter.vy();
    jets_.bVz[jets_.bMult] = daughter.vz();
    jets_.bPt[jets_.bMult] = daughterPt;
    jets_.bEta[jets_.bMult] = daughterEta;
    jets_.bPhi[jets_.bMult] = daughter.phi();
    jets_.bPdg[jets_.bMult] = daughterPdgId;
    jets_.bChg[jets_.bMult] = daughter.charge();
    jets_.bMult++;
    saveDaughters(daughter);
  }
}

// This version called for all subsequent calls
void
HiJPTJetAnalyzer::saveDaughters(const reco::Candidate &gen){

  for(unsigned i=0;i<gen.numberOfDaughters();i++){
    const reco::Candidate & daughter = *gen.daughter(i);
    double daughterPt = daughter.pt();
    if(daughterPt<1.) continue;
    double daughterEta = daughter.eta();
    if(fabs(daughterEta)>3.) continue;
    int daughterPdgId = daughter.pdgId();
    int daughterStatus = daughter.status();
    // Special case when b->b+string, both b and string contain all daughters, so only take the string
    if(gen.pdgId()==daughterPdgId && gen.status()==3 && daughterStatus==2) continue;

    // cheesy way of finding strings which were already used
    if(daughter.pdgId()==92){
      for(unsigned ist=0;ist<usedStringPts.size();ist++){
	if(fabs(daughter.pt() - usedStringPts[ist]) < 0.0001) return;
      }
      usedStringPts.push_back(daughter.pt());
    }

    jets_.bJetIndex[jets_.bMult] = jets_.nref;
    jets_.bStatus[jets_.bMult] = daughterStatus;
    jets_.bVx[jets_.bMult] = daughter.vx();
    jets_.bVy[jets_.bMult] = daughter.vy();
    jets_.bVz[jets_.bMult] = daughter.vz();
    jets_.bPt[jets_.bMult] = daughterPt;
    jets_.bEta[jets_.bMult] = daughterEta;
    jets_.bPhi[jets_.bMult] = daughter.phi();
    jets_.bPdg[jets_.bMult] = daughterPdgId;
    jets_.bChg[jets_.bMult] = daughter.charge();
    jets_.bMult++;
    saveDaughters(daughter);
  }
}

double HiJPTJetAnalyzer::getEt(math::XYZPoint pos, double energy){
  double et = energy*sin(pos.theta());
  return et;
}

math::XYZPoint HiJPTJetAnalyzer::getPosition(const DetId &id, reco::Vertex::Point vtx){
  const GlobalPoint& pos=geo->getPosition(id);
  math::XYZPoint posV(pos.x() - vtx.x(),pos.y() - vtx.y(),pos.z() - vtx.z());
  return posV;
}



DEFINE_FWK_MODULE(HiJPTJetAnalyzer);
