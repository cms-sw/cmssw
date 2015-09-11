/*
  Based on the jet response analyzer
  Modified by Matt Nguyen, November 2010

*/

#include "HeavyIonsAnalysis/JetAnalysis/interface/HiPFJetAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
//#include <Math/DistFunc.h>
#include "TMath.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"
// #include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "L1Trigger/GlobalTrigger/plugins/L1GlobalTrigger.h"


#include "DataFormats/BeamSpot/interface/BeamSpot.h"


using namespace std;
using namespace edm;
using namespace reco;


HiPFJetAnalyzer::HiPFJetAnalyzer(const edm::ParameterSet& iConfig) {


  jetTag1_ = iConfig.getParameter<InputTag>("jetTag1");
  jetTag2_ = iConfig.getParameter<InputTag>("jetTag2");
  jetTag3_ = iConfig.getParameter<InputTag>("jetTag3");
  jetTag4_ = iConfig.getParameter<InputTag>("jetTag4");

  recoJetTag1_ = iConfig.getParameter<InputTag>("recoJetTag1");
  recoJetTag2_ = iConfig.getParameter<InputTag>("recoJetTag2");
  recoJetTag3_ = iConfig.getParameter<InputTag>("recoJetTag3");
  recoJetTag4_ = iConfig.getParameter<InputTag>("recoJetTag4");

  genJetTag1_ = iConfig.getParameter<InputTag>("genJetTag1");
  genJetTag2_ = iConfig.getParameter<InputTag>("genJetTag2");
  genJetTag3_ = iConfig.getParameter<InputTag>("genJetTag3");
  genJetTag4_ = iConfig.getParameter<InputTag>("genJetTag4");

  pfCandidatesTag_ = iConfig.getParameter<InputTag>("pfCandidatesTag");
  trackTag_ = iConfig.getParameter<edm::InputTag>("trackTag");
  vertexTag_ = iConfig.getParameter<edm::InputTag>("vertexTag");

  verbose_ = iConfig.getUntrackedParameter<bool>("verbose",false);

  isMC_ = iConfig.getUntrackedParameter<bool>("isMC",false);
  genParticleThresh_ = iConfig.getParameter<double>("genParticleThresh");

  genParticleTag_ = iConfig.getParameter<InputTag>("genParticleTag");
  eventInfoTag_ = iConfig.getParameter<InputTag>("eventInfoTag");

  hasSimInfo_ = iConfig.getUntrackedParameter<bool>("hasSimInfo");
  simTracksTag_ = iConfig.getParameter<InputTag>("SimTracks");
  associatorMap_=iConfig.getParameter<edm::InputTag>("associatorMap");

  if(!isMC_){
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


  cout<<" tracks : "<<trackTag_<<endl;
  cout<<" jet collection 1: "<<jetTag1_<<endl;
  cout<<" jet collection 2: "<<jetTag2_<<endl;
  cout<<" jet collection 3: "<<jetTag3_<<endl;
  cout<<" jet collection 4: "<<jetTag4_<<endl;

  jets_.nj1 = 0;
  jets_.nj2 = 0;
  jets_.nj3 = 0;
  jets_.nj4 = 0;
  jets_.nunmatch_j1 = 0;
  jets_.nunmatch_j2 = 0;
  jets_.nunmatch_j3 = 0;
  jets_.nunmatch_j4 = 0;
  jets_.nPFcand = 0;
  jets_.ntrack = 0;
  jets_.ngenp = 0;



}



HiPFJetAnalyzer::~HiPFJetAnalyzer() { }



void
HiPFJetAnalyzer::beginRun(const edm::Run& run,
			  const edm::EventSetup & es) {}

void
HiPFJetAnalyzer::beginJob() {

  t = fs2->make<TTree>("t","Jet Analysis Tree");


  //  TTree* t= new TTree("t","Jet Response Analyzer");
  t->Branch("run",&jets_.run,"run/I");
  t->Branch("evt",&jets_.evt,"evt/I");
  t->Branch("lumi",&jets_.lumi,"lumi/I");
  t->Branch("b",&jets_.b,"b/F");

  t->Branch("vx",&jets_.vx,"vx/F");
  t->Branch("vy",&jets_.vy,"vy/F");
  t->Branch("vz",&jets_.vz,"vz/F");

  t->Branch("hasVtx",&jets_.hasVtx,"hasVtx/O");

  t->Branch("vxErr",&jets_.vxErr,"vxErr/F");
  t->Branch("vyErr",&jets_.vyErr,"vyErr/F");
  t->Branch("vzErr",&jets_.vzErr,"vzErr/F");

  t->Branch("bsx",&jets_.bsx,"bsx/F");
  t->Branch("bsy",&jets_.bsy,"bsy/F");
  t->Branch("bsz",&jets_.bsz,"bsz/F");
  t->Branch("bswx",&jets_.bswx,"bswx/F");
  t->Branch("bswy",&jets_.bswy,"bswy/F");

  t->Branch("hf",&jets_.hf,"hf/F");
  t->Branch("bin",&jets_.bin,"bin/I");

  // ICPU5
  t->Branch("nj1",&jets_.nj1,"nj1/I");
  t->Branch("nj2",&jets_.nj2,"nj2/I");
  t->Branch("nj3",&jets_.nj3,"nj3/I");
  t->Branch("nj4",&jets_.nj4,"nj4/I");

  t->Branch("rawpt_j1",jets_.rawpt_j1,"rawpt_j1[nj1]/F");
  t->Branch("jtpt_j1",jets_.jtpt_j1,"jtpt_j1[nj1]/F");
  t->Branch("jteta_j1",jets_.jteta_j1,"jteta_j1[nj1]/F");
  t->Branch("jty_j1",jets_.jty_j1,"jty_j1[nj1]/F");
  t->Branch("jtphi_j1",jets_.jtphi_j1,"jtphi_j1[nj1]/F");
  t->Branch("preL1et_j1",jets_.preL1et_j1,"preL1et_j1[nj1]/F");
  t->Branch("L2_j1",jets_.L2_j1,"L2_j1[nj1]/F");
  t->Branch("L3_j1",jets_.L3_j1,"L3_j1[nj1]/F");
  t->Branch("area_j1",jets_.area_j1,"area_j1[nj1]/F");

  if(isMC_){
    // Matched gen jets
    t->Branch("refpt_j1",jets_.refpt_j1,"refpt_j1[nj1]/F");
    t->Branch("refeta_j1",jets_.refeta_j1,"refeta_j1[nj1]/F");
    t->Branch("refy_j1",jets_.refy_j1,"refy_j1[nj1]/F");
    t->Branch("refphi_j1",jets_.refphi_j1,"refphi_j1[nj1]/F");
    t->Branch("refdrjt_j1",jets_.refdrjt_j1,"refdrjt_j1[nj1]/F");
    t->Branch("refpartonpt_j1",jets_.refpartonpt_j1,"refpartonpt_j1[nj1]/F");
    t->Branch("refpartonflavor_j1",jets_.refpartonflavor_j1,"refpartonflavor_j1[nj1]/F");

    // Unmatched gen jets
    t->Branch("nunmatch_j1",&jets_.nunmatch_j1,"nunmatch_j1/I");
    t->Branch("unmatchpt_j1",jets_.unmatchpt_j1,"unmatchpt_j1[nunmatch_j1]/F");
    t->Branch("unmatcheta_j1",jets_.unmatcheta_j1,"unmatcheta_j1[nunmatch_j1]/F");
    t->Branch("unmatchy_j1",jets_.unmatchy_j1,"unmatchy_j1[nunmatch_j1]/F");
    t->Branch("unmatchphi_j1",jets_.unmatchphi_j1,"unmatchphi_j1[nunmatch_j1]/F");
  }


  // J2

  t->Branch("rawpt_j2",jets_.rawpt_j2,"rawpt_j2[nj2]/F");
  t->Branch("jtpt_j2",jets_.jtpt_j2,"jtpt_j2[nj2]/F");
  t->Branch("jteta_j2",jets_.jteta_j2,"jteta_j2[nj2]/F");
  t->Branch("jty_j2",jets_.jty_j2,"jty_j2[nj2]/F");
  t->Branch("jtphi_j2",jets_.jtphi_j2,"jtphi_j2[nj2]/F");
  t->Branch("preL1et_j2",jets_.preL1et_j2,"preL1et_j2[nj2]/F");
  t->Branch("L2_j2",jets_.L2_j2,"L2_j2[nj2]/F");
  t->Branch("L3_j2",jets_.L3_j2,"L3_j2[nj2]/F");
  t->Branch("area_j2",jets_.area_j2,"area_j2[nj2]/F");

  if(isMC_){
    t->Branch("refpt_j2",jets_.refpt_j2,"refpt_j2[nj2]/F");
    t->Branch("refeta_j2",jets_.refeta_j2,"refeta_j2[nj2]/F");
    t->Branch("refy_j2",jets_.refy_j2,"refy_j2[nj2]/F");
    t->Branch("refphi_j2",jets_.refphi_j2,"refphi_j2[nj2]/F");
    t->Branch("refdrjt_j2",jets_.refdrjt_j2,"refdrjt_j2[nj2]/F");
    t->Branch("refpartonpt_j2",jets_.refpartonpt_j2,"refpartonpt_j2[nj2]/F");
    t->Branch("refpartonflavor_j2",jets_.refpartonflavor_j2,"refpartonflavor_j2[nj2]/F");

    // Unmatched gen jets
    t->Branch("nunmatch_j2",&jets_.nunmatch_j2,"nunmatch_j2/I");
    t->Branch("unmatchpt_j2",jets_.unmatchpt_j2,"unmatchpt_j2[nunmatch_j2]/F");
    t->Branch("unmatcheta_j2",jets_.unmatcheta_j2,"unmatcheta_j2[nunmatch_j2]/F");
    t->Branch("unmatchy_j2",jets_.unmatchy_j2,"unmatchy_j2[nunmatch_j2]/F");
    t->Branch("unmatchphi_j2",jets_.unmatchphi_j2,"unmatchphi_j2[nunmatch_j2]/F");

  }




  // J3

  t->Branch("rawpt_j3",jets_.rawpt_j3,"rawpt_j3[nj3]/F");
  t->Branch("jtpt_j3",jets_.jtpt_j3,"jtpt_j3[nj3]/F");
  t->Branch("jteta_j3",jets_.jteta_j3,"jteta_j3[nj3]/F");
  t->Branch("jty_j3",jets_.jty_j3,"jty_j3[nj3]/F");
  t->Branch("jtphi_j3",jets_.jtphi_j3,"jtphi_j3[nj3]/F");
  t->Branch("preL1et_j3",jets_.preL1et_j3,"preL1et_j3[nj3]/F");
  t->Branch("L2_j3",jets_.L2_j3,"L2_j3[nj3]/F");
  t->Branch("L3_j3",jets_.L3_j3,"L3_j3[nj3]/F");
  t->Branch("area_j3",jets_.area_j3,"area_j3[nj3]/F");

  if(isMC_){
    t->Branch("refpt_j3",jets_.refpt_j3,"refpt_j3[nj3]/F");
    t->Branch("refeta_j3",jets_.refeta_j3,"refeta_j3[nj3]/F");
    t->Branch("refy_j3",jets_.refy_j3,"refy_j3[nj3]/F");
    t->Branch("refphi_j3",jets_.refphi_j3,"refphi_j3[nj3]/F");
    t->Branch("refdrjt_j3",jets_.refdrjt_j3,"refdrjt_j3[nj3]/F");
    t->Branch("refpartonpt_j3",jets_.refpartonpt_j3,"refpartonpt_j3[nj3]/F");
    t->Branch("refpartonflavor_j3",jets_.refpartonflavor_j3,"refpartonflavor_j3[nj3]/F");

    // Unmatched gen jets
    t->Branch("nunmatch_j3",&jets_.nunmatch_j3,"nunmatch_j3/I");
    t->Branch("unmatchpt_j3",jets_.unmatchpt_j3,"unmatchpt_j3[nunmatch_j3]/F");
    t->Branch("unmatcheta_j3",jets_.unmatcheta_j3,"unmatcheta_j3[nunmatch_j3]/F");
    t->Branch("unmatchy_j3",jets_.unmatchy_j3,"unmatchy_j3[nunmatch_j3]/F");
    t->Branch("unmatchphi_j3",jets_.unmatchphi_j3,"unmatchphi_j3[nunmatch_j3]/F");
  }

  // J4

  t->Branch("rawpt_j4",jets_.rawpt_j4,"rawpt_j4[nj4]/F");
  t->Branch("jtpt_j4",jets_.jtpt_j4,"jtpt_j4[nj4]/F");
  t->Branch("jteta_j4",jets_.jteta_j4,"jteta_j4[nj4]/F");
  t->Branch("jty_j4",jets_.jty_j4,"jty_j4[nj4]/F");
  t->Branch("jtphi_j4",jets_.jtphi_j4,"jtphi_j4[nj4]/F");
  t->Branch("preL1et_j4",jets_.preL1et_j4,"preL1et_j4[nj4]/F");
  t->Branch("L2_j4",jets_.L2_j4,"L2_j4[nj4]/F");
  t->Branch("L3_j4",jets_.L3_j4,"L3_j4[nj4]/F");
  t->Branch("area_j4",jets_.area_j4,"area_j4[nj4]/F");

  if(isMC_){
    t->Branch("refpt_j4",jets_.refpt_j4,"refpt_j4[nj4]/F");
    t->Branch("refeta_j4",jets_.refeta_j4,"refeta_j4[nj4]/F");
    t->Branch("refy_j4",jets_.refy_j4,"refy_j4[nj4]/F");
    t->Branch("refphi_j4",jets_.refphi_j4,"refphi_j4[nj4]/F");
    t->Branch("refdrjt_j4",jets_.refdrjt_j4,"refdrjt_j4[nj4]/F");
    t->Branch("refpartonpt_j4",jets_.refpartonpt_j4,"refpartonpt_j4[nj4]/F");
    t->Branch("refpartonflavor_j4",jets_.refpartonflavor_j4,"refpartonflavor_j4[nj4]/F");

    // Unmatched gen jets
    t->Branch("nunmatch_j4",&jets_.nunmatch_j4,"nunmatch_j4/I");
    t->Branch("unmatchpt_j4",jets_.unmatchpt_j4,"unmatchpt_j4[nunmatch_j4]/F");
    t->Branch("unmatcheta_j4",jets_.unmatcheta_j4,"unmatcheta_j4[nunmatch_j4]/F");
    t->Branch("unmatchy_j4",jets_.unmatchy_j4,"unmatchy_j4[nunmatch_j4]/F");
    t->Branch("unmatchphi_j4",jets_.unmatchphi_j4,"unmatchphi_j4[nunmatch_j4]/F");
  }

  t->Branch("nPFcand",&jets_.nPFcand,"nPFcand/I");
  t->Branch("candId",jets_.candId,"candId[nPFcand]/I");
  t->Branch("candpt",jets_.candpt,"candpt[nPFcand]/F");
  t->Branch("candeta",jets_.candeta,"candeta[nPFcand]/F");
  //t->Branch("candy",jets_.candy,"candy[nPFcand]/F");
  t->Branch("candphi",jets_.candphi,"candphi[nPFcand]/F");



  t->Branch("ntrack",&jets_.ntrack,"ntrack/I");
  t->Branch("tracknhits",jets_.tracknhits,"tracknhits[ntrack]/I");
  t->Branch("trackpt",jets_.trackpt,"trackpt[ntrack]/F");
  t->Branch("tracketa",jets_.tracketa,"tracketa[ntrack]/F");
  t->Branch("trackphi",jets_.trackphi,"trackphi[ntrack]/F");
  t->Branch("tracksumecal",jets_.tracksumecal,"tracksumecal[ntrack]/F");
  t->Branch("tracksumhcal",jets_.tracksumhcal,"tracksumhcal[ntrack]/F");
  t->Branch("trackqual",jets_.trackqual,"trackqual[ntrack]/I");
  t->Branch("chi2",jets_.trackchi2,"chi2[ntrack]/F");
  t->Branch("chi2hit1D",jets_.trackchi2hit1D,"chi2hit1D[ntrack]/F");

  t->Branch("ptErr",jets_.trackptErr,"ptErr[ntrack]/F");

  t->Branch("d0Err",jets_.trackd0Err,"d0Err[ntrack]/F");
  t->Branch("dzErr",jets_.trackdzErr,"dzErr[ntrack]/F");

  t->Branch("d0ErrTrk",jets_.trackd0ErrTrk,"d0ErrTrk[ntrack]/F");
  t->Branch("dzErrTrk",jets_.trackdzErrTrk,"dzErrTrk[ntrack]/F");

  t->Branch("d0",jets_.trackd0,"d0[ntrack]/F");
  t->Branch("dz",jets_.trackdz,"dz[ntrack]/F");

  //  t->Branch("d0ErrBS",jets_.trackd0ErrBS,"d0ErrBS[ntrack]/F");
  //  t->Branch("dzErrBS",jets_.trackdzErrBS,"dzErrBS[ntrack]/F");
  //  t->Branch("d0BS",jets_.trackd0BS,"d0BS[ntrack]/F");
  //  t->Branch("dzBS",jets_.trackdzBS,"dzBS[ntrack]/F");

  t->Branch("nlayer",jets_.trackNlayer,"nlayer[ntrack]/I");
  t->Branch("nlayer3D",jets_.trackNlayer3D,"nlayer3D[ntrack]/I");

  if(isMC_){
    t->Branch("pthat",&jets_.pthat,"pthat/F");
    t->Branch("trackfake",jets_.trackfake,"trackfake[ntrack]/I");
    t->Branch("parton1_flavor",&jets_.parton1_flavor,"parton1_flavor/I");
    t->Branch("parton2_flavor",&jets_.parton2_flavor,"parton2_flavor/I");
    t->Branch("parton1_pt",&jets_.parton1_pt,"parton1_pt/F");
    t->Branch("parton2_pt",&jets_.parton2_pt,"parton2_pt/F");
    t->Branch("parton2_eta",&jets_.parton2_eta,"parton2_eta/F");
    t->Branch("parton1_eta",&jets_.parton1_eta,"parton1_eta/F");
    t->Branch("parton2_phi",&jets_.parton2_phi,"parton2_phi/F");
    t->Branch("parton1_phi",&jets_.parton1_phi,"parton1_phi/F");
    t->Branch("parton1_y",&jets_.parton1_y,"parton1_y/F");
    t->Branch("parton2_y",&jets_.parton2_y,"parton2_y/F");
  }
  if(genParticleThresh_>0){
    t->Branch("ngenp",&jets_.ngenp,"ngenp/I");
    t->Branch("genppdgId",jets_.genppdgId,"genppdgId[ngenp]/I");
    t->Branch("genppt",jets_.genppt,"genppt[ngenp]/F");
    t->Branch("genpeta",jets_.genpeta,"genpeta[ngenp]/F");
    t->Branch("genpphi",jets_.genpphi,"genpphi[ngenp]/F");
  }
  if(!isMC_){
    t->Branch("nL1TBit",&jets_.nL1TBit,"nL1TBit/I");
    t->Branch("l1TBit",jets_.l1TBit,"l1TBit[nL1TBit]/O");

    t->Branch("nL1ABit",&jets_.nL1ABit,"nL1ABit/I");
    t->Branch("l1ABit",jets_.l1ABit,"l1ABit[nL1ABit]/O");

    t->Branch("nHLTBit",&jets_.nHLTBit,"nHLTBit/I");
    t->Branch("hltBit",jets_.hltBit,"hltBit[nHLTBit]/O");

  }


  TH1D::SetDefaultSumw2();


}


void
HiPFJetAnalyzer::analyze(const Event& iEvent,
			 const EventSetup& iSetup) {



  int event = iEvent.id().event();
  int run = iEvent.id().run();
  int lumi = iEvent.id().luminosityBlock();

  jets_.run = run;
  jets_.evt = event;
  jets_.lumi = lumi;

  LogDebug("HiPFJetAnalyzer")<<"START event: "<<event<<" in run "<<run<<endl;

  bool hasVertex = 0;
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByLabel(edm::InputTag("offlineBeamSpot"), beamSpotH);

  edm::Handle<vector<reco::Vertex> >vertex;
  iEvent.getByLabel(vertexTag_, vertex);



  if(vertex->size()>0 || vertex->begin()->isFake()) {
    hasVertex = 1;
    jets_.vx=vertex->begin()->x();
    jets_.vy=vertex->begin()->y();
    jets_.vz=vertex->begin()->z();

    jets_.vxErr = vertex->begin()->xError();
    jets_.vyErr = vertex->begin()->yError();
    jets_.vzErr = vertex->begin()->zError();
  }else{
    jets_.vx=beamSpotH->position().x();
    jets_.vy=beamSpotH->position().y();
    jets_.vz= 0;

    jets_.vxErr = beamSpotH->BeamWidthX();
    jets_.vyErr = beamSpotH->BeamWidthY();
    jets_.vzErr = 0;
  }

  jets_.bsx=beamSpotH->position().x();
  jets_.bsy=beamSpotH->position().y();
  jets_.bsz= beamSpotH->position().z();
  jets_.bswx=beamSpotH->BeamWidthX();
  jets_.bswy=beamSpotH->BeamWidthY();

  jets_.hasVtx = hasVertex;

  int bin = -1;
  double hf = 0.;
  double b = 999.;

  // not used, taking all jet
  //double jetPtMin = 35.;


  // loop the events

  jets_.bin = bin;
  jets_.hf = hf;

  if(!isMC_){
    fillL1Bits(iEvent);
    fillHLTBits(iEvent);
  }

  edm::Handle<pat::JetCollection> jets1;
  iEvent.getByLabel(jetTag1_, jets1);

  edm::Handle<pat::JetCollection> jets2;
  iEvent.getByLabel(jetTag2_, jets2);

  edm::Handle<pat::JetCollection> jets3;
  iEvent.getByLabel(jetTag3_, jets3);

  edm::Handle<pat::JetCollection> jets4;
  iEvent.getByLabel(jetTag4_, jets4);


  edm::Handle< edm::View<reco::CaloJet> > recoJetColl1;
  iEvent.getByLabel(recoJetTag1_, recoJetColl1 );

  edm::Handle< edm::View<reco::PFJet> > recoJetColl2;
  iEvent.getByLabel(recoJetTag2_, recoJetColl2 );

  edm::Handle< edm::View<reco::PFJet> > recoJetColl3;
  iEvent.getByLabel(recoJetTag3_, recoJetColl3 );

  edm::Handle< edm::View<reco::PFJet> > recoJetColl4;
  iEvent.getByLabel(recoJetTag4_, recoJetColl4 );


  Handle<PFCandidateCollection> pfCandidates;
  iEvent.getByLabel(pfCandidatesTag_, pfCandidates);


  Handle<vector<Track> > tracks;
  iEvent.getByLabel(trackTag_, tracks);

  // do reco-to-sim association

  Handle<TrackingParticleCollection>  TPCollectionHfake;
  Handle<edm::View<reco::Track> >  trackCollection;
  // ESHandle<TrackAssociatorBase> theAssociator;
  // const TrackAssociatorByHits *theAssociatorByHits;
  reco::RecoToSimCollection recSimColl;
  edm::Handle<reco::RecoToSimCollection > recotosimCollectionH;

  if(hasSimInfo_) {
    // iEvent.getByLabel(simTracksTag_,TPCollectionHfake);
    // iEvent.getByLabel(trackTag_,trackCollection);
    // iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits",theAssociator);
    // theAssociatorByHits = (const TrackAssociatorByHits*) theAssociator.product();
    // recSimColl= theAssociatorByHits->associateRecoToSim(trackCollection,TPCollectionHfake,&iEvent);

   iEvent.getByLabel(associatorMap_,recotosimCollectionH);
   recSimColl= *(recotosimCollectionH.product());
  }



  // FILL JRA TREE

  jets_.b = b;



  jets_.nj1 = 0;
  jets_.nunmatch_j1 = 0;



  for(unsigned int j = 0 ; j < jets1->size(); ++j){
    const pat::Jet& jet1 = (*jets1)[j];

    //cout<<" jet pt "<<jet1.pt()<<endl;
    //if(jet1.pt() < jetPtMin) continue;
    jets_.rawpt_j1[jets_.nj1]=jet1.correctedJet("Uncorrected").pt();
    jets_.jtpt_j1[jets_.nj1] = jet1.pt();
    jets_.jteta_j1[jets_.nj1] = jet1.eta();
    jets_.jtphi_j1[jets_.nj1] = jet1.phi();
    jets_.jty_j1[jets_.nj1] = jet1.eta();


    //  cout<<" abs corr "<<jet1.corrFactor("L3Absolute")<<endl;
    //cout<<" abs corr "<<jet1.corrFactor("L3Absolute")<<endl;


    float L2Corr = jet1.correctedJet("L2Relative").pt()/jet1.correctedJet("Uncorrected").pt();
    float L3Corr = jet1.correctedJet("L3Absolute").pt()/jet1.correctedJet("L2Relative").pt();


    jets_.L2_j1[jets_.nj1] = L2Corr;
    jets_.L3_j1[jets_.nj1] = L3Corr;


    jets_.area_j1[jets_.nj1] = jet1.jetArea();

    // Match to reco jet to find unsubtracted jet energy

    if(1==0){
      int recoJetSize = recoJetColl1->size();

      jets_.preL1et_j1[jets_.nj1] = -1;

      //cout<<" patJet_eta "<<jet1.eta()<<" patJet_phi "<<jet1.phi()<<" patJet_et "<<jet1.et()<<endl;

      for(int iRecoJet = 0; iRecoJet < recoJetSize; ++iRecoJet){

	reco::CaloJet recoJet1 = ((*recoJetColl1)[iRecoJet]);


	//double recoJet_eta = recoJet1.eta();
	//double recoJet_phi = recoJet1.phi();
	//cout<<" recoJet_eta "<<recoJet_eta<<" recoJet_phi "<<recoJet_phi<<" recoJet_et "<<recoJet1.et()<<endl;


	if(fabs(recoJet1.eta()-jet1.eta()) < 0.001
	   && fabs(acos(cos((recoJet1.phi()-jet1.phi())))) < 0.001)
	{
	  jets_.preL1et_j1[jets_.nj1] = recoJet1.et();

	  //cout<<"Match found,  recoJet1.et "<<recoJet1.et()<< " recoJet1.eta "<<jet1.eta()<<" recoJet1.phi "<<recoJet1.phi()<<endl;
	  break;
	}
      }
      if(jets_.preL1et_j1[jets_.nj1] == -1){


	//There's a known issue here.  If the background subtraction oversubtracts I've set the patJet.et() to zero.  That would be fine if I had also set the eta and phi.  We could then recover the pre-subtracted energy.  However, I never bothered to set the eta and phi for theses jets (doh!).  Next time I repass the data I won't be so stupid.



	if(jet1.et()>0)cout<<"Match *NOT* found,  patJet1.et "<<jet1.et()<< " patJet1.eta "<<jet1.eta()<<" patJet1.phi() "<<jet1.phi()<<endl;
      }
    }
    if(isMC_){


      if(jet1.genJet()!=0 && jet1.genJet()->pt()>1.0 && jet1.genJet()->pt()<999999){
	jets_.refpt_j1[jets_.nj1] = jet1.genJet()->pt();
	jets_.refeta_j1[jets_.nj1] = jet1.genJet()->eta();
	jets_.refphi_j1[jets_.nj1] = jet1.genJet()->phi();
	jets_.refy_j1[jets_.nj1] = jet1.genJet()->eta();

	jets_.refdrjt_j1[jets_.nj1] = reco::deltaR(jet1,*(jet1.genJet()));
      }
      else{
	jets_.refpt_j1[jets_.nj1] = 0;
	jets_.refeta_j1[jets_.nj1] = -999;
	jets_.refphi_j1[jets_.nj1] = -999;
	jets_.refy_j1[jets_.nj1] = -999;
      }

      if (jet1.genParton()) {
	jets_.refpartonpt_j1[jets_.nj1] = jet1.genParton()->pt();
	jets_.refpartonflavor_j1[jets_.nj1] = jet1.genParton()->pdgId();
      } else {
	jets_.refpartonpt_j1[jets_.nj1] = -999;
	jets_.refpartonflavor_j1[jets_.nj1] = -999;
      }

    }
    jets_.nj1++;
  }

  if(isMC_){
    edm::Handle<vector<reco::GenJet> >genjets1;
    iEvent.getByLabel(genJetTag1_, genjets1);

    for(unsigned int igen = 0 ; igen < genjets1->size(); ++igen){
      const reco::GenJet & genjet1 = (*genjets1)[igen];

      float genjet_pt = genjet1.pt();

      // threshold to reduce size of output in minbias PbPb
      if(genjet_pt>20.){

	int isMatched=0;

	for(unsigned int ijet = 0 ; ijet < jets1->size(); ++ijet){
	  const pat::Jet& jet1 = (*jets1)[ijet];

	  if(jet1.genJet()){
	    if(fabs(genjet1.pt()-jet1.genJet()->pt())<0.0001 &&
	       fabs(genjet1.eta()-jet1.genJet()->eta())<0.0001 &&
	       (fabs(genjet1.phi()-jet1.genJet()->phi())<0.0001 || fabs(fabs(genjet1.phi()-jet1.genJet()->phi()) - 2.0*TMath::Pi()) < 0.0001 )){

	      isMatched =1;
	      break;
	    }
	  }
	}

	if(!isMatched){
	  jets_.unmatchpt_j1[jets_.nunmatch_j1] = genjet_pt;
	  jets_.unmatcheta_j1[jets_.nunmatch_j1] = genjet1.eta();
	  jets_.unmatchphi_j1[jets_.nunmatch_j1] = genjet1.phi();
	  jets_.unmatchy_j1[jets_.nunmatch_j1] = genjet1.eta();

	  jets_.nunmatch_j1++;

	}

      }
    }
  }





  jets_.nj2 = 0;
  jets_.nunmatch_j2 = 0;


  for(unsigned int j = 0 ; j < jets2->size(); ++j){
    const pat::Jet& jet2 = (*jets2)[j];

    //cout<<" jet pt "<<jet2.pt()<<endl;
    //if(jet2.pt() < jetPtMin) continue;
    jets_.rawpt_j2[jets_.nj2]=jet2.correctedJet("Uncorrected").pt();
    jets_.jtpt_j2[jets_.nj2] = jet2.pt();
    jets_.jteta_j2[jets_.nj2] = jet2.eta();
    jets_.jtphi_j2[jets_.nj2] = jet2.phi();
    jets_.jty_j2[jets_.nj2] = jet2.eta();
    //  cout<<" abs corr "<<jet2.corrFactor("L3Absolute")<<endl;
    //cout<<" abs corr "<<jet2.corrFactor("L3Absolute")<<endl;


    float L2Corr = jet2.correctedJet("L2Relative").pt()/jet2.correctedJet("Uncorrected").pt();
    float L3Corr = jet2.correctedJet("L3Absolute").pt()/jet2.correctedJet("L2Relative").pt();


    jets_.L2_j2[jets_.nj2] = L2Corr;
    jets_.L3_j2[jets_.nj2] = L3Corr;

    jets_.area_j2[jets_.nj2] = jet2.jetArea();

    // Match to reco jet to find unsubtracted jet energy
    if(1==0){
      int recoJetSize2 = recoJetColl2->size();

      jets_.preL1et_j2[jets_.nj2] = -1;

      //cout<<" patJet_eta "<<jet2.eta()<<" patJet_phi "<<jet2.phi()<<" patJet_et "<<jet2.et()<<endl;

      for(int iRecoJet = 0; iRecoJet < recoJetSize2; ++iRecoJet){

	reco::PFJet recoJet2 = ((*recoJetColl2)[iRecoJet]);


	//double recoJet_eta = recoJet2.eta();
	//double recoJet_phi = recoJet2.phi();
	//cout<<" recoJet_eta "<<recoJet_eta<<" recoJet_phi "<<recoJet_phi<<" recoJet_et "<<recoJet2.et()<<endl;


	if(fabs(recoJet2.eta()-jet2.eta()) < 0.001
	   && fabs(acos(cos((recoJet2.phi()-jet2.phi())))) < 0.001)
	{
	  jets_.preL1et_j2[jets_.nj2] = recoJet2.et();

	  //cout<<"Match found,  recoJet2.et "<<recoJet2.et()<< " recoJet2.eta "<<jet2.eta()<<" recoJet2.phi "<<recoJet2.phi()<<endl;
	  break;
	}
      }
      if(jets_.preL1et_j2[jets_.nj2] == -1){


	//There's a known issue here.  If the background subtraction oversubtracts I've set the patJet.et() to zero.  That would be fine if I had also set the eta and phi.  We could then recover the pre-subtracted energy.  However, I never bothered to set the eta and phi for theses jets (doh!).  Next time I repass the data I won't be so stupid.



	if(jet2.et()>0)cout<<"Match *NOT* found,  patJet2.et "<<jet2.et()<< " patJet2.eta "<<jet2.eta()<<" patJet2.phi() "<<jet2.phi()<<endl;
      }
    }
    if(isMC_){


      if(jet2.genJet()!=0 && jet2.genJet()->pt()>1.0 && jet2.genJet()->pt()<999999){
	jets_.refpt_j2[jets_.nj2] = jet2.genJet()->pt();
	jets_.refeta_j2[jets_.nj2] = jet2.genJet()->eta();
	jets_.refphi_j2[jets_.nj2] = jet2.genJet()->phi();
	jets_.refy_j2[jets_.nj2] = jet2.genJet()->eta();

	jets_.refdrjt_j2[jets_.nj2] = reco::deltaR(jet2,*(jet2.genJet()));
      }
      else{
	jets_.refpt_j2[jets_.nj2] = 0;
	jets_.refeta_j2[jets_.nj2] = -999;
	jets_.refphi_j2[jets_.nj2] = -999;
	jets_.refy_j2[jets_.nj2] = -999;
      }

      if (jet2.genParton()) {
	jets_.refpartonpt_j2[jets_.nj2] = jet2.genParton()->pt();
	jets_.refpartonflavor_j2[jets_.nj2] = jet2.genParton()->pdgId();
      } else {
	jets_.refpartonpt_j2[jets_.nj2] = -999;
	jets_.refpartonflavor_j2[jets_.nj2] = -999;
      }
    }


    jets_.nj2++;

  }

  if(isMC_){

    edm::Handle<vector<reco::GenJet> >genjets2;
    iEvent.getByLabel(genJetTag2_, genjets2);

    for(unsigned int igen = 0 ; igen < genjets2->size(); ++igen){
      const reco::GenJet & genjet2 = (*genjets2)[igen];

      float genjet_pt = genjet2.pt();

      // threshold to reduce size of output in minbias PbPb
      if(genjet_pt>20.){

	int isMatched=0;

	for(unsigned int ijet = 0 ; ijet < jets2->size(); ++ijet){
	  const pat::Jet& jet2 = (*jets2)[ijet];

	  if(jet2.genJet()){
	    if(fabs(genjet2.pt()-jet2.genJet()->pt())<0.0001 &&
	       fabs(genjet2.eta()-jet2.genJet()->eta())<0.0001 &&
	       (fabs(genjet2.phi()-jet2.genJet()->phi())<0.0001 || fabs(fabs(genjet2.phi()-jet2.genJet()->phi()) - 2.0*TMath::Pi()) < 0.0001 )){

	      isMatched =1;
	      break;
	    }
	  }
	}

	if(!isMatched){
	  jets_.unmatchpt_j2[jets_.nunmatch_j2] = genjet_pt;
	  jets_.unmatcheta_j2[jets_.nunmatch_j2] = genjet2.eta();
	  jets_.unmatchphi_j2[jets_.nunmatch_j2] = genjet2.phi();
	  jets_.unmatchy_j2[jets_.nunmatch_j2] = genjet2.eta();

	  jets_.nunmatch_j2++;

	}

      }
    }
  }



  jets_.nj3 = 0;
  jets_.nunmatch_j3 = 0;

  //cout<<" jets size "<<jets->size()<<endl;


  for(unsigned int j = 0 ; j < jets3->size(); ++j){
    const pat::Jet& jet3 = (*jets3)[j];

    //cout<<" jet pt "<<jet3.pt()<<endl;
    //if(jet3.pt() < jetPtMin) continue;
    jets_.rawpt_j3[jets_.nj3]=jet3.correctedJet("Uncorrected").pt();
    jets_.jtpt_j3[jets_.nj3] = jet3.pt();
    jets_.jteta_j3[jets_.nj3] = jet3.eta();
    jets_.jtphi_j3[jets_.nj3] = jet3.phi();
    jets_.jty_j3[jets_.nj3] = jet3.eta();
    //  cout<<" abs corr "<<jet3.corrFactor("L3Absolute")<<endl;
    //cout<<" abs corr "<<jet3.corrFactor("L3Absolute")<<endl;



    float L2Corr = jet3.correctedJet("L2Relative").pt()/jet3.correctedJet("Uncorrected").pt();
    float L3Corr = jet3.correctedJet("L3Absolute").pt()/jet3.correctedJet("L2Relative").pt();


    jets_.L2_j3[jets_.nj3] = L2Corr;
    jets_.L3_j3[jets_.nj3] = L3Corr;

    jets_.area_j3[jets_.nj3] = jet3.jetArea();

    // Match to reco jet to find unsubtracted jet energy
    if(1==0){
      int recoJetSize3 = recoJetColl3->size();

      jets_.preL1et_j3[jets_.nj3] = -1;

      //cout<<" patJet_eta "<<jet3.eta()<<" patJet_phi "<<jet3.phi()<<" patJet_et "<<jet3.et()<<endl;

      for(int iRecoJet = 0; iRecoJet < recoJetSize3; ++iRecoJet){

	reco::PFJet recoJet3 = ((*recoJetColl3)[iRecoJet]);


	//double recoJet_eta = recoJet3.eta();
	//double recoJet_phi = recoJet3.phi();
	//cout<<" recoJet_eta "<<recoJet_eta<<" recoJet_phi "<<recoJet_phi<<" recoJet_et "<<recoJet3.et()<<endl;


	if(fabs(recoJet3.eta()-jet3.eta()) < 0.001
	   && fabs(acos(cos((recoJet3.phi()-jet3.phi())))) < 0.001)
	{
	  jets_.preL1et_j3[jets_.nj3] = recoJet3.et();

	  //cout<<"Match found,  recoJet3.et "<<recoJet3.et()<< " recoJet3.eta "<<jet3.eta()<<" recoJet3.phi "<<recoJet3.phi()<<endl;
	  break;
	}
      }
      if(jets_.preL1et_j3[jets_.nj3] == -1){


	//  There's a known issue here.  If the background subtraction oversubtracts I've set the patJet.et() to zero.  That would be fine if I had also set the eta and phi.  We could then recover the pre-subtracted energy.  However, I never bothered to set the eta and phi for theses jets (doh!).  Next time I repass the data I won't be so stupid.



	if(jet3.et()>0)cout<<"Match *NOT* found,  patJet3.et "<<jet3.et()<< " patJet3.eta "<<jet3.eta()<<" patJet3.phi() "<<jet3.phi()<<endl;
      }
    }
    if(isMC_){


      if(jet3.genJet()!=0 && jet3.genJet()->pt()>1.0 && jet3.genJet()->pt()<999999){
	jets_.refpt_j3[jets_.nj3] = jet3.genJet()->pt();
	jets_.refeta_j3[jets_.nj3] = jet3.genJet()->eta();
	jets_.refphi_j3[jets_.nj3] = jet3.genJet()->phi();
	jets_.refy_j3[jets_.nj3] = jet3.genJet()->eta();

	jets_.refdrjt_j3[jets_.nj3] = reco::deltaR(jet3,*(jet3.genJet()));
      }
      else{
	jets_.refpt_j3[jets_.nj3] = 0;
	jets_.refeta_j3[jets_.nj3] = -999;
	jets_.refphi_j3[jets_.nj3] = -999;
	jets_.refy_j3[jets_.nj3] = -999;
      }

      if (jet3.genParton()) {
	jets_.refpartonpt_j3[jets_.nj3] = jet3.genParton()->pt();
	jets_.refpartonflavor_j3[jets_.nj3] = jet3.genParton()->pdgId();
      } else {
	jets_.refpartonpt_j3[jets_.nj3] = -999;
	jets_.refpartonflavor_j3[jets_.nj3] = -999;
      }

    }



    jets_.nj3++;


  }

  if(isMC_){

    edm::Handle<vector<reco::GenJet> >genjets3;
    iEvent.getByLabel(genJetTag3_, genjets3);

    for(unsigned int igen = 0 ; igen < genjets3->size(); ++igen){
      const reco::GenJet & genjet3 = (*genjets3)[igen];

      float genjet_pt = genjet3.pt();

      // threshold to reduce size of output in minbias PbPb
      if(genjet_pt>20.){

	int isMatched=0;

	for(unsigned int ijet = 0 ; ijet < jets3->size(); ++ijet){
	  const pat::Jet& jet3 = (*jets3)[ijet];

	  if(jet3.genJet()){
	    if(fabs(genjet3.pt()-jet3.genJet()->pt())<0.0001 &&
	       fabs(genjet3.eta()-jet3.genJet()->eta())<0.0001 &&
	       (fabs(genjet3.phi()-jet3.genJet()->phi())<0.0001 || fabs(fabs(genjet3.phi()-jet3.genJet()->phi()) - 2.0*TMath::Pi()) < 0.0001 )){

	      isMatched =1;
	      break;
	    }
	  }
	}

	if(!isMatched){
	  jets_.unmatchpt_j3[jets_.nunmatch_j3] = genjet_pt;
	  jets_.unmatcheta_j3[jets_.nunmatch_j3] = genjet3.eta();
	  jets_.unmatchphi_j3[jets_.nunmatch_j3] = genjet3.phi();
	  jets_.unmatchy_j3[jets_.nunmatch_j3] = genjet3.eta();

	  jets_.nunmatch_j3++;

	}

      }
    }
  }




  jets_.nj4 = 0;
  jets_.nunmatch_j4 = 0;


  for(unsigned int j = 0 ; j < jets4->size(); ++j){
    const pat::Jet& jet4 = (*jets4)[j];

    //cout<<" jet pt "<<jet4.pt()<<endl;
    //if(jet4.pt() < jetPtMin) continue;
    jets_.rawpt_j4[jets_.nj4]=jet4.correctedJet("Uncorrected").pt();
    jets_.jtpt_j4[jets_.nj4] = jet4.pt();
    jets_.jteta_j4[jets_.nj4] = jet4.eta();
    jets_.jtphi_j4[jets_.nj4] = jet4.phi();
    jets_.jty_j4[jets_.nj4] = jet4.eta();
    //  cout<<" abs corr "<<jet4.corrFactor("L3Absolute")<<endl;
    //cout<<" abs corr "<<jet4.corrFactor("L3Absolute")<<endl;


    float L2Corr = jet4.correctedJet("L2Relative").pt()/jet4.correctedJet("Uncorrected").pt();
    float L3Corr = jet4.correctedJet("L3Absolute").pt()/jet4.correctedJet("L2Relative").pt();


    jets_.L2_j4[jets_.nj4] = L2Corr;
    jets_.L3_j4[jets_.nj4] = L3Corr;

    jets_.area_j4[jets_.nj4] = jet4.jetArea();

    // Match to reco jet to find unsubtracted jet energy
    if(1==0){
      int recoJetSize4 = recoJetColl4->size();

      jets_.preL1et_j4[jets_.nj4] = -1;

      //cout<<" patJet_eta "<<jet4.eta()<<" patJet_phi "<<jet4.phi()<<" patJet_et "<<jet4.et()<<endl;

      for(int iRecoJet = 0; iRecoJet < recoJetSize4; ++iRecoJet){

	reco::PFJet recoJet4 = ((*recoJetColl4)[iRecoJet]);


	//double recoJet_eta = recoJet4.eta();
	//double recoJet_phi = recoJet4.phi();
	//cout<<" recoJet_eta "<<recoJet_eta<<" recoJet_phi "<<recoJet_phi<<" recoJet_et "<<recoJet4.et()<<endl;


	if(fabs(recoJet4.eta()-jet4.eta()) < 0.001
	   && fabs(acos(cos((recoJet4.phi()-jet4.phi())))) < 0.001)
	{
	  jets_.preL1et_j4[jets_.nj4] = recoJet4.et();

	  //cout<<"Match found,  recoJet4.et "<<recoJet4.et()<< " recoJet4.eta "<<jet4.eta()<<" recoJet4.phi "<<recoJet4.phi()<<endl;
	  break;
	}
      }
      if(jets_.preL1et_j4[jets_.nj4] == -1){


	//There's a known issue here.  If the background subtraction oversubtracts I've set the patJet.et() to zero.  That would be fine if I had also set the eta and phi.  We could then recover the pre-subtracted energy.  However, I never bothered to set the eta and phi for theses jets (doh!).  Next time I repass the data I won't be so stupid.



	if(jet4.et()>0)cout<<"Match *NOT* found,  patJet4.et "<<jet4.et()<< " patJet4.eta "<<jet4.eta()<<" patJet4.phi() "<<jet4.phi()<<endl;
      }
    }
    if(isMC_){


      if(jet4.genJet()!=0 && jet4.genJet()->pt()>1.0 && jet4.genJet()->pt()<999999){
	jets_.refpt_j4[jets_.nj4] = jet4.genJet()->pt();
	jets_.refeta_j4[jets_.nj4] = jet4.genJet()->eta();
	jets_.refphi_j4[jets_.nj4] = jet4.genJet()->phi();
	jets_.refy_j4[jets_.nj4] = jet4.genJet()->eta();

	jets_.refdrjt_j4[jets_.nj4] = reco::deltaR(jet4,*(jet4.genJet()));
      }
      else{
	jets_.refpt_j4[jets_.nj4] = 0;
	jets_.refeta_j4[jets_.nj4] = -999;
	jets_.refphi_j4[jets_.nj4] = -999;
	jets_.refy_j4[jets_.nj4] = -999;
      }

      if (jet4.genParton()) {
	jets_.refpartonpt_j4[jets_.nj4] = jet4.genParton()->pt();
	jets_.refpartonflavor_j4[jets_.nj4] = jet4.genParton()->pdgId();
      } else {
	jets_.refpartonpt_j4[jets_.nj4] = -999;
	jets_.refpartonflavor_j4[jets_.nj4] = -999;
      }

    }


    jets_.nj4++;

  }

  if(isMC_){

    edm::Handle<vector<reco::GenJet> >genjets4;
    iEvent.getByLabel(genJetTag4_, genjets4);

    for(unsigned int igen = 0 ; igen < genjets4->size(); ++igen){
      const reco::GenJet & genjet4 = (*genjets4)[igen];

      float genjet_pt = genjet4.pt();

      // threshold to reduce size of output in minbias PbPb
      if(genjet_pt>20.){

	int isMatched=0;

	for(unsigned int ijet = 0 ; ijet < jets4->size(); ++ijet){
	  const pat::Jet& jet4 = (*jets4)[ijet];

	  if(jet4.genJet()){
	    if(fabs(genjet4.pt()-jet4.genJet()->pt())<0.0001 &&
	       fabs(genjet4.eta()-jet4.genJet()->eta())<0.0001 &&
	       (fabs(genjet4.phi()-jet4.genJet()->phi())<0.0001 || fabs(fabs(genjet4.phi()-jet4.genJet()->phi()) - 2.0*TMath::Pi()) < 0.0001 )){

	      isMatched =1;
	      break;
	    }
	  }
	}

	if(!isMatched){
	  jets_.unmatchpt_j4[jets_.nunmatch_j4] = genjet_pt;
	  jets_.unmatcheta_j4[jets_.nunmatch_j4] = genjet4.eta();
	  jets_.unmatchphi_j4[jets_.nunmatch_j4] = genjet4.phi();
	  jets_.unmatchy_j4[jets_.nunmatch_j4] = genjet4.eta();

	  jets_.nunmatch_j4++;

	}

      }
    }
  }



  for( unsigned icand=0; icand<pfCandidates->size(); icand++ ) {

    const reco::PFCandidate& cand = (*pfCandidates)[icand];

    float particleEta = cand.eta();

    //if(fabs(particleEta)>2.5) continue;



    // PF PId Convention:
    // 1 = Charged Hadrons
    // 2 = Electrons (not included)
    // 3 = Muons
    // 4 = Photons
    // 5 = Neutral Hadrons




    int particleId = (int)cand.particleId();
    float particlePt = cand.pt();

    if(particlePt<0.3) continue;


    // can use varid thresholds if we want
    //if(particleId==1 && particlePt < 0.9) continue;
    //if(particleId==3 && particlePt < 0.9) continue;
    //if(particleId==4 && particlePt < 0.3) continue;
    //if(particleId==5 && particlePt < 0.9) continue;


    if(particleId==3&&particlePt>100) cout<<" likely a badly reconstructed MUON "<<endl;

    jets_.candId[jets_.nPFcand] = particleId;
    jets_.candpt[jets_.nPFcand] = particlePt;
    jets_.candeta[jets_.nPFcand] = particleEta;
    jets_.candphi[jets_.nPFcand] = cand.phi();
    //jets_.candy[jets_.nPFcand] = cand.y();

    if(particleId==3&&particlePt>100) cout<<" found a misreconstructed MUON, pT =  "<<particlePt<<endl;

    jets_.nPFcand++;

    //cout<<" jets_.nPFcand "<<jets_.nPFcand<<endl;
  }

  //cout<<" ntracks: "<<tracks->size()<<endl;

  for(unsigned int it=0; it<tracks->size(); ++it){
    const reco::Track & track = (*tracks)[it];

    int count1dhits = 0;
    double chi2n_hit1D = 0;
    trackingRecHit_iterator edh = track.recHitsEnd();
    for (trackingRecHit_iterator ith = track.recHitsBegin(); ith != edh; ++ith) {
      // const TrackingRecHit * hit = ith->get();
      //DetId detid = hit->geographicalId();
      if ((*ith)->isValid()) {
	if (typeid(*ith) == typeid(SiStripRecHit1D)) ++count1dhits;
      }
    }
    if (count1dhits > 0) {
      double chi2 = track.chi2();
      double ndof = track.ndof();
      chi2n_hit1D = (chi2+count1dhits)/double(ndof+count1dhits);
    }

    // Could makes some track selection here
    jets_.tracknhits[jets_.ntrack] = track.numberOfValidHits();
    jets_.trackpt[jets_.ntrack] = track.pt();
    jets_.tracketa[jets_.ntrack] = track.eta();
    jets_.trackphi[jets_.ntrack] = track.phi();

    jets_.trackptErr[jets_.ntrack] = track.ptError();
    jets_.trackchi2[jets_.ntrack] = track.normalizedChi2();
    jets_.trackchi2hit1D[jets_.ntrack] = chi2n_hit1D;

    jets_.tracksumecal[jets_.ntrack] = 0.;
    jets_.tracksumhcal[jets_.ntrack] = 0.;

    reco::TrackBase::TrackQuality trackQualityTight = TrackBase::qualityByName("highPurity");
    jets_.trackqual[jets_.ntrack]=(int)track.quality(trackQualityTight);

    jets_.trackfake[jets_.ntrack]=0;

    reco::TrackRef trackRef=reco::TrackRef(tracks,it);

    if(hasVertex){
      jets_.trackd0[jets_.ntrack] = -track.dxy(vertex->begin()->position());
      jets_.trackdz[jets_.ntrack] = track.dz(vertex->begin()->position());
      jets_.trackd0Err[jets_.ntrack] = sqrt ( (track.d0Error()*track.d0Error()) + (vertex->begin()->xError()*vertex->begin()->yError()) );
      jets_.trackdzErr[jets_.ntrack] = sqrt ( (track.dzError()*track.dzError()) + (vertex->begin()->zError()*vertex->begin()->zError()) );
    }else{
      jets_.trackd0[jets_.ntrack] = -track.dxy(beamSpotH->position());
      jets_.trackdz[jets_.ntrack] = 0;
      jets_.trackd0Err[jets_.ntrack] = sqrt ( (track.d0Error()*track.d0Error()) +  (beamSpotH->BeamWidthX()*beamSpotH->BeamWidthY()) );
      jets_.trackdzErr[jets_.ntrack] = 0;
    }

    jets_.trackd0BS[jets_.ntrack] = -track.dxy(beamSpotH->position());
    jets_.trackdzBS[jets_.ntrack] = track.dz(beamSpotH->position());
    jets_.trackd0ErrBS[jets_.ntrack] = sqrt ( (track.d0Error()*track.d0Error()) +  (beamSpotH->BeamWidthX()*beamSpotH->BeamWidthY()) );
    jets_.trackdzErrBS[jets_.ntrack] = 0;

    jets_.trackd0ErrTrk[jets_.ntrack] = track.d0Error();
    jets_.trackdzErrTrk[jets_.ntrack] = track.dzError();

    jets_.trackNlayer[jets_.ntrack] = track.hitPattern().trackerLayersWithMeasurement();
    jets_.trackNlayer3D[jets_.ntrack] = track.hitPattern().pixelLayersWithMeasurement() + track.hitPattern().numberOfValidStripLayersWithMonoAndStereo();

    if(hasSimInfo_)
      if(recSimColl.find(edm::RefToBase<reco::Track>(trackRef)) == recSimColl.end())
	jets_.trackfake[jets_.ntrack]=1;



    int pfCandMatchFound = 0;

    // loop over pf candidates to get calo-track matching info
    for( unsigned icand=0; icand<pfCandidates->size(); icand++ ) {

      const reco::PFCandidate& cand = (*pfCandidates)[icand];

      float cand_type = cand.particleId();

      // only charged hadrons and leptons can be asscociated with a track
      if(!(cand_type == PFCandidate::h ||     //type1
	   cand_type == PFCandidate::e ||     //type2
	   cand_type == PFCandidate::mu      //type3
	   )
	) continue;


      // if working with 2 different track collections this doesn't work
      if(cand.trackRef() != trackRef) continue;
      //if(fabs(cand.pt()-track.pt())>0.001||fabs(cand.eta()-track.eta())>0.001||fabs(acos(cos(cand.phi()-track.phi())))>0.001) continue;

      pfCandMatchFound = 1;

      for(unsigned iblock=0; iblock<cand.elementsInBlocks().size(); iblock++) {

	PFBlockRef blockRef = cand.elementsInBlocks()[iblock].first;
	unsigned indexInBlock = cand.elementsInBlocks()[iblock].second;


	const edm::OwnVector<  reco::PFBlockElement>&  elements = (*blockRef).elements();

	//This tells you what type of element it is:
	//cout<<" block type"<<elements[indexInBlock].type()<<endl;

	switch (elements[indexInBlock].type()) {

	case PFBlockElement::ECAL: {
	  reco::PFClusterRef clusterRef = elements[indexInBlock].clusterRef();
	  double eet = clusterRef->energy()/cosh(clusterRef->eta());
	  if(verbose_)cout<<" ecal energy "<<clusterRef->energy()<<endl;
	  jets_.tracksumecal[jets_.ntrack] += eet;
	  break;
	}

	case PFBlockElement::HCAL: {
	  reco::PFClusterRef clusterRef = elements[indexInBlock].clusterRef();
	  double eet = clusterRef->energy()/cosh(clusterRef->eta());
	  if(verbose_)cout<<" hcal energy "<<clusterRef->energy()<<endl;
	  jets_.tracksumhcal[jets_.ntrack] += eet;
	  break;
	}
	case PFBlockElement::TRACK: {
	  //This is just the reference to the track itself, since tracks can never be linked to other tracks
	  break;
	}
	default:
	  break;
	}
	// Could do more stuff here, e.g., pre-shower, HF

      }

    }

    if(!pfCandMatchFound){
      jets_.tracksumecal[jets_.ntrack] =-1;
      jets_.tracksumhcal[jets_.ntrack] =-1;

    }

    jets_.ntrack++;

  }

  // make configurable, so that gen particles aren't run with MB

  if(isMC_){

    edm::Handle<GenEventInfoProduct> hEventInfo;
    iEvent.getByLabel(eventInfoTag_,hEventInfo);

    jets_.pthat = hEventInfo->qScale();

    //getPartons(iEvent, iSetup );

    if(genParticleThresh_>0){
      edm::Handle <reco::GenParticleCollection> genParticles;
      iEvent.getByLabel (genParticleTag_, genParticles );


      for( unsigned igen=0; igen<genParticles->size(); igen++ ) {


	const reco::GenParticle & genp = (*genParticles)[igen];

	if(genp.status()!=1) continue;

	jets_.genppt[jets_.ngenp] = genp.pt();
	jets_.genpeta[jets_.ngenp] = genp.eta();
	jets_.genpphi[jets_.ngenp] = genp.phi();
	jets_.genppdgId[jets_.ngenp] = genp.pdgId();

	jets_.ngenp++;
      }
    }
  }


  t->Fill();



  jets_.nj1 = 0;
  jets_.nj2 = 0;
  jets_.nj3 = 0;
  jets_.nj4 = 0;
  jets_.nPFcand = 0;
  jets_.ntrack = 0;
  jets_.ngenp = 0;

}


// copied from PhysicsTools/JetMCAlgos/plugins/PartonSelector.cc
void HiPFJetAnalyzer::getPartons( const Event& iEvent, const EventSetup& iEs )
{

  //edm::Handle <reco::CandidateView> genParticles;
  edm::Handle <reco::GenParticleCollection> genParticles;
  iEvent.getByLabel (genParticleTag_, genParticles );

  auto_ptr<GenParticleRefVector> thePartons ( new GenParticleRefVector);


  const GenParticle & parton1 = (*genParticles)[ 6 ];
  jets_.parton1_flavor = abs(parton1.pdgId());
  jets_.parton1_pt = parton1.pt();
  jets_.parton1_phi =  parton1.phi();
  jets_.parton1_eta = parton1.eta();
  jets_.parton1_y = parton1.y();

  const GenParticle & parton2 = (*genParticles)[ 7 ];
  jets_.parton2_flavor = abs(parton2.pdgId());
  jets_.parton2_pt = parton2.pt();
  jets_.parton2_phi =  parton2.phi();
  jets_.parton2_eta = parton2.eta();
  jets_.parton2_y = parton2.y();




}


//--------------------------------------------------------------------------------------------------
void HiPFJetAnalyzer::fillL1Bits(const edm::Event &iEvent)
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
void HiPFJetAnalyzer::fillHLTBits(const edm::Event &iEvent)
{
  // Fill HLT trigger bits.
  Handle<TriggerResults> triggerResultsHLT;
  getProduct(hltResName_, triggerResultsHLT, iEvent);

  const TriggerResults *hltResults = triggerResultsHLT.product();
  const TriggerNames & triggerNames = iEvent.triggerNames(*hltResults);

  jets_.nHLTBit = triggerNames.size();

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
inline void HiPFJetAnalyzer::getProduct(const std::string name, edm::Handle<TYPE> &prod,
					const edm::Event &event) const
{
  // Try to access data collection from EDM file. We check if we really get just one
  // product with the given name. If not we throw an exception.

  event.getByLabel(edm::InputTag(name),prod);
  if (!prod.isValid())
    throw edm::Exception(edm::errors::Configuration, "HiPFJetAnalyzer::GetProduct()\n")
      << "Collection with label '" << name << "' is not valid" <<  std::endl;
}

//--------------------------------------------------------------------------------------------------
template <typename TYPE>
inline bool HiPFJetAnalyzer::getProductSafe(const std::string name, edm::Handle<TYPE> &prod,
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






DEFINE_FWK_MODULE(HiPFJetAnalyzer);
