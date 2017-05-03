// -*- C++ -*-
//
// Package:    HeavyIonsAnalysis/HiMixValidation
// Class:      HiMixValidation
// 
/**\class HiMixValidation HiMixValidation.cc HeavyIonsAnalysis/HiMixValidation/plugins/HiMixValidation.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Wed, 15 Oct 2014 15:12:55 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfoNew.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TTree.h"
#include "TH1D.h"
#include "TH2D.h"

#include <iostream>
using namespace std;
using namespace edm;

//
// class declaration
//

struct EventArray{

   int nmix;
   int nbx;

   ULong64_t event[100];
   ULong64_t lumi[100];
   ULong64_t run[100];
   ULong64_t file[100];

   int bx[100];

   float xVtx[100];
   float yVtx[100];
   float zVtx[100];




};



class HiMixValidation : public edm::EDAnalyzer {
   public:
      explicit HiMixValidation(const edm::ParameterSet&);
      ~HiMixValidation();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------

   edm::InputTag genParticleSrc_;
   edm::InputTag genHIsrc_;
   edm::InputTag vertexSrc_;
   edm::InputTag genJetSrc_;
   edm::InputTag jetSrc_;
   edm::InputTag playbackSrc_;
   edm::EDGetTokenT<CrossingFrame<HepMCProduct> >   cfLabel;

   TH1D *hGenParticleEtaSignal, 
      *hGenParticleEtaBkg, 
      *hGenParticleEtaBkgNpart2, 
      *hGenParticleEtaMixed,
      *hGenJetEtaSignal,
      *hGenJetEtaBkg,
      *hGenJetEtaBkgNpart2,
      *hGenJetEtaMixed,
      *hGenPhotonEtaSignal,
      *hGenPhotonEtaBkg,
      *hGenPhotonEtaBkgNpart2,
      *hGenPhotonEtaMixed,
      *hGenMuonEtaSignal,
      *hGenMuonEtaBkg,
      *hGenMuonEtaBkgNpart2,
      *hGenMuonEtaMixed,
      *hGenElectronEtaSignal,
      *hGenElectronEtaBkg,
      *hGenElectronEtaBkgNpart2,
      *hGenElectronEtaMixed,
      *hCentralityBin,
      *hNrecoVertex,
      *hZrecoVertex0,
      *hZrecoVertex1;

   TH2D *hZrecoVertices,
      *hNtrackHF,
      *hJetResponseSignal,
      *hPhotonResponseSignal,
      *hMuonResponseSignal,
      *hElectronResponseSignal,
      *hJetResponseBkg,
      *hPhotonResponseBkg,
      *hMuonResponseBkg,
      *hElectronResponseBkg,
      *hJetResponseMixed,
      *hPhotonResponseMixed,
      *hMuonResponseMixed,
      *hElectronResponseMixed,
      *hGenVertices,
      *hGenVerticesCF;

   TTree *t;

   double particlePtMin;
   double jetPtMin;
   double photonPtMin;
   double muonPtMin;
   double electronPtMin;

   double rMatch;
   double rMatch2;
   bool useCF_;

   bool doGEN_;
   bool doSIM_;
   bool doRAW_;
   bool doRECO_;
   bool doHIST_;


   edm::Service<TFileService> f;

   EventArray piles;
};

HiMixValidation::HiMixValidation(const edm::ParameterSet& iConfig)
{

   playbackSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("playbackSrc",edm::InputTag("mix"));

   genParticleSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("genpSrc",edm::InputTag("hiGenParticles"));
   genJetSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("genJetSrc",edm::InputTag("ak3HiGenJets"));
   genHIsrc_ = iConfig.getUntrackedParameter<edm::InputTag>("genHiSrc",edm::InputTag("heavyIon"));
   jetSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("jetSrc",edm::InputTag("akPu3PFJets"));

   vertexSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("vertexSrc",edm::InputTag("hiSelectedVertex"));
   particlePtMin = iConfig.getUntrackedParameter<double>("particlePtMin",0.);
   jetPtMin = iConfig.getUntrackedParameter<double>("jetPtMin",50.);
   photonPtMin = iConfig.getUntrackedParameter<double>("photonPtMin",30.);
   muonPtMin = iConfig.getUntrackedParameter<double>("muonPtMin",3.);
   electronPtMin = iConfig.getUntrackedParameter<double>("electronPtMin",3.);
   
   doGEN_ = iConfig.getUntrackedParameter<bool>("doGEN",true);
   doSIM_ = iConfig.getUntrackedParameter<bool>("doSIM",true);
   doRAW_ = iConfig.getUntrackedParameter<bool>("doRAW",true);
   doRECO_ = iConfig.getUntrackedParameter<bool>("doRECO",true);
   doHIST_ = iConfig.getUntrackedParameter<bool>("doHIST",true);

   useCF_ = iConfig.getUntrackedParameter<bool>("useCF",false);
   if(useCF_) cfLabel = consumes<CrossingFrame<edm::HepMCProduct> >(iConfig.getUntrackedParameter<edm::InputTag>("mixLabel",edm::InputTag("mix","generator")));

   rMatch = iConfig.getUntrackedParameter<double>("rMatch",0.3);
   rMatch2 = rMatch*rMatch;
};

HiMixValidation::~HiMixValidation()
{
}

void
HiMixValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   double mm = 0.1;


   Handle<CrossingFramePlaybackInfoNew> playback;
   iEvent.getByLabel(playbackSrc_,playback);

   piles.nmix = playback->eventInfo_.size()+1;
   piles.nbx = playback->nBcrossings_;

   piles.event[0] = iEvent.id().event();
   piles.run[0] = iEvent.id().run();
   piles.lumi[0] = iEvent.id().luminosityBlock();

   piles.bx[0] = iEvent.bunchCrossing();

   piles.file[0] = 0;

   int bx = 0;
   int sbx = 0;

   for(unsigned int i = 0; i < playback->eventInfo_.size(); ++i){

      if(i > sbx + playback->sizes_[bx]){
         sbx += playback->sizes_[bx];
         bx++;
      }

      EventID const& id = playback->eventInfo_[i].eventID();
      piles.event[i+1] = id.event();
      piles.run[i+1] = id.run();
      piles.lumi[i+1] = id.luminosityBlock();

      piles.bx[i+1] = playback->minBunch_ + bx;
      piles.file[i+1] = playback->eventInfo_[i].fileNameHash();

   }


   Handle<GenHIEvent> higen;
   iEvent.getByLabel(genHIsrc_,higen);
   double npart = higen->Npart();

   Handle<reco::GenParticleCollection> parts;
   iEvent.getByLabel(genParticleSrc_,parts);
   cout<<"x1"<<endl;
   double zgen[2]={-29,-29};
   cout<<"x2"<<endl;





   for(UInt_t i = 0; i < parts->size(); ++i){
      const reco::GenParticle& p = (*parts)[i];
      int sube = p.collisionId();

      if(p.numberOfMothers() == 0){
	 if(sube == 0){
	    zgen[0] = p.vz();
	 }else{
            zgen[1] = p.vz();
	 }
      }

      if (p.status()!=1) continue;
      int pdg = abs(p.pdgId());
      double eta = p.eta();
      double pt = p.pt();
      if(doHIST_){

      if(sube == 0){
	 if(p.charge() != 0 && pt > particlePtMin) hGenParticleEtaSignal->Fill(eta);
	 if(pdg == 22 && pt > photonPtMin) hGenPhotonEtaSignal->Fill(eta);	 
	 if(pdg == 11 && pt > electronPtMin) hGenElectronEtaSignal->Fill(eta);	 
	 if(pdg == 13 && pt > muonPtMin){
	    hGenMuonEtaSignal->Fill(eta);
	 }
      }else{
	 if(p.charge() != 0 && pt > particlePtMin) hGenParticleEtaBkg->Fill(eta);
         if(pdg == 22 && pt > photonPtMin) hGenPhotonEtaBkg->Fill(eta);
         if(pdg == 11 && pt > electronPtMin) hGenElectronEtaBkg->Fill(eta);
         if(pdg == 13 && pt > muonPtMin) hGenMuonEtaBkg->Fill(eta);

	 if(npart == 2){
	    if(p.charge() != 0 && pt > particlePtMin) hGenParticleEtaBkgNpart2->Fill(eta);
	    if(pdg == 22 && pt > photonPtMin) hGenPhotonEtaBkgNpart2->Fill(eta);
	    if(pdg == 11 && pt > electronPtMin) hGenElectronEtaBkgNpart2->Fill(eta);
	    if(pdg == 13 && pt > muonPtMin) hGenMuonEtaBkgNpart2->Fill(eta);
	 }
      }     
      if(p.charge() != 0 && pt > particlePtMin) hGenParticleEtaMixed->Fill(eta);
      if(pdg == 22 && pt > photonPtMin) hGenPhotonEtaMixed->Fill(eta);
      if(pdg == 11 && pt > electronPtMin) hGenElectronEtaMixed->Fill(eta);
      if(pdg == 13 && pt > muonPtMin) hGenMuonEtaMixed->Fill(eta);
      }
   }
   cout<<"x3"<<endl;

   if(doHIST_) hGenVertices->Fill(zgen[0],zgen[1]);
   cout<<"x4"<<endl;

   edm::Handle<reco::GenJetCollection> genjets;
   iEvent.getByLabel(genJetSrc_,genjets);
   edm::Handle<reco::JetView> jets;
   iEvent.getByLabel(jetSrc_,jets);
   cout<<"x5"<<endl;

   for(UInt_t i = 0; i < genjets->size(); ++i){
      const reco::GenJet& p = (*genjets)[i];
      double pt = p.pt();
      double ptmatch = 0;
      if(pt < jetPtMin) continue;

      for(UInt_t j = 0; j < jets->size(); ++j){
	 if(rMatch2 > reco::deltaR2((*genjets)[i],(*jets)[j])){
	    double recopt = (*jets)[j].pt();
	    if(recopt>ptmatch) ptmatch = recopt; 
	 }
      }

      int sube = p.getGenConstituent(0)->collisionId();
      double eta = p.eta();
      
      if(sube == 0){ 
	 hGenJetEtaSignal->Fill(eta);
	 hJetResponseSignal->Fill(pt,ptmatch);
      }else{
	 hGenJetEtaBkg->Fill(eta);
         hJetResponseBkg->Fill(pt,ptmatch);

	 if(npart == 2){
	    hGenJetEtaBkgNpart2->Fill(eta);
	 }
      }
      hGenJetEtaMixed->Fill(eta);
      hJetResponseMixed->Fill(pt,ptmatch);
   }
   cout<<"x6"<<endl;

   // Gen-Vertices from CrossingFrame
   if(useCF_){

      Handle<CrossingFrame<edm::HepMCProduct> > cf;
      cout<<"x7"<<endl;

      iEvent.getByToken(cfLabel,cf);
      cout<<"x8"<<endl;

      MixCollection<edm::HepMCProduct> mix(cf.product());
      cout<<"x9"<<endl;
      HepMC::GenVertex* genvtx = 0;
      const HepMC::GenEvent* inev = 0;
      cout<<"x10"<<endl;
      double zcf[2]={-29,-29};
      if(mix.size() != 2){
	 cout<<"More or less than 2 sub-events, mixing seems to have failed! Size : "<<mix.size()<<endl;
      }else{
	 for(int i = 0; i < 2; ++i){
	    cout<<"i "<<i<<endl;
	    const edm::HepMCProduct& bkg = mix.getObject(i);
	    cout<<"a"<<endl;
	    inev = bkg.GetEvent();
	    cout<<"b"<<endl;
	    
	    genvtx = inev->signal_process_vertex();
	    cout<<"c"<<endl;
	    
	    if(!genvtx){
	       cout<<"No Signal Process Vertex!"<<endl;
	       HepMC::GenEvent::particle_const_iterator pt=inev->particles_begin();
	       HepMC::GenEvent::particle_const_iterator ptend=inev->particles_end();
	       while(!genvtx || ( genvtx->particles_in_size() == 1 && pt != ptend ) ){
		  if(pt == ptend) cout<<"End reached, No Gen Vertex!"<<endl;
		  genvtx = (*pt)->production_vertex();
		  ++pt;
	       }
	       if(!genvtx) cout<<"No Gen Vertex!"<<endl;
	    }
	    zcf[i] = genvtx->position().z()*mm;
	    cout<<"z : "<<zcf[i]<<endl;
	    cout<<"---"<<endl;
	 }
      }

      hGenVerticesCF->Fill(zcf[0],zcf[1]);
   }

   // RECO INFO
   if(doRECO_){
      const reco::VertexCollection * recoVertices;
      edm::Handle<reco::VertexCollection> vertexCollection;
      iEvent.getByLabel(vertexSrc_,vertexCollection);
      recoVertices = vertexCollection.product();
      
      int nVertex = recoVertices->size();
      hNrecoVertex->Fill(nVertex);
      if(nVertex > 2) nVertex = 2;
      double z[2] = {-29,-29};
      for (int i = 0 ; i< nVertex; ++i){
	 z[i] = (*recoVertices)[i].position().z();
      }
      
      if(doHIST_){
	 hZrecoVertex0->Fill(z[0]);
	 hZrecoVertex1->Fill(z[1]);
	 hZrecoVertices->Fill(z[0],z[1]);
      }
   }

   t->Fill();

}


void 
HiMixValidation::beginJob()
{

   double histEtaMax = 10;
   int histEtaBins = 200;
   int NcentralityBins = 100;
   int NvtxMax = 20;

   t = f->make<TTree>("mixTree","tree for validation of mixing");


   t->Branch("nmix",&piles.nmix,"nmix/I");
   t->Branch("nbx",&piles.nbx,"nbx/I");

   t->Branch("xVtx",piles.xVtx,"xVtx[nmix]/F");
   t->Branch("yVtx",piles.yVtx,"yVtx[nmix]/F");
   t->Branch("zVtx",piles.zVtx,"zVtx[nmix]/F");

   t->Branch("bx",piles.bx,"bx[nmix]/I");

   t->Branch("event",piles.event,"event[nmix]/l");
   t->Branch("lumi",piles.lumi,"lumi[nmix]/l");
   t->Branch("run",piles.run,"run[nmix]/l");
   t->Branch("file",piles.file,"file[nmix]/l");

   if(doHIST_){

      hGenParticleEtaSignal = f->make<TH1D>("hGenParticleEtaSignal",";#eta;Particles",histEtaBins,-histEtaMax,histEtaMax);
      hGenParticleEtaBkg = f->make<TH1D>("hGenParticleEtaBkg",";#eta;Particles",histEtaBins,-histEtaMax,histEtaMax);
      hGenParticleEtaBkgNpart2 = f->make<TH1D>("hGenParticleEtaBkgNpart2",";#eta;Particles",histEtaBins,-histEtaMax,histEtaMax);
      hGenParticleEtaMixed = f->make<TH1D>("hGenParticleEtaMixed",";#eta;Particles",histEtaBins,-histEtaMax,histEtaMax);
      hGenJetEtaSignal = f->make<TH1D>("hGenJetEtaSignal",";#eta;GenJets",histEtaBins,-histEtaMax,histEtaMax);
      hGenJetEtaBkg = f->make<TH1D>("hGenJetEtaBkg",";#eta;GenJets",histEtaBins,-histEtaMax,histEtaMax);
      hGenJetEtaBkgNpart2 = f->make<TH1D>("hGenJetEtaBkgNpart2",";#eta;GenJets",histEtaBins,-histEtaMax,histEtaMax);
      hGenJetEtaMixed = f->make<TH1D>("hGenJetEtaMixed",";#eta;GenJets",histEtaBins,-histEtaMax,histEtaMax);
      hGenPhotonEtaSignal = f->make<TH1D>("hGenPhotonEtaSignal",";#eta;Photons",histEtaBins,-histEtaMax,histEtaMax);
      hGenPhotonEtaBkg = f->make<TH1D>("hGenPhotonEtaBkg",";#eta;Photons",histEtaBins,-histEtaMax,histEtaMax);
      hGenPhotonEtaBkgNpart2 = f->make<TH1D>("hGenPhotonEtaBkgNpart2",";#eta;Photons",histEtaBins,-histEtaMax,histEtaMax);
      hGenPhotonEtaMixed = f->make<TH1D>("hGenPhotonEtaMixed",";#eta;Photons",histEtaBins,-histEtaMax,histEtaMax);
      hGenMuonEtaSignal = f->make<TH1D>("hGenMuonEtaSignal",";#eta;Muons",histEtaBins,-histEtaMax,histEtaMax);
      hGenMuonEtaBkg = f->make<TH1D>("hGenMuonEtaBkg",";#eta;Muons",histEtaBins,-histEtaMax,histEtaMax);
      hGenMuonEtaBkgNpart2 = f->make<TH1D>("hGenMuonEtaBkgNpart2",";#eta;Muons",histEtaBins,-histEtaMax,histEtaMax);
      hGenMuonEtaMixed = f->make<TH1D>("hGenMuonEtaMixed",";#eta;Muons",histEtaBins,-histEtaMax,histEtaMax);
      hGenElectronEtaSignal = f->make<TH1D>("hGenElectronEtaSignal",";#eta;Electrons",histEtaBins,-histEtaMax,histEtaMax);
      hGenElectronEtaBkg = f->make<TH1D>("hGenElectronEtaBkg",";#eta;Electrons",histEtaBins,-histEtaMax,histEtaMax);
      hGenElectronEtaBkgNpart2 = f->make<TH1D>("hGenElectronEtaBkgNpart2",";#eta;Electrons",histEtaBins,-histEtaMax,histEtaMax);
      hGenElectronEtaMixed = f->make<TH1D>("hGenElectronEtaMixed",";#eta;Electrons",histEtaBins,-histEtaMax,histEtaMax);
      
      hCentralityBin = f->make<TH1D>("hCentralityBin",";Bin;Events",NcentralityBins,0,NcentralityBins);
      hNrecoVertex = f->make<TH1D>("hNrecoVertex",";Number of reconstructed vertices;Events",NvtxMax,0,NvtxMax);
      hZrecoVertex0 = f->make<TH1D>("hZrecoVertex0",";z-position of first vertex [cm];Events",60,-30,30);
      hZrecoVertex1 = f->make<TH1D>("hZrecoVertex1",";z-position of second vertex [cm];Events",60,-30,30);
      
      hZrecoVertices = f->make<TH2D>("hZrecoVertices",";z-position of first vertex;z-position of second vertex;Events",60,-30,30,60,-30,30);
      hNtrackHF = f->make<TH2D>("hNtrackHF",";E_{T}^{HF};N_{tracks};Events",50,0,5000,50,0,5000);
      
      hJetResponseSignal = f->make<TH2D>("hJetResponseSignal","",100,0,200,100,0,200);
      hPhotonResponseSignal = f->make<TH2D>("hPhotonResponseSignal","",100,0,100,100,0,100);
      hMuonResponseSignal = f->make<TH2D>("hMuonResponseSignal","",100,0,20,100,0,20);
      hElectronResponseSignal = f->make<TH2D>("hElectronResponseSignal","",100,0,20,100,0,20);
      
      hJetResponseBkg = f->make<TH2D>("hJetResponseBkg","",100,0,200,100,0,200);
      hPhotonResponseBkg = f->make<TH2D>("hPhotonResponseBkg","",100,0,100,100,0,100);
      hMuonResponseBkg = f->make<TH2D>("hMuonResponseBkg","",100,0,20,100,0,20);
      hElectronResponseBkg = f->make<TH2D>("hElectronResponseBkg","",100,0,20,100,0,20);
      
      hJetResponseMixed = f->make<TH2D>("hJetResponseMixed","",100,0,200,100,0,200);
      hPhotonResponseMixed = f->make<TH2D>("hPhotonResponseMixed","",100,0,100,100,0,100);
      hMuonResponseMixed = f->make<TH2D>("hMuonResponseMixed","",100,0,20,100,0,20);
      hElectronResponseMixed = f->make<TH2D>("hElectronResponseMixed","",100,0,20,100,0,20);
      
      hGenVertices = f->make<TH2D>("hGenVertices","",60,-30,30,60,-30,30);
      hGenVerticesCF = f->make<TH2D>("hGenVerticesCF","",60,-30,30,60,-30,30);
   }


}

void 
HiMixValidation::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
void 
HiMixValidation::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
HiMixValidation::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
HiMixValidation::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
HiMixValidation::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HiMixValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiMixValidation);
