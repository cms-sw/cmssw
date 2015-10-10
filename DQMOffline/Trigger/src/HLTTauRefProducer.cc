#include "DQMOffline/Trigger/interface/HLTTauRefProducer.h"
#include "TLorentzVector.h"
// TAU includes
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorByIsolation.h"
// ELECTRON includes
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDAssociation.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
// MUON includes
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "TLorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
//CaloTower includes
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
#include "Math/GenVector/VectorUtil.h"


using namespace edm;
using namespace reco;
using namespace std;

HLTTauRefProducer::HLTTauRefProducer(const edm::ParameterSet& iConfig)
{

 
  //One Parameter Set per Collection

  ParameterSet pfTau = iConfig.getUntrackedParameter<edm::ParameterSet>("PFTaus");
  PFTaus_ = consumes<reco::PFTauCollection>(pfTau.getUntrackedParameter<InputTag>("PFTauProducer"));
  auto discs = pfTau.getUntrackedParameter<std::vector<InputTag> >("PFTauDiscriminators");
  for(edm::InputTag& tag: discs) {
    PFTauDis_.push_back(consumes<reco::PFTauDiscriminator>(tag));
  }
  doPFTaus_ = pfTau.getUntrackedParameter<bool>("doPFTaus",false);
  ptMinPFTau_= pfTau.getUntrackedParameter<double>("ptMin",15.);

  ParameterSet  electrons = iConfig.getUntrackedParameter<edm::ParameterSet>("Electrons");
  Electrons_ = consumes<reco::GsfElectronCollection>(electrons.getUntrackedParameter<InputTag>("ElectronCollection"));
  doElectrons_ = electrons.getUntrackedParameter<bool>("doElectrons",false);
  e_doID_ = electrons.getUntrackedParameter<bool>("doID",false);
  if(e_doID_) {
    e_idAssocProd_ = consumes<reco::ElectronIDAssociationCollection>(electrons.getUntrackedParameter<InputTag>("IdCollection"));
  }
  e_ctfTrackCollectionSrc_ = electrons.getUntrackedParameter<InputTag>("TrackCollection");
  e_ctfTrackCollection_ = consumes<reco::TrackCollection>(e_ctfTrackCollectionSrc_);
  ptMinElectron_= electrons.getUntrackedParameter<double>("ptMin",15.);
  e_doTrackIso_ = electrons.getUntrackedParameter<bool>("doTrackIso",false);
  e_trackMinPt_= electrons.getUntrackedParameter<double>("ptMinTrack",1.5);
  e_lipCut_= electrons.getUntrackedParameter<double>("lipMinTrack",1.5);
  e_minIsoDR_= electrons.getUntrackedParameter<double>("InnerConeDR",0.02);
  e_maxIsoDR_= electrons.getUntrackedParameter<double>("OuterConeDR",0.6);
  e_isoMaxSumPt_= electrons.getUntrackedParameter<double>("MaxIsoVar",0.02);

  ParameterSet  muons = iConfig.getUntrackedParameter<edm::ParameterSet>("Muons");
  Muons_ = consumes<reco::MuonCollection>(muons.getUntrackedParameter<InputTag>("MuonCollection"));
  doMuons_ = muons.getUntrackedParameter<bool>("doMuons",false);
  ptMinMuon_= muons.getUntrackedParameter<double>("ptMin",15.);

  ParameterSet  jets = iConfig.getUntrackedParameter<edm::ParameterSet>("Jets");
  Jets_ = consumes<reco::CaloJetCollection>(jets.getUntrackedParameter<InputTag>("JetCollection"));
  doJets_ = jets.getUntrackedParameter<bool>("doJets");
  ptMinJet_= jets.getUntrackedParameter<double>("etMin");

  ParameterSet  towers = iConfig.getUntrackedParameter<edm::ParameterSet>("Towers");
  Towers_ = consumes<CaloTowerCollection>(towers.getUntrackedParameter<InputTag>("TowerCollection"));
  doTowers_ = towers.getUntrackedParameter<bool>("doTowers");
  ptMinTower_= towers.getUntrackedParameter<double>("etMin");
  towerIsol_= towers.getUntrackedParameter<double>("towerIsolation");

  ParameterSet  photons = iConfig.getUntrackedParameter<edm::ParameterSet>("Photons");
  Photons_ = consumes<reco::PhotonCollection>(photons.getUntrackedParameter<InputTag>("PhotonCollection"));
  doPhotons_ = photons.getUntrackedParameter<bool>("doPhotons");
  ptMinPhoton_= photons.getUntrackedParameter<double>("etMin");
  photonEcalIso_= photons.getUntrackedParameter<double>("ECALIso");

  ParameterSet  met = iConfig.getUntrackedParameter<edm::ParameterSet>("MET");
  MET_ = consumes<reco::CaloMETCollection>(met.getUntrackedParameter<InputTag>("METCollection"));
  doMET_ = met.getUntrackedParameter<bool>("doMET",false);
  ptMinMET_= met.getUntrackedParameter<double>("ptMin",15.);


  etaMax = iConfig.getUntrackedParameter<double>("EtaMax",2.5);
  

  //recoCollections
  produces<LorentzVectorCollection>("PFTaus");
  produces<LorentzVectorCollection>("Electrons");
  produces<LorentzVectorCollection>("Muons");
  produces<LorentzVectorCollection>("Jets");
  produces<LorentzVectorCollection>("Photons");
  produces<LorentzVectorCollection>("Towers");
  produces<LorentzVectorCollection>("MET");

}

HLTTauRefProducer::~HLTTauRefProducer(){ }

void HLTTauRefProducer::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{
  if(doPFTaus_)
    doPFTaus(iEvent,iES);
  if(doElectrons_)
    doElectrons(iEvent,iES);
  if(doMuons_)
    doMuons(iEvent,iES);
  if(doJets_)
    doJets(iEvent,iES);
  if(doPhotons_)
    doPhotons(iEvent,iES);
  if(doTowers_)
    doTowers(iEvent,iES);
  if(doMET_)
    doMET(iEvent,iES);
}

void 
HLTTauRefProducer::doPFTaus(edm::Event& iEvent,const edm::EventSetup& iES)
{
      auto_ptr<LorentzVectorCollection> product_PFTaus(new LorentzVectorCollection);
      //Retrieve the collection
      edm::Handle<PFTauCollection> pftaus;
      if(iEvent.getByToken(PFTaus_,pftaus)) {
        for(unsigned int i=0; i<pftaus->size(); ++i) {
	    if((*pftaus)[i].pt()>ptMinPFTau_&&fabs((*pftaus)[i].eta())<etaMax)
	      {
		reco::PFTauRef thePFTau(pftaus,i);
                bool passAll = true;
                for(edm::EDGetTokenT<reco::PFTauDiscriminator>& token: PFTauDis_) {
		  edm::Handle<reco::PFTauDiscriminator> pftaudis;
		  if(iEvent.getByToken(token, pftaudis)) {
                    if((*pftaudis)[thePFTau] < 0.5) {
                      passAll = false;
                      break;
                    }
                  }
                }
                if(passAll) {
                  LorentzVector vec((*pftaus)[i].px(),(*pftaus)[i].py(),(*pftaus)[i].pz(),(*pftaus)[i].energy());
                  product_PFTaus->push_back(vec);
                }
              }
        }
      }
      iEvent.put(product_PFTaus,"PFTaus");
}


void 
HLTTauRefProducer::doElectrons(edm::Event& iEvent,const edm::EventSetup& iES)
{
  auto_ptr<LorentzVectorCollection> product_Electrons(new LorentzVectorCollection);
  //Retrieve the collections
  
  edm::Handle<reco::ElectronIDAssociationCollection> pEleID;
  if(e_doID_){//UGLY HACK UNTIL GET ELETRON ID WORKING IN 210
   
      iEvent.getByToken(e_idAssocProd_, pEleID);
    
    if (!pEleID.isValid()){
      edm::LogInfo("")<< "Error! Can't get electronIDAssocProducer by label. ";
      e_doID_ = false;
    }
  }
  edm::Handle<reco::TrackCollection> pCtfTracks;
  iEvent.getByToken(e_ctfTrackCollection_, pCtfTracks);
  if (!pCtfTracks.isValid()) {
  edm::LogInfo("")<< "Error! Can't get " << e_ctfTrackCollectionSrc_.label() << " by label. ";
  iEvent.put(product_Electrons,"Electrons"); 
  return;
  }
  const reco::TrackCollection * ctfTracks = pCtfTracks.product();
  edm::Handle<GsfElectronCollection> electrons;
  if(iEvent.getByToken(Electrons_,electrons))
    for(size_t i=0;i<electrons->size();++i)
      {
	edm::Ref<reco::GsfElectronCollection> electronRef(electrons,i);
	bool idDec=false;
	if(e_doID_){
	  reco::ElectronIDAssociationCollection::const_iterator tagIDAssocItr;
	  tagIDAssocItr = pEleID->find(electronRef);
	  const reco::ElectronIDRef& id_tag = tagIDAssocItr->val;
	  idDec=id_tag->cutBasedDecision();
	}else idDec=true;
	if((*electrons)[i].pt()>ptMinElectron_&&fabs((*electrons)[i].eta())<etaMax&&idDec)
	  {
	    if(e_doTrackIso_){
	      reco::TrackCollection::const_iterator tr = ctfTracks->begin();
	      double sum_of_pt_ele=0;
	      for(;tr != ctfTracks->end();++tr)
		{
		  double lip = (*electrons)[i].gsfTrack()->dz() - tr->dz();
		  if(tr->pt() > e_trackMinPt_ && fabs(lip) < e_lipCut_){
		    double dphi=fabs(tr->phi()-(*electrons)[i].trackMomentumAtVtx().phi());
		    if(dphi>acos(-1.))dphi=2*acos(-1.)-dphi;
		    double deta=fabs(tr->eta()-(*electrons)[i].trackMomentumAtVtx().eta());
		    double dr_ctf_ele = sqrt(deta*deta+dphi*dphi);
		    if((dr_ctf_ele>e_minIsoDR_) && (dr_ctf_ele<e_maxIsoDR_)){
		      double cft_pt_2 = (tr->pt())*(tr->pt());
		      sum_of_pt_ele += cft_pt_2;
		    }
		  }
		}
	      double isolation_value_ele = sum_of_pt_ele/((*electrons)[i].trackMomentumAtVtx().Rho()*(*electrons)[i].trackMomentumAtVtx().Rho());
	      if(isolation_value_ele<e_isoMaxSumPt_){
		LorentzVector vec((*electrons)[i].px(),(*electrons)[i].py(),(*electrons)[i].pz(),(*electrons)[i].energy());
		product_Electrons->push_back(vec);
	      } 
	      
	    }
	    else{ 
	      LorentzVector vec((*electrons)[i].px(),(*electrons)[i].py(),(*electrons)[i].pz(),(*electrons)[i].energy());
	      product_Electrons->push_back(vec);
	    }
	  }
      }
  
  iEvent.put(product_Electrons,"Electrons"); 
}

void 
HLTTauRefProducer::doMuons(edm::Event& iEvent,const edm::EventSetup& iES)
{
  auto_ptr<LorentzVectorCollection> product_Muons(new LorentzVectorCollection);
  //Retrieve the collection
  edm::Handle<MuonCollection> muons;
      if(iEvent.getByToken(Muons_,muons))
   
      for(size_t i = 0 ;i<muons->size();++i)
	{
	 
	    if((*muons)[i].pt()>ptMinMuon_&&fabs((*muons)[i].eta())<etaMax)
	      {
		LorentzVector vec((*muons)[i].px(),(*muons)[i].py(),(*muons)[i].pz(),(*muons)[i].energy());
		product_Muons->push_back(vec);
	      }
	}


      iEvent.put(product_Muons,"Muons");
 
}


void 
HLTTauRefProducer::doJets(edm::Event& iEvent,const edm::EventSetup& iES)
{
      auto_ptr<LorentzVectorCollection> product_Jets(new LorentzVectorCollection);
      //Retrieve the collection
      edm::Handle<CaloJetCollection> jets;
      if(iEvent.getByToken(Jets_,jets))
      for(size_t i = 0 ;i<jets->size();++i)
	{
	     if((*jets)[i].et()>ptMinJet_&&fabs((*jets)[i].eta())<etaMax)
	      {
		LorentzVector vec((*jets)[i].px(),(*jets)[i].py(),(*jets)[i].pz(),(*jets)[i].energy());
		product_Jets->push_back(vec);
	      }
	}
      iEvent.put(product_Jets,"Jets");
}

void 
HLTTauRefProducer::doTowers(edm::Event& iEvent,const edm::EventSetup& iES)
{
      auto_ptr<LorentzVectorCollection> product_Towers(new LorentzVectorCollection);
      //Retrieve the collection
      edm::Handle<CaloTowerCollection> towers;
      if(iEvent.getByToken(Towers_,towers))
      for(size_t i = 0 ;i<towers->size();++i)
	{
	     if((*towers)[i].pt()>ptMinTower_&&fabs((*towers)[i].eta())<etaMax)
	      {
		//calculate isolation
		double isolET=0;
		for(unsigned int j=0;j<towers->size();++j)
		  {
		    if(ROOT::Math::VectorUtil::DeltaR((*towers)[i].p4(),(*towers)[j].p4())<0.5)
		      isolET+=(*towers)[j].pt();
		  }
		isolET-=(*towers)[i].pt();
		if(isolET<towerIsol_)
		  {
		    LorentzVector vec((*towers)[i].px(),(*towers)[i].py(),(*towers)[i].pz(),(*towers)[i].energy());
		    product_Towers->push_back(vec);
		  }
	      }
	}
      iEvent.put(product_Towers,"Towers");
}


void 
HLTTauRefProducer::doPhotons(edm::Event& iEvent,const edm::EventSetup& iES)
{
      auto_ptr<LorentzVectorCollection> product_Gammas(new LorentzVectorCollection);
      //Retrieve the collection
      edm::Handle<PhotonCollection> photons;
      if(iEvent.getByToken(Photons_,photons))
      for(size_t i = 0 ;i<photons->size();++i)
	if((*photons)[i].ecalRecHitSumEtConeDR04()<photonEcalIso_)
	{
	     if((*photons)[i].et()>ptMinPhoton_&&fabs((*photons)[i].eta())<etaMax)
	      {
		LorentzVector vec((*photons)[i].px(),(*photons)[i].py(),(*photons)[i].pz(),(*photons)[i].energy());
		product_Gammas->push_back(vec);
	      }
	}
      iEvent.put(product_Gammas,"Photons");
}

void
HLTTauRefProducer::doMET(edm::Event& iEvent,const edm::EventSetup& iES)
{
  auto_ptr<LorentzVectorCollection> product_MET(new LorentzVectorCollection);
  //Retrieve the collection
  edm::Handle<reco::CaloMETCollection> met;
  if(iEvent.getByToken(MET_,met)){
    double px = met->front().p4().Px();
    double py = met->front().p4().Py();
    double pt = met->front().p4().Pt();
    LorentzVector vec(px,py,0,pt);
    product_MET->push_back(vec);
  }
  iEvent.put(product_MET,"MET");  
}

