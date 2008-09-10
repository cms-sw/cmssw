#include "DQMOffline/Trigger/interface/HLTTauRefProducer.h"




using namespace edm;
using namespace reco;
using namespace std;

HLTTauRefProducer::HLTTauRefProducer(const edm::ParameterSet& iConfig)
{

 
  //One Parameter Set per Collection

  ParameterSet pfTau = iConfig.getUntrackedParameter<edm::ParameterSet>("PFTaus");
  PFTaus_ = pfTau.getUntrackedParameter<InputTag>("PFTauProducer");
  PFTauDis_ = pfTau.getUntrackedParameter<InputTag>("PFTauDiscriminator");
  doPFTaus_ = pfTau.getUntrackedParameter<bool>("doPFTaus",false);
  ptMinPFTau_= pfTau.getUntrackedParameter<double>("ptMin",15.);

  ParameterSet  caloTau = iConfig.getUntrackedParameter<edm::ParameterSet>("CaloTaus");
  CaloTaus_ = caloTau.getUntrackedParameter<InputTag>("CaloTauProducer");
  CaloTauDis_ = caloTau.getUntrackedParameter<InputTag>("CaloTauDiscriminator");
  doCaloTaus_ = caloTau.getUntrackedParameter<bool>("doCaloTaus",false);
  ptMinCaloTau_= caloTau.getUntrackedParameter<double>("ptMin",15.);

  ParameterSet  electrons = iConfig.getUntrackedParameter<edm::ParameterSet>("Electrons");
  Electrons_ = electrons.getUntrackedParameter<InputTag>("ElectronCollection");
  doElectrons_ = electrons.getUntrackedParameter<bool>("doElectrons",false);
  e_idAssocProd_ = electrons.getUntrackedParameter<InputTag>("IdCollection");
  e_ctfTrackCollection_= electrons.getUntrackedParameter<InputTag>("TrackCollection");
  ptMinElectron_= electrons.getUntrackedParameter<double>("ptMin",15.);
  e_doID_ = electrons.getUntrackedParameter<bool>("doID",false);
  e_doTrackIso_ = electrons.getUntrackedParameter<bool>("doTrackIso",false);
  e_trackMinPt_= electrons.getUntrackedParameter<double>("ptMinTrack",1.5);
  e_lipCut_= electrons.getUntrackedParameter<double>("lipMinTrack",1.5);
  e_minIsoDR_= electrons.getUntrackedParameter<double>("InnerConeDR",0.02);
  e_maxIsoDR_= electrons.getUntrackedParameter<double>("OuterConeDR",0.6);
  e_isoMaxSumPt_= electrons.getUntrackedParameter<double>("MaxIsoVar",0.02);
  doElecFromZ_=electrons.getUntrackedParameter<bool>("doElecFromZ",false);
  e_zMmin_= electrons.getUntrackedParameter<double>("MinZwindow",70);
  e_zMmax_= electrons.getUntrackedParameter<double>("MaxZwindow",110);
  e_FromZet_ = electrons.getUntrackedParameter<double>("ElecEtFromZcut",15);

  ParameterSet  muons = iConfig.getUntrackedParameter<edm::ParameterSet>("Muons");
  Muons_ = muons.getUntrackedParameter<InputTag>("MuonCollection");
  doMuons_ = muons.getUntrackedParameter<bool>("doMuons",false);
  ptMinMuon_= muons.getUntrackedParameter<double>("ptMin",15.);

  ParameterSet  jets = iConfig.getUntrackedParameter<edm::ParameterSet>("Jets");
  Jets_ = jets.getUntrackedParameter<InputTag>("JetCollection");
  doJets_ = jets.getUntrackedParameter<bool>("doJets");
  ptMinJet_= jets.getUntrackedParameter<double>("etMin");

  etaMax = iConfig.getUntrackedParameter<double>("EtaMax",2.5);
  

  //recoCollections
  produces<LorentzVectorCollection>("PFTaus");
  produces<LorentzVectorCollection>("CaloTaus");
  produces<LorentzVectorCollection>("Electrons");
  produces<LorentzVectorCollection>("ElectronsFromZ");
  produces<LorentzVectorCollection>("Muons");
  produces<LorentzVectorCollection>("Jets");

}

HLTTauRefProducer::~HLTTauRefProducer(){ }

void HLTTauRefProducer::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{
  if(doPFTaus_)
    doPFTaus(iEvent,iES);
  if(doCaloTaus_)
    doCaloTaus(iEvent,iES);
  if(doElectrons_||doElecFromZ_)
    doElectrons(iEvent,iES);
  if(doMuons_)
    doMuons(iEvent,iES);
  if(doJets_)
    doJets(iEvent,iES);

}

void 
HLTTauRefProducer::doPFTaus(edm::Event& iEvent,const edm::EventSetup& iES)
{
      auto_ptr<LorentzVectorCollection> product_PFTaus(new LorentzVectorCollection);
      //Retrieve the collection
      edm::Handle<PFTauCollection> pftaus;
      if(iEvent.getByLabel(PFTaus_,pftaus))
	{
	  edm::Handle<PFTauDiscriminatorByIsolation> pftaudis;
	  if(iEvent.getByLabel(PFTauDis_,pftaudis))
	    for(size_t i = 0 ;i<pftaus->size();++i)
	      {

		if((*pftaudis)[i].second==1)
		  if((*pftaus)[i].pt()>ptMinPFTau_&&fabs((*pftaus)[i].eta())<etaMax)
		    {
		      
		      LorentzVector vec((*pftaus)[i].px(),(*pftaus)[i].py(),(*pftaus)[i].pz(),(*pftaus)[i].energy());
		      product_PFTaus->push_back(vec);
	
		    }
	      }
      iEvent.put(product_PFTaus,"PFTaus");

	}




}

void 
HLTTauRefProducer::doCaloTaus(edm::Event& iEvent,const edm::EventSetup& iES)
{
      auto_ptr<LorentzVectorCollection> product_CaloTaus(new LorentzVectorCollection);
      //Retrieve the collection
      edm::Handle<CaloTauCollection> calotaus;
      if(iEvent.getByLabel(CaloTaus_,calotaus))
	{
	  edm::Handle<CaloTauDiscriminatorByIsolation> calotaudis;
	  if(iEvent.getByLabel(CaloTauDis_,calotaudis))
	    for(size_t i = 0 ;i<calotaus->size();++i)
	      {
		if((*calotaudis)[i].second==1)
		  if((*calotaus)[i].pt()>ptMinCaloTau_&&fabs((*calotaus)[i].eta())<etaMax)
		    {
		      LorentzVector vec((*calotaus)[i].px(),(*calotaus)[i].py(),(*calotaus)[i].pz(),(*calotaus)[i].energy());
		      product_CaloTaus->push_back(vec);
	
		    }
	      }
	


 iEvent.put(product_CaloTaus,"CaloTaus");
	}

     
}

void 
HLTTauRefProducer::doElectrons(edm::Event& iEvent,const edm::EventSetup& iES)
{
  auto_ptr<LorentzVectorCollection> product_Electrons(new LorentzVectorCollection);
  //Retrieve the collections
  edm::Handle<edm::ValueMap<float> > pEleID;
  iEvent.getByLabel(e_idAssocProd_,pEleID);
  if (!pEleID.isValid()){
    edm::LogInfo("")<< "Error! Can't get electronID by label. ";
  }
  
  edm::Handle<reco::TrackCollection> pCtfTracks;
  iEvent.getByLabel(e_ctfTrackCollection_, pCtfTracks);
  if (!pCtfTracks.isValid()) {
    edm::LogInfo("")<< "Error! Can't get " << e_ctfTrackCollection_.label() << " by label. ";
    return;
  }
  const reco::TrackCollection * ctfTracks = pCtfTracks.product();
  edm::Handle<GsfElectronCollection> electrons;
  if(iEvent.getByLabel(Electrons_,electrons))
    for(size_t i=0;i<electrons->size();++i)
      {
	edm::Ref<reco::GsfElectronCollection> electronRef(electrons,i);
	float idDec=0.;
	if(e_doID_){
	  idDec=(*pEleID)[electronRef];
	}else idDec=1.;
	if((*electrons)[i].pt()>ptMinElectron_&&fabs((*electrons)[i].eta())<etaMax&&idDec)
	  {
	    if(e_doTrackIso_){
	      double isolation_value_ele = ElectronTrkIsolation(ctfTracks,(*electrons)[i]);
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
  
  if(doElecFromZ_)
    doElectronsFromZ(iEvent,iES,product_Electrons);
  
  else iEvent.put(product_Electrons,"Electrons");
}

void 
HLTTauRefProducer::doElectronsFromZ(edm::Event& iEvent,const edm::EventSetup& iES,
				    auto_ptr<LorentzVectorCollection>& product_Electrons)
{
  auto_ptr<LorentzVectorCollection> product_ElectronsFromZ(new LorentzVectorCollection);
  if(product_Electrons->size()==2)
    {
      LorentzVector e1=product_Electrons->at(0);
      LorentzVector e2=product_Electrons->at(1);
      if((e1+e2).M()>e_zMmin_ && (e1+e2).M()<e_zMmax_
	 && e1.Et()>e_FromZet_ && e2.Et()>e_FromZet_){
	product_ElectronsFromZ->push_back(e1);
	product_ElectronsFromZ->push_back(e2);
      }
    }
  iEvent.put(product_ElectronsFromZ,"ElectronsFromZ"); 
}


double 
HLTTauRefProducer::ElectronTrkIsolation(const reco::TrackCollection* ctfTracks,const GsfElectron& electron)
{
  reco::TrackCollection::const_iterator tr = ctfTracks->begin();
  double sum_of_pt_ele=0;
  for(;tr != ctfTracks->end();++tr)
    {
      double lip = electron.gsfTrack()->dz() - tr->dz();
      if(tr->pt() > e_trackMinPt_ && fabs(lip) < e_lipCut_){
	double dphi=fabs(tr->phi()-electron.trackMomentumAtVtx().phi());
	if(dphi>acos(-1.))dphi=2*acos(-1.)-dphi;
	double deta=fabs(tr->eta()-electron.trackMomentumAtVtx().eta());
	double dr_ctf_ele = sqrt(deta*deta+dphi*dphi);
	if((dr_ctf_ele>e_minIsoDR_) && (dr_ctf_ele<e_maxIsoDR_)){
	  double cft_pt_2 = (tr->pt())*(tr->pt());
	  sum_of_pt_ele += cft_pt_2;
	}
      }
    }
  double isolation_value_ele = sum_of_pt_ele/(electron.trackMomentumAtVtx().Rho()*electron.trackMomentumAtVtx().Rho());
  return isolation_value_ele;
}

void 
HLTTauRefProducer::doMuons(edm::Event& iEvent,const edm::EventSetup& iES)
{
  auto_ptr<LorentzVectorCollection> product_Muons(new LorentzVectorCollection);
  //Retrieve the collection
  edm::Handle<MuonCollection> muons;
      if(iEvent.getByLabel(Muons_,muons))
   
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
      if(iEvent.getByLabel(Jets_,jets))
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

