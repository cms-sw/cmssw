#include "HLTriggerOffline/Tau/interface/HLTTauRefInfo.h"
#include "TLorentzVector.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorByIsolation.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "TLorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"



using namespace edm;
using namespace reco;
using namespace std;

HLTTauRefInfo::HLTTauRefInfo(const edm::ParameterSet& iConfig)
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
  ptMinElectron_= electrons.getUntrackedParameter<double>("ptMin",15.);

  ParameterSet  muons = iConfig.getUntrackedParameter<edm::ParameterSet>("Muons");
  Muons_ = muons.getUntrackedParameter<InputTag>("MuonCollection");
  doMuons_ = muons.getUntrackedParameter<bool>("doMuons",false);
  ptMinMuon_= muons.getUntrackedParameter<double>("ptMin",15.);

  ParameterSet  jets = iConfig.getUntrackedParameter<edm::ParameterSet>("Jets");
  Jets_ = jets.getUntrackedParameter<InputTag>("JetCollection");
  doJets_ = jets.getUntrackedParameter<bool>("doJets");
  ptMinJet_= jets.getUntrackedParameter<double>("etMin");

  ParameterSet  mc = iConfig.getUntrackedParameter<edm::ParameterSet>("MC");
  MC_      = mc.getUntrackedParameter<edm::InputTag>("GenParticles");
  doMC_    = mc.getUntrackedParameter<bool>("doMC");
  ptMinMCTau_ = mc.getUntrackedParameter<double>("ptMinTau",15);
  ptMinMCMuon_ = mc.getUntrackedParameter<double>("ptMinMuon",2);
  ptMinMCElectron_ = mc.getUntrackedParameter<double>("ptMinElectron",5);
  m_PDG_   = mc.getUntrackedParameter<int>("BosonID",23);


  etaMax = iConfig.getUntrackedParameter<double>("EtaMax",2.5);
  

  //recoCollections
  produces<LorentzVectorCollection>("PFTaus");
  produces<LorentzVectorCollection>("CaloTaus");
  produces<LorentzVectorCollection>("Electrons");
  produces<LorentzVectorCollection>("Muons");
  produces<LorentzVectorCollection>("Jets");
  //MC Collections
  produces<LorentzVectorCollection>("MCLeptons");
  produces<LorentzVectorCollection>("MCElectrons");
  produces<LorentzVectorCollection>("MCMuons");
  produces<LorentzVectorCollection>("MCTaus");
  produces<LorentzVectorCollection>("MCNeutrina");



}

HLTTauRefInfo::~HLTTauRefInfo(){ }

void HLTTauRefInfo::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{
  if(doPFTaus_)
    doPFTaus(iEvent,iES);
  if(doCaloTaus_)
    doCaloTaus(iEvent,iES);
  if(doElectrons_)
    doElectrons(iEvent,iES);
  if(doMuons_)
    doMuons(iEvent,iES);
  if(doJets_)
    doJets(iEvent,iES);
  if(doMC_)
    doMC(iEvent,iES);

}

void 
HLTTauRefInfo::doPFTaus(edm::Event& iEvent,const edm::EventSetup& iES)
{
      auto_ptr<LorentzVectorCollection> product_PFTaus(new LorentzVectorCollection);
      //Retrieve the collection
      edm::Handle<PFTauCollection> pftaus;
      iEvent.getByLabel(PFTaus_,pftaus);
      edm::Handle<PFTauDiscriminatorByIsolation> pftaudis;
      iEvent.getByLabel(PFTauDis_,pftaudis);
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

void 
HLTTauRefInfo::doCaloTaus(edm::Event& iEvent,const edm::EventSetup& iES)
{
      auto_ptr<LorentzVectorCollection> product_CaloTaus(new LorentzVectorCollection);
      //Retrieve the collection
      edm::Handle<CaloTauCollection> calotaus;
      iEvent.getByLabel(CaloTaus_,calotaus);
      edm::Handle<CaloTauDiscriminatorByIsolation> calotaudis;
      iEvent.getByLabel(CaloTauDis_,calotaudis);
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




void 
HLTTauRefInfo::doElectrons(edm::Event& iEvent,const edm::EventSetup& iES)
{
     auto_ptr<LorentzVectorCollection> product_Electrons(new LorentzVectorCollection);
      //Retrieve the collection
      edm::Handle<PixelMatchGsfElectronCollection> electrons;
      iEvent.getByLabel(Electrons_,electrons);

    
      for(size_t i = 0 ;i<electrons->size();++i)
	{
	 
	    if((*electrons)[i].pt()>ptMinElectron_&&fabs((*electrons)[i].eta())<etaMax)
	      {
		LorentzVector vec((*electrons)[i].px(),(*electrons)[i].py(),(*electrons)[i].pz(),(*electrons)[i].energy());
		product_Electrons->push_back(vec);
	      }
	}


      iEvent.put(product_Electrons,"Electrons");


}

void 
HLTTauRefInfo::doMuons(edm::Event& iEvent,const edm::EventSetup& iES)
{
     auto_ptr<LorentzVectorCollection> product_Muons(new LorentzVectorCollection);
      //Retrieve the collection
      edm::Handle<MuonCollection> muons;
      iEvent.getByLabel(Muons_,muons);

    
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
HLTTauRefInfo::doJets(edm::Event& iEvent,const edm::EventSetup& iES)
{
      auto_ptr<LorentzVectorCollection> product_Jets(new LorentzVectorCollection);
      //Retrieve the collection
      edm::Handle<CaloJetCollection> jets;
      iEvent.getByLabel(Jets_,jets);

    
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
HLTTauRefInfo::doMC(edm::Event& iEvent,const edm::EventSetup& iES)
{

  //All the code from HLTTauMCInfo is here :-) 

  std::vector<LorentzVector> product_Jets_tmp;
  product_Jets_tmp.clear();

  auto_ptr<LorentzVectorCollection> product_Leptons(new LorentzVectorCollection);
  auto_ptr<LorentzVectorCollection> product_Electrons(new LorentzVectorCollection);
  auto_ptr<LorentzVectorCollection> product_Muons(new LorentzVectorCollection);

  auto_ptr<LorentzVectorCollection> product_Jets(new LorentzVectorCollection);
  auto_ptr<LorentzVectorCollection> product_Neutrina(new LorentzVectorCollection);

  edm::Handle<HepMCProduct> evt;
  iEvent.getByLabel(MC_, evt);
  HepMC::GenEvent*  myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));
  HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
  
  
  TLorentzVector taunet; 
  TLorentzVector neutrina_tmp(0.,0.,0.,0.);
  TLorentzVector neutrino_tmp(0.,0.,0.,0.);

  for (;p != myGenEvent->particles_end(); ++p ) {
    if(abs((*p)->pdg_id())==m_PDG_&&(*p)->end_vertex())
      {
	TLorentzVector Boson((*p)->momentum().px(),(*p)->momentum().py(),(*p)->momentum().pz(),(*p)->momentum().e());
	HepMC::GenVertex::particle_iterator z = (*p)->end_vertex()->particles_begin(HepMC::descendants);
	
	for(; z != (*p)->end_vertex()->particles_end(HepMC::descendants); z++)
	  {
	    if(abs((*z)->pdg_id()) == 15 && (*z)->status()==2)
	      {
		bool lept_decay = false;
		TLorentzVector tau((*z)->momentum().px(),(*z)->momentum().py(),(*z)->momentum().pz(),(*z)->momentum().e());
		HepMC::GenVertex::particle_iterator t = (*z)->end_vertex()->particles_begin(HepMC::descendants);
		for(; t != (*z)->end_vertex()->particles_end(HepMC::descendants); t++)
		  {
		    if(abs((*t)->pdg_id()) == 11 || abs((*t)->pdg_id()) == 13)lept_decay=true;
		    if(abs((*t)->pdg_id()) == 11)
		      {
			LorentzVector vec((*t)->momentum().px(),(*t)->momentum().py(),(*t)->momentum().pz(),(*t)->momentum().e());
			product_Leptons->push_back(vec);
			if((abs(vec.eta())<etaMax)&&(vec.pt()>ptMinMCElectron_))
			 product_Electrons->push_back(vec);
		      }
		    if(abs((*t)->pdg_id()) == 13)
		      {
			LorentzVector vec((*t)->momentum().px(),(*t)->momentum().py(),(*t)->momentum().pz(),(*t)->momentum().e());
			product_Leptons->push_back(vec);
			if((abs(vec.eta())<etaMax)&&(vec.pt()>ptMinMCMuon_))
			product_Muons->push_back(vec);
		      }
		    if(abs((*t)->pdg_id()) == 16)taunet.SetPxPyPzE((*t)->momentum().px(),(*t)->momentum().py(),(*t)->momentum().pz(),(*t)->momentum().e());
		    if(abs((*t)->pdg_id()) == 16||abs((*t)->pdg_id()) == 14||abs((*t)->pdg_id()) == 12)
		      {
			neutrino_tmp.SetPxPyPzE((*t)->momentum().px(),(*t)->momentum().py(),(*t)->momentum().pz(),(*t)->momentum().e());
			neutrina_tmp+=neutrino_tmp;
		      }
		    
		  }
	      if(lept_decay==false)
		{
		  TLorentzVector jetMom=tau-taunet;
		  LorentzVector vec(jetMom.Px(),jetMom.Py(),jetMom.Pz(),jetMom.E());
		  if(jetMom.Perp() > ptMinMCTau_ && fabs(jetMom.Eta()) < etaMax)
		    product_Jets_tmp.push_back(vec);
		}
	      
	      }
	  }
      }
  }

  delete myGenEvent;
 for(size_t it =0; it<product_Jets_tmp.size();it++)
   { 
	  product_Jets->push_back(product_Jets_tmp[it]);
   }
    


  LorentzVector neutrina(neutrina_tmp.Px(),neutrina_tmp.Py(),neutrina_tmp.Pz(),neutrina_tmp.E());
  product_Neutrina->push_back(neutrina);
  iEvent.put(product_Leptons,"MCLeptons");
  iEvent.put(product_Electrons,"MCElectrons");
  iEvent.put(product_Muons,"MCMuons");
  iEvent.put(product_Jets,"MCTaus");
  iEvent.put(product_Neutrina,"MCNeutrina");
}
