#include "HLTriggerOffline/Tau/interface/HLTTauMcInfo.h"
#include "TLorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

using namespace edm;
using namespace reco;
using namespace HepMC;
using namespace std;

HLTTauMcInfo::HLTTauMcInfo(const edm::ParameterSet& iConfig)
{
  genParticles = iConfig.getParameter<InputTag>("GenParticles");
  pfTauCollection_ = iConfig.getParameter<InputTag>("PFTauProducer");
  pfTauDiscriminatorProd_ = iConfig.getParameter<InputTag>("PFTauDiscriminator");
  usePFTauMatching_ = iConfig.getParameter<bool>("UsePFTauMatching");
  m_PDG = iConfig.getParameter<int>("BosonPID");
  etaMax = iConfig.getParameter<double>("EtaMax");
  ptMin = iConfig.getParameter<double>("PtMin");
  produces<LorentzVectorCollection>("Leptons");
  produces<LorentzVectorCollection>("Electrons");
  produces<LorentzVectorCollection>("Muons");
  produces<LorentzVectorCollection>("Jets");
  produces<LorentzVectorCollection>("Neutrina");
}

HLTTauMcInfo::~HLTTauMcInfo(){ }

void HLTTauMcInfo::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{
  std::vector<LorentzVector> product_Jets_tmp;
  product_Jets_tmp.clear();

  auto_ptr<LorentzVectorCollection> product_Leptons(new LorentzVectorCollection);
  auto_ptr<LorentzVectorCollection> product_Electrons(new LorentzVectorCollection);
  auto_ptr<LorentzVectorCollection> product_Muons(new LorentzVectorCollection);

  auto_ptr<LorentzVectorCollection> product_Jets(new LorentzVectorCollection);
  auto_ptr<LorentzVectorCollection> product_Neutrina(new LorentzVectorCollection);

  edm::Handle<HepMCProduct> evt;
  iEvent.getByLabel(genParticles, evt);
  HepMC::GenEvent*  myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));
  HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
  
  
  TLorentzVector taunet; 
  TLorentzVector neutrina_tmp(0.,0.,0.,0.);
  TLorentzVector neutrino_tmp(0.,0.,0.,0.);

  for (;p != myGenEvent->particles_end(); ++p ) {
    if(abs((*p)->pdg_id())==m_PDG&&(*p)->end_vertex())
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
			if((abs(vec.eta())<etaMax)&&(vec.pt()>5))
			 product_Electrons->push_back(vec);
		      }
		    if(abs((*t)->pdg_id()) == 13)
		      {
			LorentzVector vec((*t)->momentum().px(),(*t)->momentum().py(),(*t)->momentum().pz(),(*t)->momentum().e());
			product_Leptons->push_back(vec);
			if((abs(vec.eta())<etaMax)&&(vec.pt()>2))
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
		  if(jetMom.Perp() > ptMin && fabs(jetMom.Eta()) < etaMax)
		    product_Jets_tmp.push_back(vec);
		}
	      
	      }
	  }
      }
  }

  delete myGenEvent;
  if(usePFTauMatching_)
    {
      Handle<PFTauCollection> thePFTauHandle;
      iEvent.getByLabel(pfTauCollection_,thePFTauHandle);
  
      Handle<PFTauDiscriminatorByIsolation> thePFTauDiscriminatorByIsolation;
      iEvent.getByLabel(pfTauDiscriminatorProd_,thePFTauDiscriminatorByIsolation);

      for(int it =0; it<product_Jets_tmp.size();it++)
	{ 
	  for (PFTauCollection::size_type iPFTau=0;iPFTau<thePFTauHandle->size();iPFTau++) {
	    PFTauRef thePFTau(thePFTauHandle,iPFTau);
	    if((*thePFTauDiscriminatorByIsolation)[thePFTau] ==1)
	      {
		LorentzVector lvPFTau=(*thePFTau).p4();
		if(deltaR(lvPFTau,product_Jets_tmp[it]) < 0.3)
		  product_Jets->push_back(product_Jets_tmp[it]);
	      }
	  }
	}
      
    }else{
      for(int it =0; it<product_Jets_tmp.size();it++)
	{ 
	  product_Jets->push_back(product_Jets_tmp[it]);
	}
    }




  LorentzVector neutrina(neutrina_tmp.Px(),neutrina_tmp.Py(),neutrina_tmp.Pz(),neutrina_tmp.E());
  product_Neutrina->push_back(neutrina);
  iEvent.put(product_Leptons,"Leptons");
  iEvent.put(product_Electrons,"Electrons");
  iEvent.put(product_Muons,"Muons");
  iEvent.put(product_Jets,"Jets");
  iEvent.put(product_Neutrina,"Neutrina");
}
