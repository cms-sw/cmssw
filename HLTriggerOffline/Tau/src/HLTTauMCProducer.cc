#include "HLTriggerOffline/Tau/interface/HLTTauMCProducer.h"
#include "TLorentzVector.h"


using namespace edm;
using namespace std;

HLTTauMCProducer::HLTTauMCProducer(const edm::ParameterSet& mc)
{

 
  //One Parameter Set per Collection

  MC_      = mc.getUntrackedParameter<edm::InputTag>("GenParticles");
  ptMinMCTau_ = mc.getUntrackedParameter<double>("ptMinTau",15);
  ptMinMCMuon_ = mc.getUntrackedParameter<double>("ptMinMuon",2);
  ptMinMCElectron_ = mc.getUntrackedParameter<double>("ptMinElectron",5);
  m_PDG_   = mc.getUntrackedParameter<std::vector<int> >("BosonID");
  etaMax = mc.getUntrackedParameter<double>("EtaMax",2.5);

  produces<LorentzVectorCollection>("Leptons");
  produces<LorentzVectorCollection>("Electrons");
  produces<LorentzVectorCollection>("Muons");
  produces<LorentzVectorCollection>("Taus");
  produces<LorentzVectorCollection>("Neutrina");
}

HLTTauMCProducer::~HLTTauMCProducer(){ }

void HLTTauMCProducer::produce(edm::Event& iEvent, const edm::EventSetup& iES)
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
    //Check the PDG ID
    bool pdg_ok = false;
      for(size_t pi =0;pi<m_PDG_.size();++pi)
	{
	  if(abs((*p)->pdg_id())==m_PDG_[pi])
	  pdg_ok = true;
	}   

    if(pdg_ok&&(*p)->end_vertex())
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
  iEvent.put(product_Leptons,"Leptons");
  iEvent.put(product_Electrons,"Electrons");
  iEvent.put(product_Muons,"Muons");
  iEvent.put(product_Jets,"Taus");
  iEvent.put(product_Neutrina,"Neutrina");


}


