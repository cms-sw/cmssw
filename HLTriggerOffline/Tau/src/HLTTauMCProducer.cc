#include "HLTriggerOffline/Tau/interface/HLTTauMCProducer.h"

using namespace edm;
using namespace std;
using namespace reco;

HLTTauMCProducer::HLTTauMCProducer(const edm::ParameterSet& mc)
{
 
  //One Parameter Set per Collection

  MC_      = mc.getUntrackedParameter<edm::InputTag>("GenParticles");
  ptMinMCTau_ = mc.getUntrackedParameter<double>("ptMinTau",5.);
  ptMinMCMuon_ = mc.getUntrackedParameter<double>("ptMinMuon",2.);
  ptMinMCElectron_ = mc.getUntrackedParameter<double>("ptMinElectron",5.);
  m_PDG_   = mc.getUntrackedParameter<std::vector<int> >("BosonID");
  etaMax = mc.getUntrackedParameter<double>("EtaMax",2.5);

  produces<LorentzVectorCollection>("LeptonicTauLeptons");
  produces<LorentzVectorCollection>("LeptonicTauElectrons");
  produces<LorentzVectorCollection>("LeptonicTauMuons");
  produces<LorentzVectorCollection>("HadronicTauOneProng");
  produces<LorentzVectorCollection>("HadronicTauThreeProng");
  produces<LorentzVectorCollection>("HadronicTauOneAndThreeProng");
  produces<LorentzVectorCollection>("TauOther");
  produces<LorentzVectorCollection>("Neutrina");

}

HLTTauMCProducer::~HLTTauMCProducer(){ }

void HLTTauMCProducer::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{
  //All the code from HLTTauMCInfo is here :-) 
  
  auto_ptr<LorentzVectorCollection> product_Electrons(new LorentzVectorCollection);
  auto_ptr<LorentzVectorCollection> product_Muons(new LorentzVectorCollection);
  auto_ptr<LorentzVectorCollection> product_Leptons(new LorentzVectorCollection);
  auto_ptr<LorentzVectorCollection> product_OneProng(new LorentzVectorCollection);
  auto_ptr<LorentzVectorCollection> product_ThreeProng(new LorentzVectorCollection);
  auto_ptr<LorentzVectorCollection> product_OneAndThreeProng(new LorentzVectorCollection);
  auto_ptr<LorentzVectorCollection> product_Other(new LorentzVectorCollection);
  auto_ptr<LorentzVectorCollection> product_Neutrina(new LorentzVectorCollection);
  
  edm::Handle<GenParticleCollection> genParticles;
  iEvent.getByLabel(MC_, genParticles);

  GenParticleCollection::const_iterator p = genParticles->begin();
  
  for (;p != genParticles->end(); ++p ) {
    //Check the PDG ID
    bool pdg_ok = true;
    for(size_t pi =0;pi<m_PDG_.size();++pi)
      {
	if(abs((*p).pdgId())== m_PDG_[pi] && abs((*p).status()) == 3 ){
   	  pdg_ok = true;
	  //	  cout<<" Bsoson particles: "<< (*p).pdgId()<< " " <<(*p).status() << " "<< pdg_ok<<endl;
   	}
      }
    
    // Check if the boson is one of interest and if there is a valid vertex
    if(  pdg_ok )
      {
	
	std::vector<GenParticle*> decayProducts;
	
	TLorentzVector Boson((*p).px(),(*p).py(),(*p).pz(),(*p).energy());	
	
	for (GenParticle::const_iterator BosonIt=(*p).begin(); BosonIt != (*p).end(); BosonIt++){
	  // cout<<" Dparticles: "<< (*BosonIt).pdgId() << " "<< (*BosonIt).status()<<endl;
	  
	  if (abs((*BosonIt).pdgId()) == 15 && ((*BosonIt).status()==3)) //if it is a Tau and decayed
	    {
	      decayProducts.clear();	  
	      //	      cout<<" Boson daugther particles: "<< (*BosonIt).pdgId() << " "<< (*BosonIt).status()<< endl;	      
	      for (GenParticle::const_iterator TauIt = (*BosonIt).begin(); TauIt != (*BosonIt).end(); TauIt++) {
		//	cout<<" Tau daughter particles: "<< (*TauIt).pdgId() << " "<< (*TauIt).status()<<endl;

		if (abs((*TauIt).pdgId()) == 15 && ((*TauIt).status()==2)) //if it is a Tau and decayed
		  {		    
		    decayProducts = getGenStableDecayProducts((reco::GenParticle*) & (*TauIt));	 
		    //  for (GenParticle::const_iterator TauIt2 = (*TauIt).begin(); TauIt2 != (*TauIt).end(); TauIt2++) {
		    //		      cout<<" Real Tau particles: "<< (*TauIt2).pdgId() << " "<< (*TauIt2).status()<< " mother: "<< (*TauIt2).mother()->pdgId() << endl;
		    // }
		  }
	      }
	      
	      
	      if ( !decayProducts.empty() )
		{
		  
		  LorentzVector Visible_Taus(0.,0.,0.,0.);
		  LorentzVector TauDecayProduct(0.,0.,0.,0.);
		  LorentzVector Neutrino(0.,0.,0.,0.);
		  
		  int numElectrons      = 0;
		  int numMuons          = 0;
		  int numChargedPions   = 0;
		  int numNeutralPions   = 0;
		  int numNeutrinos      = 0;
		  int numOtherParticles = 0;
		  
		  
		  for (std::vector<GenParticle*>::iterator pit = decayProducts.begin(); pit != decayProducts.end(); pit++)
		    {
		      int pdg_id = abs((*pit)->pdgId());
		      if (pdg_id == 11) numElectrons++;
		      else if (pdg_id == 13) numMuons++;
		      else if (pdg_id == 211) numChargedPions++;
		      else if (pdg_id == 111) numNeutralPions++;
		      else if (pdg_id == 12 || 
			       pdg_id == 14 || 
			       pdg_id == 16) {
			numNeutrinos++;
			if (pdg_id == 16) {
			  Neutrino.SetPxPyPzE((*pit)->px(),(*pit)->py(),(*pit)->pz(),(*pit)->energy());
			}
		      }
		      else if (pdg_id != 22) {
			numOtherParticles++;
		      }
		      
		      if (pdg_id != 12 &&
			  pdg_id != 14 && 
			  pdg_id != 16){
			TauDecayProduct.SetPxPyPzE((*pit)->px(),(*pit)->py(),(*pit)->pz(),(*pit)->energy());
			Visible_Taus+=TauDecayProduct;
		      }	
		      //		  cout<< "This has to be the same: " << (*pit)->pdgId() << " "<< (*pit)->status()<< " mother: "<< (*pit)->mother()->pdgId() << endl;
		    }
		  
		  int tauDecayMode = kOther;
		  
		  if ( numOtherParticles == 0 ){
		    if ( numElectrons == 1 ){
		      //--- tau decays into electrons
		      tauDecayMode = kElectron;
		    } else if ( numMuons == 1 ){
		      //--- tau decays into muons
		      tauDecayMode = kMuon;
		    } else {
		      //--- hadronic tau decays
		      switch ( numChargedPions ){
		      case 1 : 
			switch ( numNeutralPions ){
			case 0 : 
			  tauDecayMode = kOneProng0pi0;
			  break;
			case 1 : 
			  tauDecayMode = kOneProng1pi0;
			  break;
			case 2 : 
			  tauDecayMode = kOneProng2pi0;
			  break;
			}
			break;
		      case 3 : 
			switch ( numNeutralPions ){
			case 0 : 
			  tauDecayMode = kThreeProng0pi0;
			  break;
			case 1 : 
			  tauDecayMode = kThreeProng1pi0;
			  break;
			}
			break;
		      }		    
		    }
		  }
		  
		  
		  //		  cout<< "So we have a: " << tauDecayMode <<endl;
		  
		  if(tauDecayMode == kElectron)
		    {		          
		      if((abs(Visible_Taus.eta())<etaMax)&&(Visible_Taus.pt()>ptMinMCElectron_)){
			product_Electrons->push_back(Visible_Taus);
			product_Leptons->push_back(Visible_Taus);
		      }
		    }
		  else if (tauDecayMode == kMuon)
		    {
		      
		      if((abs(Visible_Taus.eta())<etaMax)&&(Visible_Taus.pt()>ptMinMCMuon_)){
			product_Muons->push_back(Visible_Taus);
			product_Leptons->push_back(Visible_Taus);
		      }
		    }
		  else if(tauDecayMode == kOneProng0pi0 || 
			  tauDecayMode == kOneProng1pi0 || 
			  tauDecayMode == kOneProng2pi0 ) 
		    {
		      if ((abs(Visible_Taus.eta()) < etaMax) && (Visible_Taus.pt() > ptMinMCTau_)){
			product_OneProng->push_back(Visible_Taus);
			product_OneAndThreeProng->push_back(Visible_Taus);
			product_Neutrina->push_back(Neutrino);
		      }
		    }
		  else if (tauDecayMode == kThreeProng0pi0 || 
			   tauDecayMode == kThreeProng1pi0 )
		    {
		      if((abs(Visible_Taus.eta())<etaMax)&&(Visible_Taus.pt()>ptMinMCTau_))  {
			product_ThreeProng->push_back(Visible_Taus);
			product_OneAndThreeProng->push_back(Visible_Taus);
			product_Neutrina->push_back(Neutrino);
		      }								
		    }
		  else if (tauDecayMode == kOther)
		    {
		      if((abs(Visible_Taus.eta())<etaMax)&&(Visible_Taus.pt()>ptMinMCTau_))  {
			product_Other->push_back(Visible_Taus);
		      }
		    }	    				       
		}
	    }
	}			       
	//  
      }
  }
  iEvent.put(product_Leptons,"LeptonicTauLeptons");
  iEvent.put(product_Electrons,"LeptonicTauElectrons");
  iEvent.put(product_Muons,"LeptonicTauMuons");
  iEvent.put(product_OneProng,"HadronicTauOneProng");
  iEvent.put(product_ThreeProng,"HadronicTauThreeProng");
  iEvent.put(product_OneAndThreeProng,"HadronicTauOneAndThreeProng");
  iEvent.put(product_Other, "TauOther");
  iEvent.put(product_Neutrina,"Neutrina"); 
  
  						       
}
// Helper Function

std::vector<reco::GenParticle*> HLTTauMCProducer::getGenStableDecayProducts(const reco::GenParticle* particle)
{
  std::vector<GenParticle*> decayProducts;
  decayProducts.clear();

  //  std::cout << " Are we ever here?: "<< (*particle).numberOfDaughters() << std::endl;
  for ( GenParticle::const_iterator daughter_particle = (*particle).begin();daughter_particle != (*particle).end(); ++daughter_particle ){   

    int pdg_id = abs((*daughter_particle).pdgId());

//    // check if particle is stable
    if ( pdg_id == 11 || pdg_id == 12 || pdg_id == 13 || pdg_id == 14 || pdg_id == 16 ||  pdg_id == 111 || pdg_id == 211 ){
      // stable particle, identified by pdg code
      decayProducts.push_back((reco::GenParticle*) &(* daughter_particle));
    } 
    else {
//      // unstable particle, identified by non-zero decay vertex
      std::vector<GenParticle*> addDecayProducts = getGenStableDecayProducts((reco::GenParticle*) &(*daughter_particle));
      for ( std::vector<GenParticle*>::const_iterator adddaughter_particle = addDecayProducts.begin(); adddaughter_particle != addDecayProducts.end(); ++adddaughter_particle ){
	decayProducts.push_back((*adddaughter_particle));
      }
    }
  }
  return decayProducts;
}
