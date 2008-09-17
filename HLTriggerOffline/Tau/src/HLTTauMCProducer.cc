#include "HLTriggerOffline/Tau/interface/HLTTauMCProducer.h"
#include "TLorentzVector.h"


using namespace edm;
using namespace std;

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
  
  edm::Handle<HepMCProduct> evt;
  iEvent.getByLabel(MC_, evt);
  HepMC::GenEvent*  myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));
  HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
  
  HepMC::GenVertex* TauDecVtx = 0 ;
  
  for (;p != myGenEvent->particles_end(); ++p ) {
    //Check the PDG ID
    bool pdg_ok = false;
    for(size_t pi =0;pi<m_PDG_.size();++pi)
      {
	if(abs((*p)->pdg_id())== m_PDG_[pi]){
	  pdg_ok = true;
	}
      }

    // Check if the boson is one of interest and if there is a valid end_vertex
    if(  pdg_ok  &&
	 (*p)->end_vertex() )
      {
	
	//	TLorentzVector Boson((*p)->momentum().px(),(*p)->momentum().py(),(*p)->momentum().pz(),(*p)->momentum().e());

	HepMC::GenVertex::particle_iterator z = (*p)->end_vertex()->particles_begin(HepMC::descendants);  // Find the beginning of particles after the vertex to loop over there
	for(; z != (*p)->end_vertex()->particles_end(HepMC::descendants); z++)
	  {
	    if(abs((*z)->pdg_id()) == 15 && (*z)->status()==2)  // if it is a Tau
	      {

		LorentzVector Visible_Taus(0.,0.,0.,0.);
		LorentzVector TauDecayProduct(0.,0.,0.,0.);
		
		LorentzVector Neutrino(0.,0.,0.,0.);
		
		vector<HepMC::GenParticle*> TauDaught;
		TauDaught = getGenStableDecayProducts(*z);
		TauDecVtx = (*z)->end_vertex();

		if ( TauDecVtx != 0 )
		  {
		    
		    int numElectrons      = 0;
		    int numMuons          = 0;
		    int numChargedPions   = 0;
		    int numNeutralPions   = 0;
		    int numNeutrinos      = 0;
		    int numOtherParticles = 0;
		    
		    for(vector<HepMC::GenParticle*>::const_iterator pit=TauDaught.begin() ; pit!=TauDaught.end() ; ++pit) 
		      {	      
			int pdg_id = abs((*pit)->pdg_id());
			if (pdg_id == 11) numElectrons++;
			else if (pdg_id == 13) numMuons++;
			else if (pdg_id == 211) numChargedPions++;
			else if (pdg_id == 111) numNeutralPions++;
			else if (pdg_id == 12 || 
				 pdg_id == 14 || 
				 pdg_id == 16) {
			  numNeutrinos++;
			  if (pdg_id == 16) {
			    Neutrino.SetPxPyPzE((*pit)->momentum().px(),(*pit)->momentum().py(),(*pit)->momentum().pz(),(*pit)->momentum().e());
			  }
			}
			else if (pdg_id != 22) {
			  numOtherParticles++;
			}
			
			if (pdg_id != 12 &&
			    pdg_id != 14 && 
			    pdg_id != 16){
			  TauDecayProduct.SetPxPyPzE((*pit)->momentum().px(),(*pit)->momentum().py(),(*pit)->momentum().pz(),(*pit)->momentum().e());
			  Visible_Taus+=TauDecayProduct;
			}	
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
			if ((abs(Visible_Taus.eta()) < etaMax) && (Visible_Taus.pt() > ptMinMCElectron_)){
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
      }
  }
  
  delete myGenEvent;
  
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

std::vector<HepMC::GenParticle*> HLTTauMCProducer::getGenStableDecayProducts(const HepMC::GenParticle* particle)
{
  HepMC::GenVertex* vertex = particle->end_vertex();

  std::vector<HepMC::GenParticle*> decayProducts;
  for ( HepMC::GenVertex::particles_out_const_iterator daughter_particle = vertex->particles_out_const_begin(); 
	daughter_particle != vertex->particles_out_const_end(); ++daughter_particle ){
    int pdg_id = abs((*daughter_particle)->pdg_id());

    // check if particle is stable
    if ( pdg_id == 11 || pdg_id == 12 || pdg_id == 13 || pdg_id == 14 || pdg_id == 16 ||  pdg_id == 111 || pdg_id == 211 ){
      // stable particle, identified by pdg code
      decayProducts.push_back((*daughter_particle));
    } else if ( (*daughter_particle)->end_vertex() != NULL ){
      // unstable particle, identified by non-zero decay vertex

      std::vector<HepMC::GenParticle*> addDecayProducts = getGenStableDecayProducts(*daughter_particle);

      for ( std::vector<HepMC::GenParticle*>::const_iterator adddaughter_particle = addDecayProducts.begin(); adddaughter_particle != addDecayProducts.end(); ++adddaughter_particle ){
	decayProducts.push_back((*adddaughter_particle));
      }
    } else {
      // stable particle, not identified by pdg code
      decayProducts.push_back((*daughter_particle));
    }
  }
   
  return decayProducts;
}
