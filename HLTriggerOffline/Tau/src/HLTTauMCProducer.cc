#include "HLTriggerOffline/Tau/interface/HLTTauMCProducer.h"

using namespace edm;
using namespace std;
using namespace reco;

HLTTauMCProducer::HLTTauMCProducer(const edm::ParameterSet& mc)
{
 
  //One Parameter Set per Collection

  MC_      = consumes<GenParticleCollection>(mc.getUntrackedParameter<edm::InputTag>("GenParticles"));
  MCMET_   = consumes<GenMETCollection>(mc.getUntrackedParameter<edm::InputTag>("GenMET"));
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
  produces<LorentzVectorCollection>("MET");
  produces<std::vector<int> >("Mothers");

}

HLTTauMCProducer::~HLTTauMCProducer(){}

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
  auto_ptr<LorentzVectorCollection> product_MET(new LorentzVectorCollection);
  auto_ptr<std::vector<int> > product_Mothers(new std::vector<int>);
  
  edm::Handle<GenParticleCollection> genParticles;
  iEvent.getByToken(MC_, genParticles);

  if(!genParticles.isValid()) return;

  // Look for MET 
  edm::Handle<reco::GenMETCollection> genMet;
  iEvent.getByToken(MCMET_, genMet);
  LorentzVector MET(0.,0.,0.,0.);
  if(genMet.isValid()){
    MET = LorentzVector(
        genMet->front().px(),
        genMet->front().py(),
        0,
        genMet->front().pt()
    );
  }     
  product_MET->push_back(MET);

  // Look for primary bosons
  // It is not guaranteed that primary bosons are stored in event history.
  // Is it really needed when check if taus from the boson is removed? 
  // Kept for backward compatibility
  for(GenParticleCollection::const_iterator p = genParticles->begin(); p != genParticles->end(); ++p) {
    //Check the PDG ID
    bool pdg_ok = false;
    for(size_t pi =0;pi<m_PDG_.size();++pi)
      {
	if(abs((*p).pdgId())== m_PDG_[pi] && ( (*p).isHardProcess() || (*p).status() == 3 ) ){
   	  pdg_ok = true;
	  //cout<<" Bsoson particles: "<< (*p).pdgId()<< " " <<(*p).status() << " "<< pdg_ok<<endl;
	  break;
   	}
      }
    
    // Check if the boson is one of interest and if there is a valid vertex
    if( pdg_ok )
      {
	product_Mothers->push_back((*p).pdgId());

	TLorentzVector Boson((*p).px(),(*p).py(),(*p).pz(),(*p).energy());	
      }
  }// End of search for the bosons

  // Look for taus
  GenParticleRefVector allTaus;
  unsigned index = 0;
  for(GenParticleCollection::const_iterator p = genParticles->begin(); p != genParticles->end(); ++p, ++index) {    
    const GenParticle& genP = *p;
    //accept only isPromptDecayed() particles
    if( !genP.isPromptDecayed() ) continue;
    //check if it is tau, i.e. if |pdgId|=15
    if( std::abs( genP.pdgId() ) == 15 ) {
      GenParticleRef genRef(genParticles, index);
      //check if it is the last tau in decay/radiation chain
      GenParticleRefVector daugTaus;
      getGenDecayProducts(genRef, daugTaus, 0, 15);      
      if( daugTaus.size()==0 )
	allTaus.push_back(genRef);
    }
  }

  // Find stable tau decay products and build visible taus
  for(GenParticleRefVector::const_iterator t = allTaus.begin(); t != allTaus.end(); ++t) {
    //look for all stable (status=1) decay products
    GenParticleRefVector decayProducts;
    getGenDecayProducts(*t,decayProducts,1);

    //build visible taus and recognize decay mode
    if( !decayProducts.empty() ) {
      
      LorentzVector Visible_Taus(0.,0.,0.,0.);
      LorentzVector TauDecayProduct(0.,0.,0.,0.);
      LorentzVector Neutrino(0.,0.,0.,0.);
      
      int numElectrons      = 0;
      int numMuons          = 0;
      int numChargedPions   = 0;
      int numNeutralPions   = 0;
      int numPhotons        = 0;
      int numNeutrinos      = 0;
      int numOtherParticles = 0;
      
      for(GenParticleRefVector::const_iterator pit = decayProducts.begin(); pit != decayProducts.end(); ++pit) {
	int pdg_id = abs((*pit)->pdgId());
	if (pdg_id == 11) numElectrons++;
	else if (pdg_id == 13) numMuons++;
	else if (pdg_id == 211 || pdg_id == 321 ) numChargedPions++; //Count both pi+ and K+
	else if (pdg_id == 111 || pdg_id == 130 || pdg_id == 310) numNeutralPions++; //Count both pi0 and K0_L/S
	else if (pdg_id == 12 || 
		 pdg_id == 14 || 
		 pdg_id == 16) {
	  numNeutrinos++;
	  if (pdg_id == 16) {
	    Neutrino.SetPxPyPzE((*pit)->px(),(*pit)->py(),(*pit)->pz(),(*pit)->energy());
	  }
	}
	else if (pdg_id == 22) numPhotons++;
	else {
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
	    if( numNeutralPions !=0 ){
	      tauDecayMode = kOther;
	      break;
	    }
	    switch ( numPhotons ){
	    case 0:
	      tauDecayMode = kOneProng0pi0;
              break;
	    case 2:
	      tauDecayMode = kOneProng1pi0;
              break;
	    case 4:
	      tauDecayMode = kOneProng2pi0;
              break;
	    default:
	      tauDecayMode = kOther;
	      break;
	    }
	    break;
	  case 3 : 
	    if( numNeutralPions !=0 ){
	      tauDecayMode = kOther;
	      break;
	    }
	    switch ( numPhotons ){
	    case 0 : 
	      tauDecayMode = kThreeProng0pi0;
	      break;
	    case 2 : 
	      tauDecayMode = kThreeProng1pi0;
	      break;
	    default:
	      tauDecayMode = kOther;
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

  iEvent.put(product_Leptons,"LeptonicTauLeptons");
  iEvent.put(product_Electrons,"LeptonicTauElectrons");
  iEvent.put(product_Muons,"LeptonicTauMuons");
  iEvent.put(product_OneProng,"HadronicTauOneProng");
  iEvent.put(product_ThreeProng,"HadronicTauThreeProng");
  iEvent.put(product_OneAndThreeProng,"HadronicTauOneAndThreeProng");
  iEvent.put(product_Other, "TauOther");
  iEvent.put(product_Neutrina,"Neutrina");
  iEvent.put(product_MET,"MET"); 
  iEvent.put(product_Mothers,"Mothers"); 
  
}

// Helper Function

void HLTTauMCProducer::getGenDecayProducts(const GenParticleRef& mother, GenParticleRefVector& products,
					   int status, int pdgId ) {

  const GenParticleRefVector& daughterRefs = mother->daughterRefVector();
  
  for(GenParticleRefVector::const_iterator d = daughterRefs.begin(); d != daughterRefs.end(); ++d) {
    
    if( (status==0 || (*d)->status() == status) && 
	(pdgId==0 || std::abs((*d)->pdgId()) == pdgId) ) {
      
      products.push_back(*d);
    }
    else 
      getGenDecayProducts(*d, products, status, pdgId);
  }
  
} 
