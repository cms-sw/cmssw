/** \class HLTLeptonTauNonCollEMProducer
 *
  * See header file for documentation
 *

 *
 *  \authorK A Petridis
 *
 */

#include "RecoTauTag/HLTProducers/interface/HLTLeptonTauNonCollEMProducer.h"


//
// constructors and destructor
//


HLTLeptonTauNonCollEMProducer::HLTLeptonTauNonCollEMProducer(const edm::ParameterSet& iConfig) :
  jetSrc_  (iConfig.getParameter<vtag> ("JetSrc")),
  leptonTag_ (iConfig.getParameter<edm::InputTag>("LeptonTag")),
  min_dphi_(iConfig.getParameter<double>        ("MinDphi")),
  min_deta_(iConfig.getParameter<double>        ("MinDeta")),
  label_   (iConfig.getParameter<std::string>   ("@module_label"))
  
  
{
  
  //register your products
  produces<reco::EMIsolatedTauTagInfoCollection>("Tau");
  produces<reco::HLTFilterObjectWithRefs>("Lepton");
  produces<reco::CaloJetCollection>("TauJet");
  
  
}

HLTLeptonTauNonCollEMProducer::~HLTLeptonTauNonCollEMProducer()
{
  edm::LogInfo("") << "Destroyed !";
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTLeptonTauNonCollEMProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;

  edm::LogInfo("") << "Start at event="<<iEvent.id().event();

  // Needed to store jets used for triggering.
  
  EMIsolatedTauTagInfoCollection * extendedCollection = new EMIsolatedTauTagInfoCollection;
  auto_ptr<EMIsolatedTauTagInfoCollection> product_Tau (extendedCollection);
  CaloJetCollection * baseCollection = new CaloJetCollection;
  auto_ptr<reco::CaloJetCollection> product_TauTag(baseCollection);
  auto_ptr<HLTFilterObjectWithRefs>
    product_Lepton (new HLTFilterObjectWithRefs);
  
  
  RefToBase<Candidate> ref_lepton;
  Handle<HLTFilterObjectWithRefs> leptonHandle;
  iEvent.getByLabel (leptonTag_,leptonHandle);

  int nTau=0,nLep=0;
  for( vtag::const_iterator s = jetSrc_.begin(); s != jetSrc_.end(); ++ s ) 
    {
      nTau=0;nLep=0;
      std::vector<int> tau_list,lep_list;
           
      Handle<EMIsolatedTauTagInfoCollection> jetsHandle;
      iEvent.getByLabel(*s, jetsHandle);
      EMIsolatedTauTagInfoCollection::const_iterator jet = jetsHandle->begin();
      
      for (; jet != jetsHandle->end(); ++jet) 
	{
	  nLep=0;
	  for(unsigned int i=0; i<leptonHandle->size(); i++)
	    {
	      nLep++;
	      ref_lepton = leptonHandle->getParticleRef(i);
	      double dphi=fabs(leptonHandle->getParticleRef(i).get()->phi()-jet->jet().get()->phi());
	      if(dphi>acos(-1.0))dphi=2*acos(-1.0)-dphi;
	      double deta=fabs(leptonHandle->getParticleRef(i).get()->eta()-jet->jet().get()->eta());
	      if(dphi>min_dphi_||deta>min_deta_)
		{
		  //Check if already in list
		  bool tau_decision=true;
		  for(int i=0;i<tau_list.size();i++)
		    {
		      if(nTau==tau_list[i])tau_decision=false;
		    }
		  
		  //Check if already in list
		  bool em_decision=true;
		  for(int ii=0;ii<lep_list.size();ii++)
		    {
		      if(nLep==lep_list[ii])em_decision=false;
		    }	    
		  
		  //If not in list then add
		  if(tau_decision==true)
		    {
		      extendedCollection->push_back((*jet));
		      const CaloJet* calojet = dynamic_cast<const CaloJet*>((jet->jet().get()));
		      baseCollection->push_back(*calojet);
		    }
		  
		  //If not in list then add
		  if(em_decision==true)
		    {
		      product_Lepton->putParticle(ref_lepton);
		    }
		  
		}
	    }
	}
    }
  
  iEvent.put(product_Tau,"Tau");
  iEvent.put(product_Lepton,"Lepton");
  iEvent.put(product_TauTag,"TauJet");
}
