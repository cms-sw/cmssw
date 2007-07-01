/** \class HLTLeptonTauNonCollProducer
 *
  * See header file for documentation
 *

 *
 *  \authorK A Petridis
 *
 */

#include "RecoTauTag/HLTProducers/interface/HLTLeptonTauNonCollProducer.h"


//
// constructors and destructor
//


HLTLeptonTauNonCollProducer::HLTLeptonTauNonCollProducer(const edm::ParameterSet& iConfig) :
  jetSrc_  (iConfig.getParameter<vtag> ("JetSrc")),
  leptonTag_ (iConfig.getParameter<edm::InputTag>("LeptonTag")),
  min_dphi_(iConfig.getParameter<double>        ("MinDphi")),
  min_deta_(iConfig.getParameter<double>        ("MinDeta")),
  label_   (iConfig.getParameter<std::string>   ("@module_label"))
  
  
{
  
  //register your products
  produces<reco::IsolatedTauTagInfoCollection>("Tau");
  produces<reco::HLTFilterObjectWithRefs>("Lepton");
  produces<reco::JetTagCollection>("TauTag");
}

HLTLeptonTauNonCollProducer::~HLTLeptonTauNonCollProducer()
{
  edm::LogInfo("") << "Destroyed !";
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTLeptonTauNonCollProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;

  edm::LogInfo("") << "Start at event="<<iEvent.id().event();

  // Needed to store jets used for triggering.
  
  IsolatedTauTagInfoCollection * extendedCollection = new IsolatedTauTagInfoCollection;
  auto_ptr<IsolatedTauTagInfoCollection> product_Tau (extendedCollection);
  auto_ptr<HLTFilterObjectWithRefs>
    product_Lepton (new HLTFilterObjectWithRefs);
  JetTagCollection * baseCollection = new JetTagCollection;
  auto_ptr<reco::JetTagCollection> product_TauTag(baseCollection);  

  
  
  RefToBase<Candidate> ref_lepton;
  Handle<HLTFilterObjectWithRefs> leptonHandle;
  iEvent.getByLabel (leptonTag_,leptonHandle);

  int nTau=0,nLep=0;
  for( vtag::const_iterator s = jetSrc_.begin(); s != jetSrc_.end(); ++ s ) 
    {
      nTau=0;nLep=0;
      std::vector<int> tau_list,lep_list;
           
      Handle<IsolatedTauTagInfoCollection> jetsHandle;
      iEvent.getByLabel(*s, jetsHandle);
      IsolatedTauTagInfoCollection::const_iterator jet = jetsHandle->begin();
      
      for (; jet != jetsHandle->end(); ++jet) 
	{
	  float discriminator = jet->discriminator();
	  JetTracksAssociationRef jetTracks;
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
		      jetTracks = jet->jtaRef();
		      JetTag jetTag(discriminator);
		      baseCollection->push_back(jetTag);
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
  iEvent.put(product_TauTag,"TauTag");
}
