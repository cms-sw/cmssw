#include "RecoJets/JetPlusTracks/test/JetCollisionFilter.h"

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"

#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
namespace cms
{


JetCollisionFilter::JetCollisionFilter(const edm::ParameterSet& iConfig)
{

 
//   mInputCaloTower = iConfig.getParameter<edm::InputTag>("src0");   
   mInputJets = iConfig.getParameter<edm::InputTag>("src1");


   allowMissingInputs_=iConfig.getUntrackedParameter<bool>("AllowMissingInputs",true);

 
	  
}

JetCollisionFilter::~JetCollisionFilter()
{
    cout<<" JetCollisionFilter destructor "<<endl;
}

void JetCollisionFilter::beginJob()
{
 
}
void JetCollisionFilter::endJob()
{

}

bool JetCollisionFilter::filter(
                                   edm::Event& iEvent,
                                   const edm::EventSetup& theEventSetup
                                )  
{
    cout<<" JetCollisionFilter analyze for Run "<<iEvent.id().run()<<" Event "
    <<iEvent.id().event()<<" Lumi block "<<iEvent.getLuminosityBlock().id().luminosityBlock()<<endl;
    bool retcode = false;
// Check for the proper bunch-crossing (51 or 2724). These numbers may change from run to run

    int bx = iEvent.bunchCrossing();
//    if(bx != 51 && bx != 2724 ) {std::cout<<" Event with bad bunchcrossing? "<<bx<<std::endl;} 

    std::cout<<" Event with tecnical bits 40,41 and bx "<<bx<<std::endl;

// Check if Primary vertex exist

   edm::Handle<reco::VertexCollection> pvHandle; 
   iEvent.getByLabel("offlinePrimaryVertices",pvHandle);
   const reco::VertexCollection & vertices = *pvHandle.product();
   bool result = false;   

   for(reco::VertexCollection::const_iterator it=vertices.begin() ; it!=vertices.end() ; ++it)
   {
      if(it->tracksSize() > 1 && 
         ( fabs(it->z()) <= 15. ) &&
         ( fabs(it->position().rho()) <= 2. )
       ) result = true;
   }

   if(!result) { std::cout<<" Vertex is outside 15 cms "<<std::endl; return retcode;}

   std::cout<<" PV exists in the acceptable range (+-15 cm) and bx = "<<bx<<std::endl;  


/*   
  std::vector<edm::Provenance const*> theProvenance;
  iEvent.getAllProvenance(theProvenance);
  for( std::vector<edm::Provenance const*>::const_iterator ip = theProvenance.begin();
                                                      ip != theProvenance.end(); ip++)
  {
     cout<<" Print all module/label names "<<(**ip).moduleName()<<" "<<(**ip).moduleLabel()<<
     " "<<(**ip).productInstanceName()<<endl;
  }
*/
    

// CaloJets

    edm::Handle<reco::JPTJetCollection> jets0;
    iEvent.getByLabel(mInputJets, jets0);
    if (!jets0.isValid()) {
      // can't find it!
      if (!allowMissingInputs_) {cout<<"CaloTowers are missed "<<endl; 
	*jets0;  // will throw the proper exception
      }
    } else {
      reco::JPTJetCollection::const_iterator jet = jets0->begin ();
      
      cout<<" Size of jets "<<jets0->size()<<endl;
      
      if(jets0->size() > 0 )
	{
	  for (; jet != jets0->end (); jet++)
	    {
	      

		  cout<<" Raw et "<<(*jet).et()<<" Eta "<<(*jet).eta()<<" Phi "<<(*jet).phi()<<endl;
                  if((*jet).et()>10.) retcode = true;

	    } // Calojets cycle
	} // jets collection non-empty
    } // valid collection

    return retcode; 
   
}
} // namespace cms

// define this class as a plugin
#include "FWCore/Framework/interface/MakerMacros.h"
using namespace cms;
DEFINE_FWK_MODULE(JetCollisionFilter);
