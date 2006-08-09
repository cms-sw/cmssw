/** \class HLTAnalFilt
 *
 * See header file for documentation
 *
 *  $Date: 2006/07/27 08:44:30 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTexample/interface/HLTAnalFilt.h"

#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<typeinfo>

//
// constructors and destructor
//
 
HLTAnalFilt::HLTAnalFilt(const edm::ParameterSet& iConfig)
{
   inputTag_ = iConfig.getParameter< edm::InputTag > ("inputTag");
}

HLTAnalFilt::~HLTAnalFilt()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTAnalFilt::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace reco;

   // get hold of products from Event

   edm::Handle<edm::TriggerResults> tref;
   iEvent.getByType(tref);
   LogDebug("") << "TriggerResults: " << (*tref);

   edm::Handle<HLTFilterObjectWithRefs> ref;
   iEvent.getByLabel(inputTag_,ref);

   HLTParticle particle;
   const Candidate* candidate;

   const unsigned int n(ref->size());
   LogDebug("") << inputTag_.encode() + " Size = " << n;
   for (unsigned int i=0; i!=n; i++) {
     particle=ref->getParticle(i);
     candidate=(ref->getParticleRef(i)).get();
     LogTrace("") << i << " E: " 
               << particle.energy() << " " << candidate->energy() << " "  
		  << typeid(*candidate).name() << " "
                  << particle.eta() << " " << particle.phi() ;
   }

   return;
}
