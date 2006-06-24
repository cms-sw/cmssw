/** \class HLTAnalCand
 *
 * See header file for documentation
 *
 *  $Date: 2006/06/18 17:44:04 $
 *  $Revision: 1.6 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTAnalCand.h"

#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<typeinfo>

//
// constructors and destructor
//
 
HLTAnalCand::HLTAnalCand(const edm::ParameterSet& iConfig)
{
   inputTag_ = iConfig.getParameter< edm::InputTag > ("inputTag");
}

HLTAnalCand::~HLTAnalCand()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTAnalCand::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace reco;

   // get hold of products from Event

   edm::Handle<HLTFilterObjectWithRefs> ref;
   iEvent.getByLabel(inputTag_,ref);

   HLTParticle particle;
   const Candidate* candidate;

   const unsigned int n(ref->numberParticles());
   LogDebug("") << inputTag_.encode() + " Size = " << n;
   for (unsigned int i=0; i!=n; i++) {
     particle=ref->getParticle(i);
     candidate=(ref->getParticleRef(i)).get();
     LogTrace("") << i << " E: " 
               << particle.energy() << " " << candidate->energy() << " "  
               << typeid(*candidate).name();
   }

   return;
}
