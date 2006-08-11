/** \class HLTAnalFilt
 *
 * See header file for documentation
 *
 *  $Date: 2006/08/10 17:10:51 $
 *  $Revision: 1.4 $
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
 
HLTAnalFilt::HLTAnalFilt(const edm::ParameterSet& iConfig) :
  inputTag_(iConfig.getParameter<edm::InputTag>("inputTag"))
{
  LogDebug("") << "Input: " << inputTag_.encode();
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
   using namespace edm;
   using namespace reco;

   // get hold of (single?) TriggerResults object
   vector<Handle<TriggerResults> > trhv;
   iEvent.getManyByType(trhv);
   const unsigned int n(trhv->size());
   LogDebug("") << "Number of TriggerResults objects found: " << n;
   for (unsigned int i=0; i!=n; i++) {
     LogDebug("") << "TriggerResult object " << i << " bits: " << (*trhv)[i];
   }

   // get hold of requested filter object
   Handle<HLTFilterObjectWithRefs> ref;
   try {iEvent.getByLabel(inputTag_,ref);} catch(...) {;}
   if (ref.isValid()) {
     const unsigned int n(ref->size());
     LogDebug("") << inputTag_.encode() + " Size = " << n;
     for (unsigned int i=0; i!=n; i++) {
       // some Xchecks
       const Candidate* candidate;
       HLTParticle particle(ref->getParticle(i));
       const Candidate* candidate((ref->getParticleRef(i)).get());
       LogTrace("") << i << " E: " 
		    << particle.energy() << " " << candidate->energy() << " "  
		    << typeid(*candidate).name() << " "
		    << particle.eta() << " " << particle.phi() ;
     }
   } else {
     LogDebug("") << "Filterobject " + inputTag_.encode() + " not found!";
   }

   return;
}
