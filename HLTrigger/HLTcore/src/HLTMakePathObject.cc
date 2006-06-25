/** \class HLTMakePathObject
 *
 * See header file for documentation
 *
 *  $Date: 2006/06/18 17:44:04 $
 *  $Revision: 1.3 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTMakePathObject.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/HLTReco/interface/HLTPathObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
HLTMakePathObject::HLTMakePathObject(const edm::ParameterSet& iConfig)
{
   inputTags_ = iConfig.getParameter<std::vector<edm::InputTag> >("inputTags");

   LogDebug("") << "found labels: " << inputTags_.size();

   //register your products
   produces<reco::HLTPathObject>();
}

HLTMakePathObject::~HLTMakePathObject()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTMakePathObject::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;

   auto_ptr<HLTPathObject> pathobject (new HLTPathObject);
   RefToBase<HLTFilterObjectBase> ref;

   Handle<HLTFilterObjectWithRefs> filterobject;

   unsigned int m(0);
   const unsigned int n(inputTags_.size());
   for (unsigned int i=0; i!=n; i++) {
     try { iEvent.getByLabel(inputTags_[i],filterobject); }
     catch (...) { continue; }
     ref=RefToBase<HLTFilterObjectBase>(RefProd<HLTFilterObjectWithRefs>(filterobject));
     pathobject->put(ref);
     m++;
   }

   iEvent.put(pathobject);

   LogDebug("") << "Number of filter objects processed: " << m;

   return;
}
