/** \class HLTMakeGlobalObject
 *
 * See header file for documentation
 *
 *  $Date: 2006/06/18 17:44:04 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTMakeGlobalObject.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/HLTReco/interface/HLTGlobalObject.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/HLTReco/interface/HLTPathObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
HLTMakeGlobalObject::HLTMakeGlobalObject(const edm::ParameterSet& iConfig)
{
   inputTags_ = iConfig.getParameter<std::vector<edm::InputTag> >("inputTags");

   LogDebug("") << "found labels: " << inputTags_.size();

   //register your products
   produces<reco::HLTGlobalObject>();
}

HLTMakeGlobalObject::~HLTMakeGlobalObject()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTMakeGlobalObject::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   auto_ptr<reco::HLTGlobalObject> globalobject (new reco::HLTGlobalObject);

   Handle<reco::HLTPathObject> pathobject;

   unsigned int m(0);
   const unsigned int n(inputTags_.size());
   for (unsigned int i=0; i!=n; i++) {
     try { iEvent.getByLabel(inputTags_[i],pathobject); }
     catch (...) { continue; }
     m++;
     globalobject->put(RefProd<reco::HLTPathObject>(pathobject));
   }

   iEvent.put(globalobject);

   LogDebug("") << "Number of path objects processed: " << m;

   return;
}
