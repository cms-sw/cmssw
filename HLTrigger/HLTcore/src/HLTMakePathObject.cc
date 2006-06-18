/** \class HLTMakePathObject
 *
 * See header file for documentation
 *
 *  $Date: 2006/06/17 03:37:47 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTMakePathObject.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/HLTReco/interface/HLTPathObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cassert>

//
// constructors and destructor
//
HLTMakePathObject::HLTMakePathObject(const edm::ParameterSet& iConfig)
{
   labels_ = iConfig.getParameter<std::vector<std::string> >("labels");
   indices_= iConfig.getParameter<std::vector<unsigned int> >("indices");

   LogDebug("") << "found labels: " << labels_.size() << " " << indices_.size();

   assert(labels_.size()==indices_.size());

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
   using namespace edm;
   using namespace std;

   auto_ptr<reco::HLTPathObject> pathobject (new reco::HLTPathObject);
   edm::RefToBase<reco::HLTFilterObjectBase> ref;

   Handle<reco::HLTFilterObjectWithRefs> filterobject;

   unsigned int m(0);
   const unsigned int n(labels_.size());
   for (unsigned int i=0; i!=n; i++) {
     iEvent.getByLabel(labels_[i],filterobject);
     ref=edm::RefToBase<reco::HLTFilterObjectBase>(edm::RefProd<reco::HLTFilterObjectWithRefs>(filterobject));
     pathobject->put(ref);
     m++;
   }

   iEvent.put(pathobject);

   LogDebug("") << "Number of filter objects processed: " << m;

   return;
}
