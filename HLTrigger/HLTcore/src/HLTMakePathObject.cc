/** \class HLTMakePathObject
 *
 * See header file for documentation
 *
 *  $Date: 2006/05/12 18:13:30 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTMakePathObject.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/HLTReco/interface/HLTPathObject.h"
#include <cassert>

//
// constructors and destructor
//
HLTMakePathObject::HLTMakePathObject(const edm::ParameterSet& iConfig)
{
   labels_ = iConfig.getParameter<std::vector<std::string> >("labels");
   indices_= iConfig.getParameter<std::vector<unsigned int> >("indices");

   std::cout << "HLTMakePathObject: found labels: " << labels_.size() << " " << indices_.size() << std::endl;

   assert(labels_.size()==indices_.size());

   //register your products
   produces<reco::HLTPathObject>();
}

HLTMakePathObject::~HLTMakePathObject()
{
   std::cout << "HLTMakePathObject destroyed! " << std::endl;
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

   cout << "HLTMakePathObject start:" << endl;

   auto_ptr<reco::HLTPathObject> pathobject (new reco::HLTPathObject);
   edm::RefToBase<reco::HLTFilterObjectBase> ref;

   Handle<reco::HLTFilterObjectWithRefs> filterobject;

   const unsigned int n(labels_.size());
   for (unsigned int i=0; i!=n; i++) {
     iEvent.getByLabel(labels_[i],filterobject);
     ref=edm::RefToBase<reco::HLTFilterObjectBase>(edm::RefProd<reco::HLTFilterObjectWithRefs>(filterobject));
     pathobject->put(ref);
   }

   iEvent.put(pathobject);

   cout << "HLTMakePathObject stop: " << n << endl;

   return;
}
