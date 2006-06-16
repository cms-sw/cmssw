/** \class HLTMakeGlobalObject
 *
 * See header file for documentation
 *
 *  $Date: 2006/05/12 18:13:30 $
 *  $Revision: 1.1 $
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
#include <cassert>

//
// constructors and destructor
//
HLTMakeGlobalObject::HLTMakeGlobalObject(const edm::ParameterSet& iConfig)
{
   labels_ = iConfig.getParameter<std::vector<std::string> >("labels");
   indices_= iConfig.getParameter<std::vector<unsigned int> >("indices");

   std::cout << "HLTMakeGlobalObject: found labels: " << labels_.size() << " " << indices_.size() << std::endl;

   assert(labels_.size()==indices_.size());

   //register your products
   produces<reco::HLTGlobalObject<reco::HLTPathObject<reco::HLTFilterObjectWithRefs> > >();
}

HLTMakeGlobalObject::~HLTMakeGlobalObject()
{
   std::cout << "HLTMakeGlobalObject destroyed! " << std::endl;
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

   cout << "HLTMakeGlobalObject start:" << endl;

   Handle<reco::HLTPathObject<reco::HLTFilterObjectWithRefs> > pathobject;

   auto_ptr<reco::HLTGlobalObject<reco::HLTPathObject<reco::HLTFilterObjectWithRefs> > > globalobject 
       (new reco::HLTGlobalObject<reco::HLTPathObject<reco::HLTFilterObjectWithRefs> >);

   const unsigned int n(labels_.size());
   for (unsigned int i=0; i!=n; i++) {
     try { iEvent.getByLabel(labels_[i],pathobject); }
     catch (...) { continue; }
     globalobject->put(indices_[i],RefProd<reco::HLTPathObject<reco::HLTFilterObjectWithRefs> >(pathobject));
   }

   iEvent.put(globalobject);

   cout << "HLTMakeGlobalObject stop: " << n << endl;

   return;
}
