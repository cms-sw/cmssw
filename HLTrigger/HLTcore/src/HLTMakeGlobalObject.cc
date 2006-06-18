/** \class HLTMakeGlobalObject
 *
 * See header file for documentation
 *
 *  $Date: 2006/06/17 03:37:47 $
 *  $Revision: 1.3 $
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

#include <cassert>

//
// constructors and destructor
//
HLTMakeGlobalObject::HLTMakeGlobalObject(const edm::ParameterSet& iConfig)
{
   labels_ = iConfig.getParameter<std::vector<std::string> >("labels");
   indices_= iConfig.getParameter<std::vector<unsigned int> >("indices");

   LogDebug("") << "found labels: " << labels_.size() << " " << indices_.size();
   assert(labels_.size()==indices_.size());

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
   const unsigned int n(labels_.size());
   for (unsigned int i=0; i!=n; i++) {
     try { iEvent.getByLabel(labels_[i],pathobject); }
     catch (...) { continue; }
     m++;
     globalobject->put(RefProd<reco::HLTPathObject>(pathobject));
   }

   iEvent.put(globalobject);

   LogDebug("") << "Number of path objects processed: " << m;

   return;
}
