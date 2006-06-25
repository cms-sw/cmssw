/** \class HLTMakeSummaryObjects
 *
 * See header file for documentation
 *
 *  $Date: 2006/06/25 19:03:02 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTMakeSummaryObjects.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/HLTReco/interface/HLTGlobalObject.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/HLTReco/interface/HLTPathObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
HLTMakeSummaryObjects::HLTMakeSummaryObjects(const edm::ParameterSet& iConfig)
{
   nTrig_ = iConfig.getParameter< unsigned int > ("nTrig");

   //register your products
   produces<reco::HLTPathObject  >();
   produces<reco::HLTGlobalObject>();
}

HLTMakeSummaryObjects::~HLTMakeSummaryObjects()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTMakeSummaryObjects::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;


   // get all possible types of filter objects

   vector<Handle<HLTFilterObjectBase    > > fob0;
   vector<Handle<HLTFilterObject        > > fob1;
   vector<Handle<HLTFilterObjectWithRefs> > fob2;

   iEvent.getManyByType(fob0);
   iEvent.getManyByType(fob1);
   iEvent.getManyByType(fob2);

   const unsigned int n0(fob0.size());
   const unsigned int n1(fob1.size());
   const unsigned int n2(fob2.size());

   LogDebug("") << "Number of filter objects found: " << n0 << " " << n1 << " " << n2;


   // for subsequent use, store in a vector RefToBases to all filter objects found

   const unsigned int n(n0+n1+n2);
   vector<RefToBase<HLTFilterObjectBase> > fobs(n);
   unsigned int i(0);
   for (unsigned int i0=0; i0!=n0; i0++) {
     fobs[i]=RefToBase<HLTFilterObjectBase>(RefProd<HLTFilterObjectBase>(fob0[i0]));
     i++;
   }
   for (unsigned int i1=0; i1!=n1; i1++) {
     fobs[i]=RefToBase<HLTFilterObjectBase>(RefProd<HLTFilterObject    >(fob1[i1]));
     i++;
   }
   for (unsigned int i2=0; i2!=n2; i2++) {
     fobs[i]=RefToBase<HLTFilterObjectBase>(RefProd<HLTFilterObjectWithRefs>(fob2[i2]));
     i++;
   }


   // construct the path objects and insert them in the Event
   // currently we construct and insert "empty" path objects for paths
   // for which there is not filter object found!

   for (unsigned int p=0; p!=nTrig_; p++) {

     // order within path according to module index
     map<unsigned int, unsigned int> xref;
     for (unsigned int i=0; i!=n; i++) {
       if (fobs[i]->path()==p) {
         xref[fobs[i]->module()]=i;
       }
     }

     // path object for path with number p
     auto_ptr<HLTPathObject>   pathobject   (new HLTPathObject(p));
     map<unsigned int, unsigned int>::const_iterator iter;
     for (iter=xref.begin(); iter!=xref.end(); iter++) {
       LogDebug("") << "Path " << p << " " << iter->first << " " << iter->second;
       pathobject->put(fobs[iter->second]);
     }

     iEvent.put(pathobject);
     LogDebug("") << "Path " << p << " Number of filter objects: " << xref.size();
   }


   // get all path objects just inserted, and make and insert the single global object 
   // currently we insert an "empty" global object even if no path objects are found!

   vector<Handle<HLTPathObject> > pobs;
   iEvent.getManyByType(pobs);

   // order according to path number
   map<unsigned int, unsigned int> xref;
   for (unsigned int i=0; i!=pobs.size(); i++) {
     xref[pobs[i]->path()]=i;
   }

   // global object
   auto_ptr<HLTGlobalObject> globalobject (new HLTGlobalObject);
   map<unsigned int, unsigned int>::const_iterator iter;
   for (iter=xref.begin(); iter!=xref.end(); iter++) {
     LogDebug("") << "Global " << iter->first << " " << iter->second;
     globalobject->put(RefProd<HLTPathObject>(pobs[iter->second]));
   }

   iEvent.put(globalobject);
   LogDebug("") << "Number of path objects processed: " << pobs.size();

   return;
}
