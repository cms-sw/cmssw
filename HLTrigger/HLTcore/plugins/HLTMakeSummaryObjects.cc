/** \class HLTMakeSummaryObjects
 *
 * See header file for documentation
 *
 *  $Date: 2007/02/17 14:27:00 $
 *  $Revision: 1.14 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTMakeSummaryObjects.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/HLTReco/interface/HLTGlobalObject.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/HLTReco/interface/HLTPathObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<string>
#include<vector>
#include<cassert>

//
// constructors and destructor
//
HLTMakeSummaryObjects::HLTMakeSummaryObjects(const edm::ParameterSet& iConfig)
  : tns_(0)
{

  if (edm::Service<edm::service::TriggerNamesService>().isAvailable()) {
    // get tns pointer
    tns_ = edm::Service<edm::service::TriggerNamesService>().operator->();
    if (tns_!=0) {
      const std::vector<std::string>& paths(tns_->getTrigPaths());
      const unsigned int n(paths.size());
      LogDebug("") << "Number of trigger paths: " << n;

      //register your products
      for (unsigned int p=0; p!=n; ++p) {
	produces<reco::HLTPathObject>(paths[p]);
	LogTrace("") << "Trigger path " << p << ": " << paths[p];
      }
      produces<reco::HLTGlobalObject>();
    } else {
      LogDebug("") << "HLT Error: TriggerNamesService pointer = 0!";
    }
  } else {
    LogDebug("") << "HLT Error: TriggerNamesService not available!";
  }

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

   if (tns_==0) {
     LogDebug("") << "HLT Error: tns_==0. Zero path summary objects and one empty global summary object put into the Event!";
     auto_ptr<HLTGlobalObject> globalobject (new HLTGlobalObject(0));
     iEvent.put(globalobject);
     return;
   }

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

   // store RefToBases and their labels, to all filter objects found,
   // in single vectors in order to allow unified treatment

   const unsigned int nfobs(n0+n1+n2);
   vector<RefToBase<HLTFilterObjectBase> > fobs(nfobs);
   vector<string> fobnames(nfobs);

   unsigned int i(0);
   for (unsigned int i0=0; i0!=n0; ++i0) {
     fobs[i]=RefToBase<HLTFilterObjectBase>(RefProd<HLTFilterObjectBase>(fob0[i0]));
     fobnames[i]=fob0[i0].provenance()->moduleLabel();
     ++i;
   }
   for (unsigned int i1=0; i1!=n1; ++i1) {
     fobs[i]=RefToBase<HLTFilterObjectBase>(RefProd<HLTFilterObject    >(fob1[i1]));
     fobnames[i]=fob1[i1].provenance()->moduleLabel();
     ++i;
   }
   for (unsigned int i2=0; i2!=n2; ++i2) {
     fobs[i]=RefToBase<HLTFilterObjectBase>(RefProd<HLTFilterObjectWithRefs>(fob2[i2]));
     fobnames[i]=fob2[i2].provenance()->moduleLabel();
     ++i;
   }
   assert (i==nfobs);

   // from now on, use only this combined vector!

   // construct the path objects and insert them in the Event
   // - currently we construct and insert "empty" path objects for paths
   // for which there is not filter object found!

   const vector<string>& paths(tns_->getTrigPaths());
   const unsigned int n(paths.size());
   vector<OrphanHandle<HLTPathObject> > pobs(n);

   // loop over all trigger paths
   for (unsigned int p=0; p!=n; ++p) {
     // path with path number p according to trigger names service
     LogTrace("") << "Trigger path " << p << ": " << paths[p];

     // create, fill and insert path summary object for path with number p
     auto_ptr<HLTPathObject> pathobject (new HLTPathObject(p));

     // the following two (instead of one) nested loops are needed to
     // cover the case that a filter module instance appears several
     // times in the trigger table, but due to Fw optimisation only
     // one is actually run and puts a filter object into the event
     // which is supposed to be re-used.

     // loop over all modules on trigger path p
     const vector<string>& modules(tns_->getTrigPathModules(p));
     const unsigned int pm(modules.size());
     for (unsigned int m=0; m!=pm; ++m) {
       // module with module number m and instance name
       const string& module(modules[m]);

       // number of objects alreay found and inserted
       unsigned int count(0);

       // loop over filter objects actually in this event
       for (unsigned int i=0; i!=nfobs; ++i) {
	 // filter object fobs[i] produced by module instance name?
	 if (module==fobnames[i]) {
	   // no other found already?
	   if (count==0) {
	     // insert and document
	     pathobject->put(fobs[i]);
             LogTrace("") << "  Path/module " << paths[p] << " " << module
             << " [" << p << " , " << m << " ] " << i
             << " [" << fobs[i]->path() << " , " << fobs[i]->module() << " ] ";
	   } else {
	     // have already found at least one earlier - a problem!
             LogTrace("") << "  Path/module " << paths[p] << " " << module
             << " [" << p << " , " << m << " ] " << i
             << " [" << fobs[i]->path() << " , " << fobs[i]->module() << " ] " 
             << " is duplicate - ignored: " << count;
	   }
	   count++;
	 }
       }
     }

     LogTrace("") << "Trigger path " << p << ": " << paths[p]
		  << " Number of filter objects: " << pathobject->size();
     pobs[p]=iEvent.put(pathobject,paths[p]);
   }

   // create, fill and insert the single global object of size n
   // - currently we insert an "empty" global object (n=0)
   // if no path objects are found (n=0)!

   auto_ptr<HLTGlobalObject> globalobject (new HLTGlobalObject(n));
   for (unsigned int p=0; p!=n; ++p) {
     globalobject->put(RefProd<HLTPathObject>(pobs[p]));
   }

   iEvent.put(globalobject);
   LogTrace("") << "Number of path objects processed: " << pobs.size();

   return;
}
