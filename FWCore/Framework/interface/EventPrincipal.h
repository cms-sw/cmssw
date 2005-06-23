#ifndef EDM_EVENT_PRINCIPAL_HH
#define EDM_EVENT_PRINCIPAL_HH

/*----------------------------------------------------------------------
  
EventPrincipal: This is the class responsible for management of
EDProducts. It is not seen by reconstruction code; such code sees the
Event class, which is a proxy for EventPrincipal.

The major internal component of the EventPrincipal is the Group, which
contains an EDProduct and its associated Provenance, along with
ancillary transient information regarding the two. Groups are handled
through shared pointers.

The EventPrincipal returns Handle<EDProduct>, rather than a shared
pointer to a Group, when queried.

$Id: EventPrincipal.h,v 1.2 2005/06/03 04:04:47 wmtan Exp $

----------------------------------------------------------------------*/
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

#include "boost/shared_ptr.hpp"

#include "FWCore/CoreFramework/interface/BranchKey.h"
#include "FWCore/EDProduct/interface/CollisionID.h"
#include "FWCore/EDProduct/interface/EDP_ID.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/CoreFramework/interface/EventAux.h"
#include "FWCore/CoreFramework/interface/Handle.h"
#include "FWCore/CoreFramework/interface/ProcessNameList.h"
#include "FWCore/CoreFramework/interface/Retriever.h"
#include "FWCore/CoreFramework/interface/Selector.h"

#include "FWCore/CoreFramework/src/Group.h"
#include "FWCore/CoreFramework/src/TypeID.h"

namespace edm {
    
  class EventPrincipal
  {
  public:
    typedef std::vector<boost::shared_ptr<Group> > GroupVec;
    typedef GroupVec::const_iterator               const_iterator;
    typedef std::vector<std::string>               ProcessNameList;
    typedef ProcessNameList::const_iterator        process_name_const_iterator;
    typedef boost::shared_ptr<Group>               SharedGroupPtr;
    typedef Handle<EDProduct>                      BasicHandle;
    typedef std::vector<BasicHandle>               BasicHandleVec;
    
    EventPrincipal();
    EventPrincipal(const CollisionID& id, Retriever& r, const ProcessNameList& nl = ProcessNameList());
    ~EventPrincipal();

    CollisionID id() const;

    // next two will not be available for a little while...
    //      const Run& getRun() const; 
    //      const LuminositySection& getLuminositySection() const; 
    
    void put(std::auto_ptr<EDProduct> edp,
	     std::auto_ptr<Provenance> prov);

    BasicHandle  get(EDP_ID oid) const;

    BasicHandle  getBySelector(TypeID id, const Selector& s) const;

    BasicHandle  getByLabel(TypeID id,
			    const std::string& label) const;

    void getMany(TypeID id, 
		 const Selector&,
		 BasicHandleVec& results) const;

    // ----- access to all products

    const_iterator begin() const { return groups_.begin(); }
    const_iterator end() const { return groups_.end(); }

    process_name_const_iterator beginProcess() const 
    { return aux_.process_history_.begin(); }

    process_name_const_iterator endProcess() const 
    { return aux_.process_history_.end(); }

    const ProcessNameList& processHistory() const;    

    // ----- manipulation of provenance

    void put(std::auto_ptr<Provenance> prov, bool accessible);

    const Provenance* getProvenance(EDP_ID id) const;


    // ----- Add a new Group
    // *this takes ownership of the Group, which in turn owns its
    // data.
    void addGroup(std::auto_ptr<Group> g);

    // ----- Mark this EventPrincipal as having been updated in the
    // given Process.
    void addToProcessHistory(const std::string& processName);

  private:
    EventAux aux_;	// persistent

    // EDP_ID is the index into these vectors
    GroupVec groups_; // products and provenances are persistent

    // users need to vary the info in the BranchKey object
    // to store the output of different code versions for the
    // same configured module (e.g. change process_name)

    // indices are to product/provenance slot
    typedef std::map<BranchKey,int> BranchDict;
    BranchDict labeled_dict_; // 1->1

    typedef std::map<std::string, std::vector<int> > TypeDict;
    TypeDict type_dict_; // 1->many

    // it is probably straightforward to load the BranchKey
    // dictionary above with information from the input source - 
    // mostly because this is information from provenance.
    // The product provanance are likewise easily filled.
    // The typeid index is more of a problem. Here
    // the I/O subsystem will need to take the friendly product
    // name string from provenance and get back a TypeID object.
    // Getting the products loaded (from the file) is another
    // issue. Here we may need some sort of hook into the I/O
    // system unless we want to preload all products (probably
    // not a good idea).
    // At MiniBooNE, this products object was directly part of
    // ROOT and the "gets" caused things to load properly - and
    // this is where the reservation for an object came into
    // the picture i.e. the "maker" function of the event.
    // should eventprincipal be pure interface?
    // should ROOT just be present here?

    // luminosity section and run need to be added and are a problem

    // What goes into the event header(s)? Which need to be persistent?


    // Pointer to the 'service' that will be used to obtain EDProducts
    // from the persistent store.
    Retriever* store_;

    // Make my Retriever get the EDProduct for a Group.  The Group is
    // a cache, and so can be modified through the const reference.
    // We do not change the *number* of groups through this call, and so
    // *this is const.
    void resolve_(const Group& g) const;

  };
}
#endif
