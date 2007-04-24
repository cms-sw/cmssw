/*----------------------------------------------------------------------
  $Id: Principal.cc,v 1.2 2007/04/09 22:18:56 wdd Exp $
  ----------------------------------------------------------------------*/

#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "boost/lambda/lambda.hpp"
#include "boost/lambda/bind.hpp"

#include "Reflex/Type.h"
#include "Reflex/Base.h" // (needed for Type::HasBase to work correctly)

#include "FWCore/Framework/interface/Principal.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/ReflexTools.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/for_all.h"
#include "FWCore/Framework/interface/Selector.h"

using namespace std;
using ROOT::Reflex::Type;
using boost::lambda::_1;

namespace edm {

  Principal::Principal(ProductRegistry const& reg,
			       ProcessConfiguration const& pc,
			       ProcessHistoryID const& hist,
			       boost::shared_ptr<DelayedReader> rtrv) :
    EDProductGetter(),
    processHistoryID_(hist),
    processHistoryPtr_(boost::shared_ptr<ProcessHistory>(new ProcessHistory)),
    processConfiguration_(pc),
    processHistoryModified_(false),
    groups_(),
    branchDict_(),
    productDict_(),
    productLookup_(),
    elementLookup_(),
    inactiveGroups_(),
    inactiveBranchDict_(),
    inactiveProductDict_(),
    preg_(&reg),
    store_(rtrv)
  {
    if (processHistoryID_ != ProcessHistoryID()) {
      ProcessHistoryRegistry& history(*ProcessHistoryRegistry::instance());
      assert(history.notEmpty());
      bool found = history.getMapped(processHistoryID_, *processHistoryPtr_);
      assert(found);
    }
    groups_.reserve(reg.productList().size());
  }

  Principal::~Principal() {
  }

  Principal::size_type
  Principal::numEDProducts() const {
    return groups_.size();
  }
   
  void 
  Principal::addGroup(auto_ptr<Group> group) {
    assert (!group->productDescription().className().empty());
    assert (!group->productDescription().friendlyClassName().empty());
    assert (!group->productDescription().moduleLabel().empty());
    assert (!group->productDescription().processName().empty());
    SharedGroupPtr g(group);

    BranchKey const bk = BranchKey(g->productDescription());
    //cerr << "addGroup DEBUG 2---> " << bk.friendlyClassName_ << endl;
    //cerr << "addGroup DEBUG 3---> " << bk << endl;

    bool accessible = g->productAvailable();
    BranchDict & branchDict = (accessible ? branchDict_ : inactiveBranchDict_ );
    ProductDict & productDict = (accessible ? productDict_ : inactiveProductDict_ );
    GroupVec & groups = (accessible ? groups_ : inactiveGroups_ );

    BranchDict::iterator itFound = branchDict.find(bk);
    if (itFound != branchDict.end()) {
      if(groups[itFound->second]->replace(*g)) {
	return;
      } else {
	// the products are lost at this point!
	throw edm::Exception(edm::errors::InsertFailure,"AlreadyPresent")
	  << "addGroup: Problem found while adding product provanence, "
	  << "product already exists for ("
	  << bk.friendlyClassName_ << ","
	  << bk.moduleLabel_ << ","
	  << bk.productInstanceName_ << ","
	  << bk.processName_
	  << ")\n";
      }
    }

    // a memory allocation failure in modifying the product
    // data structures will cause things to be out of sync
    // we do not have any rollback capabilities as products 
    // and the indices are updated

    size_type slotNumber = groups.size();
    groups.push_back(g);

    branchDict[bk] = slotNumber;

    productDict[g->productDescription().productID()] = slotNumber;

    //cerr << "addGroup DEBUG 4---> " << bk.friendlyClassName_ << endl;

    if (accessible) {

      ProcessLookup& processLookup = productLookup_[bk.friendlyClassName_];
      vector<int>& vint = processLookup[bk.processName_];
      vint.push_back(slotNumber);

      ROOT::Reflex::Type type(ROOT::Reflex::Type::ByName(g->productDescription().className()));
      if (bool(type)) {

        // Here we look in the object named "type" for a typedef
        // named "value_type" and get the Reflex::Type for it.
        // Then check to ensure the Reflex dictionary is defined
        // for this value_type.
        // I do not throw an exception here if the check fails
        // because there are known cases where the dictionary does
        // not exist and we do not need to support those cases.
        ROOT::Reflex::Type valueType;
        if (edm::value_type_of(type, valueType) && bool(valueType)) {

          fillElementLookup(valueType, slotNumber, bk);

          // Repeat this for all public base classes of the value_type
          std::vector<ROOT::Reflex::Type> baseTypes;
          edm::public_base_classes(valueType, baseTypes);

          for (std::vector<ROOT::Reflex::Type>::iterator iter = baseTypes.begin(),
                                                       iend = baseTypes.end();
               iter != iend;
               ++iter) {
            fillElementLookup(*iter, slotNumber, bk);
          }
        }
      }
    }
  }

  void
  Principal::fillElementLookup(const ROOT::Reflex::Type & type,
                                   int slotNumber,
                                   const BranchKey& bk) {

    TypeID typeID(type.TypeInfo());
    std::string friendlyClassName = typeID.friendlyClassName();

    ProcessLookup& processLookup = elementLookup_[friendlyClassName];
    vector<int>& vint = processLookup[bk.processName_];
    vint.push_back(slotNumber);
  }

  void
  Principal::addToProcessHistory() const {
    if (processHistoryModified_) return;
    ProcessHistory& ph = *processHistoryPtr_;
    std::string const& processName = processConfiguration_.processName();
    for (ProcessHistory::const_iterator it = ph.begin(), itEnd = ph.end(); it != itEnd; ++it) {
      if (processName == it->processName()) {
	throw edm::Exception(errors::Configuration, "Duplicate Process")
	  << "The process name " << processName << " was previously used on these products.\n"
	  << "Please modify the configuration file to use a distinct process name.\n";
      }
    }
    ph.push_back(processConfiguration_);
    //OPTIMIZATION NOTE:  As of 0_9_0_pre3
    // For very simple Sources (e.g. EmptySource) this routine takes up nearly 50% of the time per event.
    // 96% of the time for this routine is being spent in computing the
    // ProcessHistory id which happens because we are reconstructing the ProcessHistory for each event.
    // (The process ID is first computed in the call to 'insertMapped(..)' below.)
    // It would probably be better to move the ProcessHistory construction out to somewhere
    // which persists for longer than one Event
    ProcessHistoryRegistry::instance()->insertMapped(ph);
    processHistoryID_ = ph.id();
    processHistoryModified_ = true;
  }

  ProcessHistory const&
  Principal::processHistory() const {
    return *processHistoryPtr_;
  }

  void 
  Principal::put(auto_ptr<EDProduct> edp,
		     auto_ptr<Provenance> prov) {

    if (!prov->productID().isValid()) {
      throw edm::Exception(edm::errors::InsertFailure,"Null Product ID")
	<< "put: Cannot put product with null Product ID."
	<< "\n";
    }
    if (edp.get() == 0) {
      throw edm::Exception(edm::errors::InsertFailure,"Null Pointer")
	<< "put: Cannot put because auto_ptr to product is null."
	<< "\n";
    }
    // Group assumes ownership
    auto_ptr<Group> g(new Group(edp, prov));
    this->addGroup(g);
    this->addToProcessHistory();
  }

  Principal::SharedConstGroupPtr const
  Principal::getGroup(ProductID const& oid, bool resolve) const {
    ProductDict::const_iterator i = productDict_.find(oid);
    if (i == productDict_.end()) {
      return getInactiveGroup(oid);
    }
    size_type slotNumber = i->second;
    assert(slotNumber < groups_.size());

    SharedConstGroupPtr const& g = groups_[slotNumber];
    if (resolve && g->provenance().isPresent()) {
      this->resolve_(*g);
    }
    return g;
  }

  Principal::SharedConstGroupPtr const
  Principal::getInactiveGroup(ProductID const& oid) const {
    ProductDict::const_iterator i = inactiveProductDict_.find(oid);
    if (i == inactiveProductDict_.end()) {
      return SharedConstGroupPtr();
    }
    size_type slotNumber = i->second;
    assert(slotNumber < inactiveGroups_.size());

    SharedConstGroupPtr const& g = inactiveGroups_[slotNumber];
    return g;
  }

  BasicHandle
  Principal::get(ProductID const& oid) const {
    if (oid == ProductID())
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "get by product ID: invalid ProductID supplied\n";

    ProductDict::const_iterator i = productDict_.find(oid);
    if (i == productDict_.end()) {
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "get by product ID: no product with given id: "<<oid<<"\n";
    }
    size_type slotNumber = i->second;
    assert(slotNumber < groups_.size());

    SharedConstGroupPtr const& g = groups_[slotNumber];
    resolve_(*g);

    // Check for case where we tried on demand production and
    // it failed to produce the object
    if (g->onDemand()) {
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "get by product ID: no product with given id: " << oid << "\n"
        << "onDemand production failed to produce it.\n";
    }
    return BasicHandle(g->product(), &g->provenance());
  }

  BasicHandle
  Principal::getBySelector(TypeID const& productType, 
			       SelectorBase const& sel) const {

    BasicHandleVec results;

    int nFound = findGroups(productType,
                            productLookup_,
                            sel,
                            results,
                            true);

    if (nFound == 0) {
      throw edm::Exception(edm::errors::ProductNotFound)
	<< "getBySelector: Found zero products matching all criteria\n"
	<< "Looking for type: " << productType << "\n";
    }
    if (nFound > 1) {
      throw edm::Exception(edm::errors::ProductNotFound)
        << "getBySelector: Found more than one product matching all criteria\n"
	<< "Looking for type: " << productType << "\n";
    }
    return results[0];
  }

  BasicHandle
  Principal::getByLabel(TypeID const& productType, 
			    string const& label,
			    string const& productInstanceName) const {

    BasicHandleVec results;

    edm::Selector sel(edm::ModuleLabelSelector(label) &&
                      edm::ProductInstanceNameSelector(productInstanceName));

    int nFound = findGroups(productType,
                            productLookup_,
                            sel,
                            results,
                            true);

    if (nFound == 0) {
      throw edm::Exception(edm::errors::ProductNotFound)
	<< "getByLabel: Found zero products matching all criteria\n"
	<< "Looking for type: " << productType << "\n"
	<< "Looking for module label: " << label << "\n"
	<< "Looking for productInstanceName: " << productInstanceName << "\n";
    }
    if (nFound > 1) {
      throw edm::Exception(edm::errors::ProductNotFound)
        << "getByLabel: Found more than one product matching all criteria\n"
	<< "Looking for type: " << productType << "\n"
	<< "Looking for module label: " << label << "\n"
	<< "Looking for productInstanceName: " << productInstanceName << "\n";
    }
    return results[0];
  }

  BasicHandle
  Principal::getByLabel(TypeID const& productType,
			    string const& label,
			    string const& productInstanceName,
			    string const& processName) const
  {

    BasicHandleVec results;

    edm::Selector sel(edm::ModuleLabelSelector(label) &&
                      edm::ProductInstanceNameSelector(productInstanceName) &&
                      edm::ProcessNameSelector(processName));

    int nFound = findGroups(productType,
                            productLookup_,
                            sel,
                            results,
                            true);

    if (nFound == 0) {
      throw edm::Exception(edm::errors::ProductNotFound)
	<< "getByLabel: Found zero products matching all criteria\n"
	<< "Looking for type: " << productType << "\n"
	<< "Looking for module label: " << label << "\n"
	<< "Looking for productInstanceName: " << productInstanceName << "\n"
	<< "Looking for process: " << processName << "\n";
    }
    if (nFound > 1) {
      throw edm::Exception(edm::errors::ProductNotFound)
        << "getByLabel: Found more than one product matching all criteria\n"
	<< "Looking for type: " << productType << "\n"
	<< "Looking for module label: " << label << "\n"
	<< "Looking for productInstanceName: " << productInstanceName << "\n"
	<< "Looking for process: " << processName << "\n";
    }
    return results[0];
  }
 

  void 
  Principal::getMany(TypeID const& productType, 
			 SelectorBase const& sel,
			 BasicHandleVec& results) const {

    findGroups(productType,
               productLookup_,
               sel,
               results,
               false);

    return;
  }

  BasicHandle
  Principal::getByType(TypeID const& productType) const {

    BasicHandleVec results;

    edm::MatchAllSelector sel;

    int nFound = findGroups(productType,
                            productLookup_,
                            sel,
                            results,
                            true);

    if (nFound == 0) {
      throw edm::Exception(edm::errors::ProductNotFound)
	<< "getByType: Found zero products matching all criteria\n"
	<< "Looking for type: " << productType << "\n";
    }
    if (nFound > 1) {
      throw edm::Exception(edm::errors::ProductNotFound)
        << "getByType: Found more than one product matching all criteria\n"
	<< "Looking for type: " << productType << "\n";
    }
    return results[0];
  }

  void 
  Principal::getManyByType(TypeID const& productType, 
			       BasicHandleVec& results) const {

    edm::MatchAllSelector sel;

    findGroups(productType,
               productLookup_,
               sel,
               results,
               false);
    return;
  }

  int
  Principal::getMatchingSequence(TypeID const& typeID,
                                     SelectorBase const& selector,
                                     BasicHandleVec& results,
                                     bool stopIfProcessHasMatch) const {

    // One new argument is the element lookup container
    // Otherwise this just passes through the arguments to findGroups
    return findGroups(typeID,
                      elementLookup_,
                      selector,
                      results,
                      stopIfProcessHasMatch);
  }

  Provenance const&
  Principal::getProvenance(ProductID const& oid) const {
    if (oid == ProductID())
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "getProvenance: invalid ProductID supplied\n";

    ProductDict::const_iterator i = productDict_.find(oid);
    if (i == productDict_.end()) {
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "getProvenance: no product with given id : "<< oid <<"\n";
    }

    size_type slotNumber = i->second;
    assert(slotNumber < groups_.size());

    SharedConstGroupPtr const& g = groups_[slotNumber];

    if (g->onDemand()) {
      unscheduledFill(*g);
    }
    // We already tried to produce the unscheduled products above
    // If they still are not there, then throw
    if (g->onDemand()) {
      throw edm::Exception(edm::errors::ProductNotFound)
	<< "getProvenance: no product with given ProductID: "<< oid <<"\n";
    }
    return g->provenance();
  }

  // This one is mostly for test printout purposes
  // No attempt to trigger on demand execution
  // Skips provenance when the EDProduct is not there
  void
  Principal::getAllProvenance(vector<Provenance const*> & provenances) const {
    provenances.clear();
    for (Principal::const_iterator i = groups_.begin(), iEnd = groups_.end(); i != iEnd; ++i) {
      SharedConstGroupPtr g = *i;
      if ((*i)->provenanceAvailable()) provenances.push_back(&(*i)->provenance());
    }
  }

  EDProduct const *
  Principal::getIt(ProductID const& oid) const {
    return get(oid).wrapper();
  }

  int
  Principal::findGroups(TypeID const& typeID,
                            TypeLookup const& typeLookup,
                            SelectorBase const& selector,
                            BasicHandleVec& results,
                            bool stopIfProcessHasMatch) const {

    assert(results.empty());

    TypeLookup::const_iterator i = typeLookup.find(typeID.friendlyClassName());

    if (i == typeLookup.end()) return 0;

    const ProcessLookup& processLookup = i->second;

    // Handle groups for current process, note that we need to
    // look at the current process even if it is not in the processHistory
    // because of potential unscheduled (onDemand) production
    findGroupsForProcess(processConfiguration_.processName(),
                         processLookup,
                         selector,
                         results);

    // Loop over processes in reverse time order.  Sometimes we want to stop
    // after we find a process with matches so check for that at each step.
    for (ProcessHistory::const_reverse_iterator iproc = processHistory().rbegin(),
                                                eproc = processHistory().rend();
         iproc != eproc && (results.empty() || !stopIfProcessHasMatch);
         ++iproc) {

      // We just dealt with the current process before the loop so skip it
      if (iproc->processName() == processConfiguration_.processName()) continue;

      findGroupsForProcess(iproc->processName(),
                           processLookup,
                           selector,
                           results);
    }
    return results.size();
  }

  void 
  Principal::findGroupsForProcess(std::string const& processName,
                                      ProcessLookup const& processLookup,
                                      SelectorBase const& selector,
                                      BasicHandleVec& results) const {

    ProcessLookup::const_iterator j = processLookup.find(processName);

    if (j == processLookup.end()) return;

    // This is a vector of indexes into the groups_ vector
    // These indexes point to groups with desired process name (and
    // also type when this function is called from findGroups)
    vector<int> const& vindex = j->second;

    for (vector<int>::const_iterator ib(vindex.begin()), ie(vindex.end());
           ib != ie;
           ++ib) {

      assert(static_cast<unsigned>(*ib) < groups_.size());
      SharedGroupPtr const& group = groups_[*ib];

      if (selector.match(group->provenance())) {

        // Try unscheduled execution or delayed read when needed
        this->resolve_(*group);

        // Unscheduled execution can fail to produce the EDProduct so check
        if (!group->onDemand()) {

          // Found a good match, save it
          results.push_back(BasicHandle(group->product(), &group->provenance()));
        }
      }
    }
    return;
  }

  void
  Principal::resolve_(Group const& g) const {
    if (!g.productAvailable())
      throw edm::Exception(errors::ProductNotFound,"InaccessibleProduct")
	<< "resolve_: product is not accessible\n"
	<< g.provenance() << '\n';

    if (g.product()) return; // nothing to do.

    // Try unscheduled production.
    if (g.onDemand()) {
      unscheduledFill(g);
      return;
    }

    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(g.productDescription());
    auto_ptr<EDProduct> edp(store_->get(bk, this));

    // Now fixup the Group
    g.setProduct(edp);
  }
}
