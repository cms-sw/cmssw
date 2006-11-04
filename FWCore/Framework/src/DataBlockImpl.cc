/*----------------------------------------------------------------------
$Id: DataBlockImpl.cc,v 1.2 2006/10/31 23:54:01 wmtan Exp $
----------------------------------------------------------------------*/
#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "DataFormats/Common/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/DataBlockImpl.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace std;

namespace edm {

  DataBlockImpl::DataBlockImpl(ProductRegistry const& reg,
				 ProcessHistoryID const& hist,
				 boost::shared_ptr<DelayedReader> rtrv) :
    EDProductGetter(),
    processHistoryID_(hist),
    processHistoryPtr_(boost::shared_ptr<ProcessHistory>(new ProcessHistory)),
    groups_(),
    branchDict_(),
    productDict_(),
    typeDict_(),
    inactiveGroups_(),
    inactiveBranchDict_(),
    inactiveProductDict_(),
    inactiveTypeDict_(),
    preg_(&reg),
    store_(rtrv)
  {
    if (processHistoryID_ != ProcessHistoryID()) {
      assert(ProcessHistoryRegistry::instance()->size());
      bool found = ProcessHistoryRegistry::instance()->getMapped(processHistoryID_, *processHistoryPtr_);
      assert(found);
    }
    groups_.reserve(reg.productList().size());
  }

  DataBlockImpl::~DataBlockImpl() {
  }

  unsigned long
  DataBlockImpl::numEDProducts() const {
    return groups_.size();
  }
   
  void 
  DataBlockImpl::addGroup(auto_ptr<Group> group) {
    assert (!group->productDescription().className().empty());
    assert (!group->productDescription().friendlyClassName().empty());
    assert (!group->productDescription().moduleLabel().empty());
    assert (!group->productDescription().processName().empty());
    SharedGroupPtr g(group);

    BranchKey const bk = BranchKey(g->productDescription());
    //cerr << "addGroup DEBUG 2---> " << bk.friendlyClassName_ << endl;
    //cerr << "addGroup DEBUG 3---> " << bk << endl;

    bool accessible = g->isAccessible();
    BranchDict & branchDict = (accessible ? branchDict_ : inactiveBranchDict_ );
    ProductDict & productDict = (accessible ? productDict_ : inactiveProductDict_ );
    TypeDict & typeDict = (accessible ? typeDict_ : inactiveTypeDict_ );
    GroupVec & groups = (accessible ? groups_ : inactiveGroups_ );

    BranchDict::iterator itFound = branchDict.find(bk);
    if (itFound != branchDict.end()) {
       if(!groups[itFound->second]->product()) {
          // is null, so this new one must be the one generated 'unscheduled'
          groups[itFound->second]->swapProduct(*g);
          //NOTE: other API's of DataBlockImpl give out the Provenance* so need to preserve the memory
          groups[itFound->second]->provenance() = g->provenance();
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

    unsigned long slotNumber = groups.size();
    groups.push_back(g);

    branchDict[bk] = slotNumber;

    productDict[g->productDescription().productID()] = slotNumber;

    //cerr << "addGroup DEBUG 4---> " << bk.friendlyClassName_ << endl;

    vector<int>& vint = typeDict[bk.friendlyClassName_];

    vint.push_back(slotNumber);
  }

  void
  DataBlockImpl::addToProcessHistory(ProcessConfiguration const& processConfiguration) {
    ProcessHistory& ph = *processHistoryPtr_;
    std::string const& processName = processConfiguration.processName();
    for (ProcessHistory::const_iterator it = ph.begin(); it != ph.end(); ++it) {
      if (processName == it->processName()) {
	throw edm::Exception(errors::Configuration, "Duplicate Process")
	  << "The process name " << processName << " was previously used on these products.\n"
	  << "Please modify the configuration file to use a distinct process name.";
      }
    }
    ph.push_back(processConfiguration);
    //OPTIMIZATION NOTE:  As of 0_9_0_pre3
    // For very simple Sources (e.g. EmptySource) this routine takes up nearly 50% of the time per event.
    // 96% of the time for this routine is being spent in computing the
    // ProcessHistory id which happens because we are reconstructing the ProcessHistory for each event.
    // (The process ID is first computed in the call to 'insertMapped(..)' below.)
    // It would probably be better to move the ProcessHistory construction out to somewhere
    // which persists for longer than one Event
    ProcessHistoryRegistry::instance()->insertMapped(ph);
    processHistoryID_ = ph.id();
  }

  ProcessHistory const&
  DataBlockImpl::processHistory() const {
    return *processHistoryPtr_;
  }

  void 
  DataBlockImpl::put(auto_ptr<EDProduct> edp,
		      auto_ptr<Provenance> prov) {

    if (prov->productID() == ProductID()) {
	throw edm::Exception(edm::errors::InsertFailure,"Null Product ID")
	  << "put: Cannot put product with null Product ID."
	  << "\n";
    }
    ProductID oid = prov->productID();

    // Group assumes ownership
    std::auto_ptr<Group> g(new Group(edp, prov));
    g->setID(oid);
    this->addGroup(g);
  }

  DataBlockImpl::SharedConstGroupPtr const
  DataBlockImpl::getGroup(ProductID const& oid, bool resolve) const {
    ProductDict::const_iterator i = productDict_.find(oid);
    if (i == productDict_.end()) {
	return getInactiveGroup(oid);
    }
    unsigned long slotNumber = i->second;
    assert(slotNumber < groups_.size());

    SharedConstGroupPtr const& g = groups_[slotNumber];
    if (resolve && g->provenance().isPresent()) {
      this->resolve_(*g, true);
    }
    return g;
  }

  DataBlockImpl::SharedConstGroupPtr const
  DataBlockImpl::getInactiveGroup(ProductID const& oid) const {
    ProductDict::const_iterator i = inactiveProductDict_.find(oid);
    if (i == inactiveProductDict_.end()) {
	return SharedConstGroupPtr();
    }
    unsigned long slotNumber = i->second;
    assert(slotNumber < inactiveGroups_.size());

    SharedConstGroupPtr const& g = inactiveGroups_[slotNumber];
    return g;
  }

  BasicHandle
  DataBlockImpl::get(ProductID const& oid) const {
    if (oid == ProductID())
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "get by product ID: invalid ProductID supplied\n";

    ProductDict::const_iterator i = productDict_.find(oid);
    if (i == productDict_.end()) {
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "get by product ID: no product with given id: "<<oid<<"\n";
    }
    unsigned long slotNumber = i->second;
    assert(slotNumber < groups_.size());

    SharedConstGroupPtr const& g = groups_[slotNumber];
    this->resolve_(*g);
    return BasicHandle(g->product(), &g->provenance());
  }

  BasicHandle
  DataBlockImpl::getBySelector(TypeID const& tid, 
				SelectorBase const& sel) const {
    TypeDict::const_iterator i = typeDict_.find(tid.friendlyClassName());

    if(i==typeDict_.end()) {
	// TODO: Perhaps stuff like this should go to some error
	// logger?  Or do we want huge message inside the exception
	// that is thrown?
	edm::Exception err(edm::errors::ProductNotFound,"InvalidType");
	err << "getBySelector: no products found of correct type\n";
	err << "No products found of correct type\n";
	err << "We are looking for: '"
	     << tid
	     << "'\n";
	if (typeDict_.empty()) {
	    err << "typeDict_ is empty!\n";
	} else {
	    err << "We found only the following:\n";
	    TypeDict::const_iterator j = typeDict_.begin();
	    TypeDict::const_iterator e = typeDict_.end();
	    while (j != e) {
		err << "...\t" << j->first << '\n';
		++j;
	    }
	}
	err << ends;
	throw err;
    }

    vector<int> const& vint = i->second;

    if (vint.empty()) {
	// should never happen!!
	throw edm::Exception(edm::errors::ProductNotFound,"EmptyList")
	  <<  "getBySelector: no products found for\n"
	  << tid;
    }

    int found_count = 0;
    int found_slot = -1; // not a legal value!
    vector<int>::const_iterator ib(vint.begin()),ie(vint.end());

    BasicHandle result;

    while(ib!=ie) {
	SharedGroupPtr const& g = groups_[*ib];

	if (fillAndMatchSelector(g->provenance(), sel)) {
	    ++found_count;
	    if (found_count > 1) {
		throw edm::Exception(edm::errors::ProductNotFound,
				     "TooManyMatches")
		  << "getBySelector: too many products found, "
		  << "expected one, got " << found_count << ", for\n"
		  << tid;
	    }
	    found_slot = *ib;
	    this->resolve_(*g);
	    result = BasicHandle(g->product(), &g->provenance());
	}
	++ib;
    }

    if (found_count == 0) {
	throw edm::Exception(edm::errors::ProductNotFound,"TooFewProducts")
	  << "getBySelector: too few products found (zero) for\n"
	  << tid;
    }

    return result;
  }

    
  BasicHandle
  DataBlockImpl::getByLabel(TypeID const& tid, 
			     string const& label,
			     string const& productInstanceName) const {
    // The following is not the most efficient way of doing this. It
    // is the simplest implementation of the required policy, given
    // the current organization of the DataBlockImpl. This should be
    // reviewed.

    // THE FOLLOWING IS A HACK! It must be removed soon, with the
    // correct policy of making the assumed label be ... whatever we
    // set the policy to be. I don't know the answer right now...

    ProcessHistory::const_reverse_iterator iproc = processHistory().rbegin();
    ProcessHistory::const_reverse_iterator eproc = processHistory().rend();
    while (iproc != eproc) {
	string const& processName = iproc->processName();
	BranchKey bk(tid.friendlyClassName(), label, productInstanceName, processName);
	BranchDict::const_iterator i = branchDict_.find(bk);

	if (i != branchDict_.end()) {
	    // We found what we want.
            assert(i->second >= 0);
            assert(unsigned(i->second) < groups_.size());
	    SharedConstGroupPtr group = groups_[i->second];
	    this->resolve_(*group);
	    return BasicHandle(group->product(), &group->provenance());    
	}
	++iproc;
    }
    // We failed to find the product we're looking for, under *any*
    // process name... throw!
    throw edm::Exception(errors::ProductNotFound,"NoMatch")
      << "getByLabel: could not find a product with module label \"" << label
      << "\"\nof type " << tid
      << " with product instance label \"" << (productInstanceName.empty() ? "" : productInstanceName) << "\"\n";
  }

  BasicHandle
  DataBlockImpl::getByLabel(TypeID const& tid,
			     string const& label,
			     string const& productInstanceName,
			     string const& processName) const
  {
    BranchKey bk(tid.friendlyClassName(), label, productInstanceName, processName);
    BranchDict::const_iterator i = branchDict_.find(bk);
 
    if (i == branchDict_.end()) {
      // We failed to find the product we're looking for
      throw edm::Exception(errors::ProductNotFound,"NoMatch")
        << "getByLabel: could not find a product with module label \"" << label
        << "\"\nof type " << tid
        << " with product instance label \"" << (productInstanceName.empty() ? "" : productInstanceName) << "\""
        << " and process name " << processName << "\n";
    }
    // We found what we want.
    assert(i->second >= 0);
    assert(unsigned(i->second) < groups_.size());
    SharedConstGroupPtr group = groups_[i->second];
    this->resolve_(*group);
    return BasicHandle(group->product(), &group->provenance());
  }
 

  void 
  DataBlockImpl::getMany(TypeID const& tid, 
			  SelectorBase const& sel,
			  BasicHandleVec& results) const {
    // We make no promise that the input 'fill_me_up' is unchanged if
    // an exception is thrown. If such a promise is needed, then more
    // care needs to be taken.
    TypeDict::const_iterator i = typeDict_.find(tid.friendlyClassName());

    if(i==typeDict_.end()) {
	return;
	// it is not an error to return no items
	// throw edm::Exception(errors::ProductNotFound,"NoMatch")
	//   << "getMany: no products found of correct type\n" << tid;
    }

    vector<int> const& vint = i->second;

    if(vint.empty()) {
	// should never happen!!
	throw edm::Exception(edm::errors::ProductNotFound,"EmptyList")
	  <<  "getMany: no products found for\n"
	  << tid;
    }

    vector<int>::const_iterator ib(vint.begin()), ie(vint.end());
    while(ib != ie) {
      SharedGroupPtr const& g = groups_[*ib];
      if (sel.match(g->provenance())) {
        this->resolve_(*g);
        results.push_back(BasicHandle(g->product(), &g->provenance()));
      }
      ++ib;
    }
  }

  BasicHandle
  DataBlockImpl::getByType(TypeID const& tid) const {

    TypeDict::const_iterator i = typeDict_.find(tid.friendlyClassName());

    if(i==typeDict_.end()) {
      throw edm::Exception(errors::ProductNotFound,"NoMatch")
        << "getByType: no product found of correct type\n" << tid;
    }

    vector<int> const& vint = i->second;

    if(vint.empty()) {
      // should never happen!!
      throw edm::Exception(edm::errors::ProductNotFound,"EmptyList")
        <<  "getByType: no product found for\n"
        << tid;
    }

    if(vint.size() > 1) {
      throw edm::Exception(edm::errors::ProductNotFound, "TooManyMatches")
        << "getByType: too many products found, "
        << "expected one, got " << vint.size() << ", for\n"
        << tid;
    }

    SharedConstGroupPtr const& g = groups_[vint[0]];
    this->resolve_(*g);
    return BasicHandle(g->product(), &g->provenance());
  }

  void 
  DataBlockImpl::getManyByType(TypeID const& tid, 
			  BasicHandleVec& results) const {
    // We make no promise that the input 'fill_me_up' is unchanged if
    // an exception is thrown. If such a promise is needed, then more
    // care needs to be taken.
    TypeDict::const_iterator i = typeDict_.find(tid.friendlyClassName());

    if(i==typeDict_.end()) {
		return;
      // it is not an error to find no match
      // throw edm::Exception(errors::ProductNotFound,"NoMatch")
      //   << "getManyByType: no products found of correct type\n" << tid;
    }

    vector<int> const& vint = i->second;

    if(vint.empty()) {
      // should never happen!!
      throw edm::Exception(edm::errors::ProductNotFound,"EmptyList")
        <<  "getManyByType: no products found for\n"
        << tid;
    }

    vector<int>::const_iterator ib(vint.begin()), ie(vint.end());
    while(ib != ie) {
      SharedConstGroupPtr const& g = groups_[*ib];
      this->resolve_(*g);
      results.push_back(BasicHandle(g->product(), &g->provenance()));
      ++ib;
    }
  }

  Provenance const&
  DataBlockImpl::getProvenance(ProductID const& oid) const {
    if (oid == ProductID())
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "getProvenance: invalid ProductID supplied\n";

    ProductDict::const_iterator i = productDict_.find(oid);
    if (i == productDict_.end()) {
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "getProvenance: no product with given id : "<< oid <<"\n";
    }

    unsigned long slotNumber = i->second;
    assert(slotNumber < groups_.size());

    SharedConstGroupPtr const& g = groups_[slotNumber];
    return g->provenance();
  }

  void
  DataBlockImpl::getAllProvenance(std::vector<Provenance const*> & provenances) const {
    provenances.clear();
    for (DataBlockImpl::const_iterator i = groups_.begin(); i != groups_.end(); ++i) {
      provenances.push_back(&(*i)->provenance());
    }
  }

  void
  DataBlockImpl::resolve_(Group const& g, bool unconditional) const {
    if (!unconditional && !g.isAccessible())
      throw edm::Exception(errors::ProductNotFound,"InaccessibleProduct")
	<< "resolve_: product is not accessible\n"
	<< g.provenance();

    if (g.product()) return; // nothing to do.

    // Try unscheduled production.
    if (unscheduledFill(g)) return;

    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(g.productDescription());
    auto_ptr<EDProduct> edp(store_->get(bk, this));

    // Now fixup the Group
    g.setProduct(edp);
  }

  EDProduct const *
  DataBlockImpl::getIt(ProductID const& oid) const {
    return get(oid).wrapper();
  }
}
