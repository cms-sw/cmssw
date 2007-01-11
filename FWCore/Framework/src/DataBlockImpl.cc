/*----------------------------------------------------------------------
  $Id: DataBlockImpl.cc,v 1.6 2007/01/10 05:58:48 wmtan Exp $
  ----------------------------------------------------------------------*/
#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "Reflex/Type.h"
#include "Reflex/Base.h" // (needed for Type::HasBase to work correctly)

#include "DataFormats/Common/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/DataBlockImpl.h"
#include "FWCore/Framework/src/ReflexTools.h"
#include "FWCore/Utilities/interface/EDMException.h"



using namespace std;

namespace edm {

  DataBlockImpl::DataBlockImpl(ProductRegistry const& reg,
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
    typeDict_(),
    inactiveGroups_(),
    inactiveBranchDict_(),
    inactiveProductDict_(),
    inactiveTypeDict_(),
    preg_(&reg),
    store_(rtrv),
    unscheduled_(false)
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

  DataBlockImpl::size_type
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

    size_type slotNumber = groups.size();
    groups.push_back(g);

    branchDict[bk] = slotNumber;

    productDict[g->productDescription().productID()] = slotNumber;

    //cerr << "addGroup DEBUG 4---> " << bk.friendlyClassName_ << endl;

    vector<int>& vint = typeDict[bk.friendlyClassName_];

    vint.push_back(slotNumber);
  }

  void
  DataBlockImpl::addToProcessHistory() {
    if (processHistoryModified_) return;
    ProcessHistory& ph = *processHistoryPtr_;
    std::string const& processName = processConfiguration_.processName();
    for (ProcessHistory::const_iterator it = ph.begin(); it != ph.end(); ++it) {
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
    auto_ptr<Group> g(new Group(edp, prov));
    g->setID(oid);
    this->addGroup(g);
    this->addToProcessHistory();
  }

  DataBlockImpl::SharedConstGroupPtr const
  DataBlockImpl::getGroup(ProductID const& oid, bool resolve) const {
    ProductDict::const_iterator i = productDict_.find(oid);
    if (i == productDict_.end()) {
      return getInactiveGroup(oid);
    }
    size_type slotNumber = i->second;
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
    size_type slotNumber = i->second;
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
    size_type slotNumber = i->second;
    assert(slotNumber < groups_.size());

    SharedConstGroupPtr const& g = groups_[slotNumber];
    this->resolve_(*g);
    return BasicHandle(g->product(), &g->provenance());
  }

  BasicHandle
  DataBlockImpl::getBySelector(TypeID const& productType, 
			       SelectorBase const& sel) const {
    TypeDict::const_iterator i 
      = typeDict_.find(productType.friendlyClassName());

    if(i==typeDict_.end()) {
      // TODO: Perhaps stuff like this should go to some error
      // logger?  Or do we want huge message inside the exception
      // that is thrown?
      edm::Exception err(edm::errors::ProductNotFound,"InvalidType");
      err << "getBySelector: no products found of correct type\n";
      err << "No products found of correct type\n";
      err << "We are looking for: '"
	  << productType
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
	<< productType<<"\n";
    }

    int found_count = 0;
    int found_slot = -1; // not a legal value!
    vector<int>::const_iterator ib(vint.begin()),ie(vint.end());

    BasicHandle result;

    while(ib!=ie) {
      SharedGroupPtr const& g = groups_[*ib];

      bool match = (unscheduled_ ? fillAndMatchSelector(g->provenance(), sel) : sel.match(g->provenance()));
      if (match) {
	++found_count;
	if (found_count > 1) {
	  throw edm::Exception(edm::errors::ProductNotFound,
			       "TooManyMatches")
	    << "getBySelector: too many products found for\n"
	    << productType<<"\n";
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
	  << productType<<"\n";
    }

    return result;
  }

    
  BasicHandle
  DataBlockImpl::getByLabel(TypeID const& productType, 
			    string const& label,
			    string const& productInstanceName) const {
    // The following is not the most efficient way of doing this. It
    // is the simplest implementation of the required policy, given
    // the current organization of the DataBlockImpl. This should be
    // reviewed.

    // THE FOLLOWING IS A HACK! It must be removed soon, with the
    // correct policy of making the assumed label be ... whatever we
    // set the policy to be. I don't know the answer right now...

    if (unscheduled_ && !processHistoryModified_) {
      // Unscheduled processing is enabled, and the current process is not
      // in the ProcessHistory.  We must check the current process first.
      BranchKey bk(productType.friendlyClassName(), label, productInstanceName, processConfiguration_.processName());
      BranchDict::const_iterator i = branchDict_.find(bk);
      if (i != branchDict_.end()) {
	// We found what we want.
        assert(i->second >= 0);
        assert(unsigned(i->second) < groups_.size());
	SharedConstGroupPtr group = groups_[i->second];
	this->resolve_(*group);
	return BasicHandle(group->product(), &group->provenance());    
      }
    }
 
    ProcessHistory::const_reverse_iterator iproc = processHistory().rbegin();
    ProcessHistory::const_reverse_iterator eproc = processHistory().rend();
    while (iproc != eproc) {
      string const& processName = iproc->processName();
      BranchKey bk(productType.friendlyClassName(), label, productInstanceName, processName);
      BranchDict::const_iterator i = branchDict_.find(bk);

      if (i != branchDict_.end()) {
	// We found what we want.
	assert(i->second >= 0);
        assert(size_t(i->second) < groups_.size());
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
      << "\"\nof type " << productType
      << " with product instance label \"" << productInstanceName << "\"\n";
  }

  BasicHandle
  DataBlockImpl::getByLabel(TypeID const& productType,
			    string const& label,
			    string const& productInstanceName,
			    string const& processName) const
  {
    BranchKey bk(productType.friendlyClassName(), label, 
		 productInstanceName, processName);
    BranchDict::const_iterator i = branchDict_.find(bk);
 
    if (i == branchDict_.end()) {
      // We failed to find the product we're looking for
      throw edm::Exception(errors::ProductNotFound,"NoMatch")
        << "getByLabel: could not find a product with module label \"" 
	<< label
        << "\"\nof type " << productType
        << " with product instance label \"" 
	<< productInstanceName << "\""
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
  DataBlockImpl::getMany(TypeID const& productType, 
			 SelectorBase const& sel,
			 BasicHandleVec& results) const {
    // We make no promise that the input 'fill_me_up' is unchanged if
    // an exception is thrown. If such a promise is needed, then more
    // care needs to be taken.
    TypeDict::const_iterator i = typeDict_.find(productType.friendlyClassName());

    if(i==typeDict_.end()) {
      return;
      // it is not an error to return no items
      // throw edm::Exception(errors::ProductNotFound,"NoMatch")
      //   << "getMany: no products found of correct type\n" << productType;
    }

    vector<int> const& vint = i->second;

    if(vint.empty()) {
      // should never happen!!
      throw edm::Exception(edm::errors::ProductNotFound,"EmptyList")
	<<  "getMany: no products found for\n"
	<< productType << '\n';
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
  DataBlockImpl::getByType(TypeID const& productType) const {

    TypeDict::const_iterator i = typeDict_.find(productType.friendlyClassName());

    if(i==typeDict_.end()) {
      throw edm::Exception(errors::ProductNotFound,"NoMatch")
        << "getByType: no product found of correct type\n" << productType
	<< '\n';
    }

    vector<int> const& vint = i->second;

    if(vint.empty()) {
      // should never happen!!
      throw edm::Exception(edm::errors::ProductNotFound,"EmptyList")
        <<  "getByType: no product found for\n"
        << productType <<'\n';
    }

    if(vint.size() > 1) {
      throw edm::Exception(edm::errors::ProductNotFound, "TooManyMatches")
        << "getByType: too many products found, "
        << "expected one, got " << vint.size() << ", for\n"
        << productType << '\n';
    }

    SharedConstGroupPtr const& g = groups_[vint[0]];
    this->resolve_(*g);
    return BasicHandle(g->product(), &g->provenance());
  }

  void 
  DataBlockImpl::getManyByType(TypeID const& productType, 
			       BasicHandleVec& results) const {
    // We make no promise that the input 'fill_me_up' is unchanged if
    // an exception is thrown. If such a promise is needed, then more
    // care needs to be taken.
    TypeDict::const_iterator i = typeDict_.find(productType.friendlyClassName());

    if(i==typeDict_.end()) {
      return;
      // it is not an error to find no match
      // throw edm::Exception(errors::ProductNotFound,"NoMatch")
      //   << "getManyByType: no products found of correct type\n" << productType;
    }

    vector<int> const& vint = i->second;

    if(vint.empty()) {
      // should never happen!!
      throw edm::Exception(edm::errors::ProductNotFound,"EmptyList")
        <<  "getManyByType: no products found for\n"
        << productType << '\n';
    }

    vector<int>::const_iterator ib(vint.begin()), ie(vint.end());
    while(ib != ie) {
      SharedConstGroupPtr const& g = groups_[*ib];
      this->resolve_(*g);
      results.push_back(BasicHandle(g->product(), &g->provenance()));
      ++ib;
    }
  }

  class NeitherSameNorDerivedType
  {
  public:
    typedef DataBlockImpl::MatchingGroups::value_type arg_t;

    explicit NeitherSameNorDerivedType(type_info const& elementType) :
      typeToMatch_(ROOT::Reflex::Type::ByTypeInfo(elementType)) 
    { }

    bool operator()(arg_t const& group) const
    {
      return !matches(group);
    }
  private:
    ROOT::Reflex::Type typeToMatch_;

    // Return true if the given group contains an EDProduct that is a
    // sequence, and if the value_type of that sequence is either the
    // same as our valuetype, or derives from our valuetype.
    bool matches(arg_t const& group) const
    {
      ROOT::Reflex::Type wrapperType = 
	ROOT::Reflex::Type::ByTypeInfo(typeid(*(group->product())));
      ROOT::Reflex::Type elementType;
      bool sequenceFound = is_sequence_wrapper(wrapperType, elementType);
      bool matchFound;
      if (sequenceFound)
	{
	  if ( elementType == typeToMatch_ || 
	       elementType.HasBase(typeToMatch_) )
	    {
	      matchFound = true;
	    }
	  else
	    {
	      matchFound = false;
	    }
    }
      return matchFound;
    }

  };

  BasicHandle
  DataBlockImpl::getMatchingSequence(type_info const& valuetype,
				     string const& moduleLabel,
				     string const& productInstanceName) const
  {
    // This code is in need of optimization. This is a prototype hack,
    // to get something functioning.
    BasicHandle result;

    // Find the matches for this module label and product instance
    // name, for *all* processes, grouped by process.
    MatchingGroupLookup matches;
    getAllMatches_(moduleLabel, productInstanceName, matches);
    
    // Shortcut -- we only have to continue if there is something that
    // might match.
    if (!matches.empty()) {
      
      // Loop through all processes, in backwards time order.
      ProcessHistory::const_reverse_iterator iproc = processHistory().rbegin();
      ProcessHistory::const_reverse_iterator eproc = processHistory().rend();
      while (iproc != eproc) {
	MatchingGroupLookup::iterator candidatesForProcess = 
	  matches.find(iproc->processName());

	if (candidatesForProcess != matches.end()) {
	  // We have found one or more groups for this process...
	  MatchingGroups& candidateGroups = candidatesForProcess->second;
	  assert(!candidateGroups.empty());
	  
	  NeitherSameNorDerivedType removalPredicate(valuetype);
	  candidateGroups.remove_if(removalPredicate);
	  
	  if (candidateGroups.size() == 1) {
	    // We've found what we're looking for, and it is unique.
	    // We can return the result.
	    BasicHandle result(candidateGroups.front()->product(),
			       &(candidateGroups.front()->provenance()));
	    assert (result.isValid());
	    return result;
	  } else if (candidateGroups.size() > 1) {
	    // We've found more than one match; this is an exceptional
	    // condition.
	    throw Exception(errors::ProductNotFound, "TooManyMatches")
	      << "DataBlockImpl::getMatchingSequence has found "
	      << candidateGroups.size()
	      << " matches for\n"
	      << "  module label: " << moduleLabel
	      << "  productInstanceName: " << productInstanceName
	      << "  processName: " << iproc->processName()
	      << "  value type: " << valuetype.name()
	      << '\n';
	  }
	  // If we're here, no candidate group survived the removal
	  // predicate. Go to the previous process and try again.
	}
	++iproc;
      }
    }
    // If we never find a match, throw an exception.
    //             TODO: Implement this function.
    throw edm::Exception(errors::ProductNotFound,"NoMatch")
      << "DataBlockImpl::getMatchingSequence could not find "
      << "any product with module label \"" 
      << moduleLabel
      << " and product instance name \"" 
      << productInstanceName
      << "\"\n";

    // The following never gets executed, but it makes some compilers
    // happy.
    return result;
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

    size_type slotNumber = i->second;
    assert(slotNumber < groups_.size());

    SharedConstGroupPtr const& g = groups_[slotNumber];
    return g->provenance();
  }

  void
  DataBlockImpl::getAllProvenance(vector<Provenance const*> & provenances) const {
    provenances.clear();
    for (DataBlockImpl::const_iterator i = groups_.begin(); i != groups_.end(); ++i) {
      provenances.push_back(&(*i)->provenance());
    }
  }

  void 
  DataBlockImpl::getAllMatches_(string const& moduleLabel,
				string const& productInstanceName,
				MatchingGroupLookup& matches) const {
    for (const_iterator i=groups_.begin(), e=groups_.end(); i != e; ++i) {
      Group const& currentGroup = **i;
      if (currentGroup.isAccessible() && 
	  currentGroup.moduleLabel() == moduleLabel &&
	  currentGroup.productInstanceName() == productInstanceName) {
	matches[currentGroup.processName()].push_back(*i);
      }
    }    
  }

  void
  DataBlockImpl::resolve_(Group const& g, bool unconditional) const {
    if (!unconditional && !g.isAccessible())
      throw edm::Exception(errors::ProductNotFound,"InaccessibleProduct")
	<< "resolve_: product is not accessible\n"
	<< g.provenance() << '\n';

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
