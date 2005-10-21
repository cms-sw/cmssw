/*----------------------------------------------------------------------
$Id: EventPrincipal.cc,v 1.26 2005/09/30 21:16:00 paterno Exp $
----------------------------------------------------------------------*/
//#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <algorithm>

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace std;

namespace edm {

  EventPrincipal::EventPrincipal() :
    aux_(),
    groups_(),
    branchDict_(),
    typeDict_(),
    preg_(0),
    store_()
  { }


  EventPrincipal::EventPrincipal(EventID const& id,
				 Timestamp const& time,
                                 ProductRegistry const& reg,
				 ProcessNameList const& nl,
				 boost::shared_ptr<DelayedReader> rtrv) :
   aux_(id,time),
   groups_(),
   branchDict_(),
   typeDict_(),
   preg_(&reg),
   store_(rtrv)
  {
    aux_.process_history_ = nl;
    groups_.reserve(reg.productList().size());
  }

  EventPrincipal::~EventPrincipal() {
  }

  EventID
  EventPrincipal::id() const {
    return aux_.id_;
  }

  Timestamp
  EventPrincipal::time() const {
    return aux_.time_;
  }

  unsigned long
  EventPrincipal::numEDProducts() const {
    return groups_.size();
  }
   
  void 
  EventPrincipal::addGroup(auto_ptr<Group> group) {
    assert (!group->productDescription().fullClassName_.empty());
    assert (!group->productDescription().friendlyClassName_.empty());
    assert (!group->productDescription().module.moduleLabel_.empty());
    assert (!group->productDescription().module.processName_.empty());
    SharedGroupPtr g(group);

    BranchKey const bk = BranchKey(g->productDescription());
    //cerr << "addGroup DEBUG 2---> " << bk.friendlyClassName_ << endl;
    //cerr << "addGroup DEBUG 3---> " << bk << endl;

    if (branchDict_.find(bk) != branchDict_.end()) {
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

    // a memory allocation failure in modifying the product
    // data structures will cause things to be out of sync
    // we do not have any rollback capabilities as products 
    // and the indices are updated

    unsigned long slotNumber = groups_.size();
    groups_.push_back(g);

    branchDict_[bk] = slotNumber;

    productDict_[g->productDescription().productID_] = slotNumber;

    //cerr << "addGroup DEBUG 4---> " << bk.friendlyClassName_ << endl;

    vector<int>& vint = typeDict_[bk.friendlyClassName_];

    vint.push_back(slotNumber);
  }

  void
  EventPrincipal::addToProcessHistory(string const& processName) {
    aux_.process_history_.push_back(processName);
  }

  ProcessNameList const&
  EventPrincipal::processHistory() const {
    return aux_.process_history_;
  }

  void 
  EventPrincipal::put(auto_ptr<EDProduct> edp,
		      auto_ptr<Provenance> prov) {
    prov->product.init();
    ProductRegistry::ProductList const& pl = preg_->productList();
    BranchKey const bk(prov->product);
    ProductRegistry::ProductList::const_iterator it = pl.find(bk);
    if (it == pl.end()) {
	throw edm::Exception(edm::errors::InsertFailure,"Not Registered")
	  << "put: Problem found while adding product. "
	  << "No product is registered for ("
	  << bk.friendlyClassName_ << ","
          << bk.moduleLabel_ << ","
          << bk.productInstanceName_ << ","
          << bk.processName_
	  << ")\n";
    }
    prov->product.productID_ = it->second.productID_;
    ProductID id = it->second.productID_;
    // Group assumes ownership
    auto_ptr<Group> g(new Group(edp, prov));
    g->setID(id);
    this->addGroup(g);
  }

  EventPrincipal::SharedGroupPtr const
  EventPrincipal::getGroup(ProductID const& oid) const {
    ProductDict::const_iterator i = productDict_.find(oid);
    if (i == productDict_.end()) {
	return SharedGroupPtr();
    }
    unsigned long slotNumber = i->second;
    assert(slotNumber < groups_.size());

    return groups_[slotNumber];
  }

  BasicHandle
  EventPrincipal::get(ProductID const& oid) const {
    if (oid == ProductID())
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "get by product ID: invalid ProductID supplied\n";

    SharedGroupPtr const& g = getGroup(oid);
    if (g == SharedGroupPtr()) {
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "get by product ID: no product with given id\n";
    }
    this->resolve_(*g);
    return BasicHandle(g->product(), &g->provenance());
  }

  BasicHandle
  EventPrincipal::getBySelector(TypeID const& id, 
				Selector const& sel) const {
    TypeDict::const_iterator i = typeDict_.find(id.friendlyClassName());

    if(i==typeDict_.end()) {
	// TODO: Perhaps stuff like this should go to some error
	// logger?  Or do we want huge message inside the exception
	// that is thrown?
	edm::Exception err(edm::errors::ProductNotFound,"InvalidType");
	err << "getBySelector: no products found of correct type\n";
	err << "No products found of correct type\n";
	err << "We are looking for: '"
	     << id
	     << "'\n";
	if (typeDict_.empty()) {
	    err << "typeDict_ is empty!\n";
	} else {
	    err << "We found only the following:\n";
	    TypeDict::const_iterator i = typeDict_.begin();
	    TypeDict::const_iterator e = typeDict_.end();
	    while (i != e) {
		err << "...\t" << i->first << '\n';
		++i;
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
	  << id;
    }

    int found_count = 0;
    int found_slot = -1; // not a legal value!
    vector<int>::const_iterator ib(vint.begin()),ie(vint.end());

    BasicHandle result;

    while(ib!=ie) {
	SharedGroupPtr const& g = groups_[*ib];
	if(sel.match(g->provenance())) {
	    ++found_count;
	    if (found_count > 1) {
		throw edm::Exception(edm::errors::ProductNotFound,
				     "TooManyMatches")
		  << "getBySelector: too many products found, "
		  << "expected one, got " << found_count << ", for\n"
		  << id;
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
	  << id;
    }

    return result;
  }

    
  BasicHandle
  EventPrincipal::getByLabel(TypeID const& id, 
			     string const& label,
			     string const& productInstanceName) const {
    // The following is not the most efficient way of doing this. It
    // is the simplest implementation of the required policy, given
    // the current organization of the EventPrincipal. This should be
    // reviewed.

    // THE FOLLOWING IS A HACK! It must be removed soon, with the
    // correct policy of making the assumed label be ... whatever we
    // set the policy to be. I don't know the answer right now...

    ProcessNameList::const_reverse_iterator iproc = aux_.process_history_.rbegin();
    ProcessNameList::const_reverse_iterator eproc = aux_.process_history_.rend();
    while (iproc != eproc) {
	string const& processName_ = *iproc;
	BranchKey bk(id, label, productInstanceName, processName_);
	BranchDict::const_iterator i = branchDict_.find(bk);

	if (i != branchDict_.end()) {
	    // We found what we want.
            assert(i->second >= 0);
            assert(unsigned(i->second) < groups_.size());
	    SharedGroupPtr group = groups_[i->second];
	    this->resolve_(*group);
	    return BasicHandle(group->product(), &group->provenance());    
	}
	++iproc;
    }
    // We failed to find the product we're looking for, under *any*
    // process name... throw!
    throw edm::Exception(errors::ProductNotFound,"NoMatch")
      << "getByLabel: could not find a required product " << label
      << "\nof type " << id
      << " with user tag " << (productInstanceName.empty() ? "\"\"" : productInstanceName) << '\n';
  }

  void 
  EventPrincipal::getMany(TypeID const& id, 
			  Selector const& sel,
			  BasicHandleVec& results) const {
    // We make no promise that the input 'fill_me_up' is unchanged if
    // an exception is thrown. If such a promise is needed, then more
    // care needs to be taken.
    TypeDict::const_iterator i = typeDict_.find(id.friendlyClassName());

    if(i==typeDict_.end()) {
	throw edm::Exception(errors::ProductNotFound,"NoMatch")
	  << "getMany: no products found of correct type\n" << id;
    }

    vector<int> const& vint = i->second;

    if(vint.empty()) {
	// should never happen!!
	throw edm::Exception(edm::errors::ProductNotFound,"EmptyList")
	  <<  "getMany: no products found for\n"
	  << id;
    }

    vector<int>::const_iterator ib(vint.begin()), ie(vint.end());
    while(ib != ie) {
	SharedGroupPtr const& g = groups_[*ib];
	if(sel.match(g->provenance())) {
	    this->resolve_(*g);
	    results.push_back(BasicHandle(g->product(), &g->provenance()));
	}
	++ib;
    }
  }

  BasicHandle
  EventPrincipal::getByType(TypeID const& id) const {

    TypeDict::const_iterator i = typeDict_.find(id.friendlyClassName());

    if(i==typeDict_.end()) {
      throw edm::Exception(errors::ProductNotFound,"NoMatch")
        << "getByType: no product found of correct type\n" << id;
    }

    vector<int> const& vint = i->second;

    if(vint.empty()) {
      // should never happen!!
      throw edm::Exception(edm::errors::ProductNotFound,"EmptyList")
        <<  "getByType: no product found for\n"
        << id;
    }

    if(vint.size() > 1) {
      throw edm::Exception(edm::errors::ProductNotFound, "TooManyMatches")
        << "getByType: too many products found, "
        << "expected one, got " << vint.size() << ", for\n"
        << id;
    }

    SharedGroupPtr const& g = groups_[vint[0]];
    this->resolve_(*g);
    return BasicHandle(g->product(), &g->provenance());
  }

  void 
  EventPrincipal::getManyByType(TypeID const& id, 
			  BasicHandleVec& results) const {
    // We make no promise that the input 'fill_me_up' is unchanged if
    // an exception is thrown. If such a promise is needed, then more
    // care needs to be taken.
    TypeDict::const_iterator i = typeDict_.find(id.friendlyClassName());

    if(i==typeDict_.end()) {
      throw edm::Exception(errors::ProductNotFound,"NoMatch")
        << "getManyByType: no products found of correct type\n" << id;
    }

    vector<int> const& vint = i->second;

    if(vint.empty()) {
      // should never happen!!
      throw edm::Exception(edm::errors::ProductNotFound,"EmptyList")
        <<  "getManyByType: no products found for\n"
        << id;
    }

    vector<int>::const_iterator ib(vint.begin()), ie(vint.end());
    while(ib != ie) {
      SharedGroupPtr const& g = groups_[*ib];
      this->resolve_(*g);
      results.push_back(BasicHandle(g->product(), &g->provenance()));
      ++ib;
    }
  }

  Provenance const&
  EventPrincipal::getProvenance(ProductID const& oid) const {
    if (oid == ProductID())
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "getProvenance: invalid ProductID supplied\n";

    ProductDict::const_iterator i = productDict_.find(oid);
    if (i == productDict_.end()) {
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "getProvenance: no product with given id\n";
    }

    unsigned long slotNumber = i->second;
    assert(slotNumber < groups_.size());

    SharedGroupPtr const& g = groups_[slotNumber];
    return g->provenance();
  }

  void
  EventPrincipal::getAllProvenance(std::vector<Provenance const*> & provenances) const {
    provenances.clear();
    for (EventPrincipal::const_iterator i = groups_.begin(); i != groups_.end(); ++i) {
      provenances.push_back(&(*i)->provenance());
    }
  }

  void
  EventPrincipal::resolve_(Group const& g) const {
    if (!g.isAccessible())
      throw edm::Exception(errors::ProductNotFound,"InaccessibleProduct")
	<< "resolve_: product is not accessible\n"
	<< g.provenance();

    if (g.product()) return; // nothing to do.
    
    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(g.productDescription());
    auto_ptr<EDProduct> edp(store_->get(bk));

    // Now fixup the Group
    g.setProduct(edp);
  }

}
