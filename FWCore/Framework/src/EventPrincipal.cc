/*----------------------------------------------------------------------
$Id: EventPrincipal.cc,v 1.16 2005/07/26 04:42:28 wmtan Exp $
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
    labeled_dict_(),
    type_dict_(),
    store_(0),
    preg_(0)
  {
  }

  EventPrincipal::EventPrincipal(CollisionID const& id,
	 Retriever& r, ProductRegistry const& reg, ProcessNameList const& nl) :
    aux_(id),
    groups_(),
    labeled_dict_(),
    type_dict_(),
    store_(&r),
    preg_(&reg)
  {
    aux_.process_history_ = nl;
    groups_.reserve(reg.productList().size());
  }
 
  EventPrincipal::~EventPrincipal() {
  }

  CollisionID
  EventPrincipal::id() const {
    return aux_.id_;
  }

  void 
  EventPrincipal::addGroup(auto_ptr<Group> group) {
    assert (!group->productDescription().full_product_type_name.empty());
    assert (!group->productDescription().friendly_product_type_name.empty());
    assert (!group->productDescription().module.module_label.empty());
    assert (!group->productDescription().module.process_name.empty());
    SharedGroupPtr g(group);

    BranchKey const bk = BranchKey(g->productDescription());
    //cerr << "addGroup DEBUG 2---> " << bk.friendly_class_name << endl;
    //cerr << "addGroup DEBUG 3---> " << bk << endl;


    if (labeled_dict_.find(bk) != labeled_dict_.end()) {
	// the products are lost at this point!
	throw edm::Exception(edm::errors::InsertFailure,"AlreadyPresent")
	  << "addGroup: Problem found while adding product provanence, "
	  << "product already exists for ("
	  << bk.friendly_class_name << ","
          << bk.module_label << ","
          << bk.product_instance_name << ","
          << bk.process_name
	  << ")";
    }

    // a memory allocation failure in modifying the product
    // data structures will cause things to be out of sync
    // we do not have any rollback capabilities as products 
    // and the indices are updated

    groups_.push_back(g);

    unsigned long slotNumber = g->productDescription().product_id.id_;

    labeled_dict_[bk] = slotNumber;

    //cerr << "addGroup DEBUG 4---> " << bk.friendly_class_name << endl;

    vector<int>& vint = type_dict_[bk.friendly_class_name];

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
    assert (it != pl.end());
    prov->product.product_id = it->second.product_id;
    ProductID id = it->second.product_id;
    // Group assumes ownership
    auto_ptr<Group> g(new Group(edp, prov));
    g->setID(id);
    this->addGroup(g);
  }

  BasicHandle
  EventPrincipal::get(ProductID oid) const {
    if (oid == ProductID())
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "Event::get by product ID: invalid ProductID supplied";

    unsigned long slotNumber = oid.id_;
    if (slotNumber >= groups_.size())
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "Event::get by product ID: no product with given id";

    SharedGroupPtr const& g = groups_[slotNumber];
    this->resolve_(*g);
    return BasicHandle(g->product(), &g->provenance());
  }

  BasicHandle
  EventPrincipal::getBySelector(TypeID id, 
				Selector const& sel) const {
    TypeDict::const_iterator i = type_dict_.find(id.friendlyClassName());

    if(i==type_dict_.end()) {
	// TODO: Perhaps stuff like this should go to some error
	// logger?  Or do we want huge message inside the exception
	// that is thrown?
	edm::Exception err(edm::errors::ProductNotFound,"InvalidType");
	err << "Event::getBySelector: no products found of correct type\n";
	err << "No products found of correct type\n";
	err << "We are looking for: '"
	     << id
	     << "'\n";
	if (type_dict_.empty()) {
	    err << "type_dict_ is empty!\n";
	} else {
	    err << "We found only the following:\n";
	    TypeDict::const_iterator i = type_dict_.begin();
	    TypeDict::const_iterator e = type_dict_.end();
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
  EventPrincipal::getByLabel(TypeID id, 
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
	string const& process_name = *iproc;
	BranchKey bk(id, label, productInstanceName, process_name);
	BranchDict::const_iterator i = labeled_dict_.find(bk);

	if (i != labeled_dict_.end()) {
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
      << " with user tag " << (productInstanceName.empty() ? "\"\"" : productInstanceName);
  }

  void 
  EventPrincipal::getMany(TypeID id, 
			  Selector const& sel,
			  BasicHandleVec& results) const {
    // We make no promise that the input 'fill_me_up' is unchanged if
    // an exception is thrown. If such a promise is needed, then more
    // care needs to be taken.
    TypeDict::const_iterator i = type_dict_.find(id.friendlyClassName());

    if(i==type_dict_.end()) {
	throw edm::Exception(errors::ProductNotFound,"NoMatch")
	  << "getMany: no products found of correct type " << id;
    }

    vector<int> const& vint = i->second;

    if(vint.empty()) {
	// should never happen!!
	throw edm::Exception(edm::errors::ProductNotFound,"EmptyList")
	  <<  "getMany: no products found for\n"
	  << id;
    }

    vector<int>::const_iterator ib(vint.begin()),ie(vint.end());
    while(ib!=ie) {
	SharedGroupPtr const& g = groups_[*ib];
	if(sel.match(g->provenance())) {
	    this->resolve_(*g);
	    results.push_back(BasicHandle(g->product(), &g->provenance()));
	}
	++ib;
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
