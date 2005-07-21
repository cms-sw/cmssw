/*----------------------------------------------------------------------
$Id: EventPrincipal.cc,v 1.13 2005/07/14 22:50:53 wmtan Exp $
----------------------------------------------------------------------*/
//#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace std;

namespace
{
  const unsigned long initial_size = 200;  // optimization guess...
} 

namespace edm {

  EventPrincipal::EventPrincipal() :
    aux_(),
    groups_(),
    labeled_dict_(),
    type_dict_(),
    store_(0)
  {
    groups_.reserve(initial_size);
  }

  EventPrincipal::EventPrincipal(const CollisionID& id,
				 Retriever& r, const ProcessNameList& nl) :
    aux_(id),
    groups_(),
    labeled_dict_(),
    type_dict_(),
    store_(&r)
  {
    aux_.process_history_ = nl;
    groups_.reserve(initial_size);
  }
 
  EventPrincipal::~EventPrincipal()
  {
  }

  CollisionID
  EventPrincipal::id() const
  {
    return aux_.id_;
  }

  void 
  EventPrincipal::addGroup(auto_ptr<Group> group)
  {
    assert (!group->provenance()->product.full_product_type_name.empty());
    assert (!group->provenance()->product.friendly_product_type_name.empty());
    assert (!group->provenance()->product.module.module_label.empty());
    assert (!group->provenance()->product.module.process_name.empty());
    SharedGroupPtr g(group);

    BranchKey bk(*g->provenance());
    //cerr << "addGroup DEBUG 2---> " << bk.friendly_class_name << endl;
    //cerr << "addGroup DEBUG 3---> " << bk << endl;


    if (labeled_dict_.find(bk) != labeled_dict_.end())
      {
        string class_name = g->provenance()->product.friendly_product_type_name;
        string module_label = g->provenance()->product.module.module_label;
        string product_instance_name = g->provenance()->product.product_instance_name;
        string process_name = g->provenance()->product.module.process_name;
	// the products are lost at this point!
	throw edm::Exception(edm::errors::InsertFailure,"AlreadyPresent")
	  << "addGroup: Problem found while adding product provanence, "
	  << "product already exists for ("
	  << class_name << ","
          << module_label << ","
          << product_instance_name << ","
          << process_name
	  << ")";
      }

    // a memory allocation failure in modifying the product
    // data structures will cause things to be out of sync
    // we do not have any rollback capabilities as products 
    // and the indices are updated

    groups_.push_back(g);
    // The ID we assign is the size *after* the Group has been pushed
    // into the vector. This means the Group in slot 0 has ID 1, etc.
    unsigned long slotNumber = g->provenance()->product_id - 1;

    labeled_dict_[bk] = slotNumber;

    //cerr << "addGroup DEBUG 4---> " << bk.friendly_class_name << endl;

    vector<int>& vint = type_dict_[bk.friendly_class_name];

    vint.push_back(slotNumber);
  }

  void
  EventPrincipal::addToProcessHistory(const string& processName)
  {
    aux_.process_history_.push_back(processName);
  }

  const ProcessNameList&
  EventPrincipal::processHistory() const
  {
    return aux_.process_history_;
  }

  void 
  EventPrincipal::put(auto_ptr<EDProduct> edp,
		      auto_ptr<Provenance> prov)
  {
    // Group assumes ownership
    auto_ptr<Group> g(new Group(edp, prov));
    // The ID we assign is the size *after* the Group has been pushed
    // into the vector. This means the Group in slot 0 has ID 1, etc.
    GroupVec::size_type sz = groups_.size()+1;
    g->setID(sz);
    this->addGroup(g);
  }

  BasicHandle
  EventPrincipal::get(EDP_ID oid) const
  {
    if (oid == EDP_ID())
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "Event::get by product ID: invalid EDP_ID supplied";

    // Remember that the object in slot 0 has EDP_ID of 1.
    unsigned long slotNumber = oid-1;
    if (slotNumber >= groups_.size())
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "Event::get by product ID: no product with given id";

    const SharedGroupPtr& g = groups_[slotNumber];
    this->resolve_(*g);
    return BasicHandle(g->product(), g->provenance());
  }

  BasicHandle
  EventPrincipal::getBySelector(TypeID id, 
				const Selector& sel) const
  {
    TypeDict::const_iterator i = type_dict_.find(id.friendlyClassName());

    if(i==type_dict_.end())
      {
	// TODO: Perhaps stuff like this should go to some error
	// logger?  Or do we want huge message inside the exception
	// that is thrown?
	edm::Exception err(edm::errors::ProductNotFound,"InvalidType");
	err << "Event::getBySelector: no products found of correct type\n";
	err << "No products found of correct type\n";
	err << "We are looking for: '"
	     << id
	     << "'\n";
	if (type_dict_.empty())
	  {
	    err << "type_dict_ is empty!\n";
	  }
	else
	  {
	    err << "We found only the following:\n";
	    TypeDict::const_iterator i = type_dict_.begin();
	    TypeDict::const_iterator e = type_dict_.end();
	    while (i != e)
	      {
		err << "...\t" << i->first << '\n';
		++i;
	      }
	  }
	err << ends;
	throw err;
      }

    const vector<int>& vint = i->second;

    if (vint.empty())
      {
	// should never happen!!
	throw edm::Exception(edm::errors::ProductNotFound,"EmptyList")
	  <<  "getBySelector: no products found for\n"
	  << id;
      }

    int found_count = 0;
    int found_slot = -1; // not a legal value!
    vector<int>::const_iterator ib(vint.begin()),ie(vint.end());

    BasicHandle result;

    while(ib!=ie)
      {
	const SharedGroupPtr& g = groups_[*ib];
	if(sel.match(*g->provenance()))
	  {
	    ++found_count;
	    if (found_count > 1)
	      {
		throw edm::Exception(edm::errors::ProductNotFound,
				     "TooManyMatches")
		  << "getBySelector: too many products found, "
		  << "expected one, got " << found_count << ", for\n"
		  << id;
	      }
	    found_slot = *ib;
	    this->resolve_(*g);
	    result = BasicHandle(g->product(), g->provenance());
	  }
	++ib;
      }

    if (found_count == 0)
      {
	throw edm::Exception(edm::errors::ProductNotFound,"TooFewProducts")
	  << "getBySelector: too few products found (zero) for\n"
	  << id;
      }

    return result;
  }

    
  BasicHandle
  EventPrincipal::getByLabel(TypeID id, 
			     const string& label,
			     const string& productInstanceName) const
  {
    // The following is not the most efficient way of doing this. It
    // is the simplest implementation of the required policy, given
    // the current organization of the EventPrincipal. This should be
    // reviewed.

    // THE FOLLOWING IS A HACK! It must be removed soon, with the
    // correct policy of making the assumed label be ... whatever we
    // set the policy to be. I don't know the answer right now...

    ProcessNameList::const_reverse_iterator iproc = aux_.process_history_.rbegin();
    ProcessNameList::const_reverse_iterator eproc = aux_.process_history_.rend();
    while (iproc != eproc)
      {
	const string& process_name = *iproc;
	BranchKey bk(id, label, productInstanceName, process_name);
	BranchDict::const_iterator i = labeled_dict_.find(bk);

	if (i != labeled_dict_.end())
	  {
	    // We found what we want.
            assert(i->second >= 0);
            assert(unsigned(i->second) < groups_.size());
	    SharedGroupPtr group = groups_[i->second];
	    this->resolve_(*group);
	    return BasicHandle(group->product(), group->provenance());    
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
			  const Selector& sel,
			  BasicHandleVec& results) const
  {
    // We make no promise that the input 'fill_me_up' is unchanged if
    // an exception is thrown. If such a promise is needed, then more
    // care needs to be taken.
    TypeDict::const_iterator i = type_dict_.find(id.friendlyClassName());

    if(i==type_dict_.end())
      {
	throw edm::Exception(errors::ProductNotFound,"NoMatch")
	  << "getMany: no products found of correct type " << id;
      }

    const vector<int>& vint = i->second;

    if(vint.empty())
      {
	// should never happen!!
	throw edm::Exception(edm::errors::ProductNotFound,"EmptyList")
	  <<  "getMany: no products found for\n"
	  << id;
      }

    vector<int>::const_iterator ib(vint.begin()),ie(vint.end());
    while(ib!=ie)
      {
	const SharedGroupPtr& g = groups_[*ib];
	if(sel.match(*g->provenance())) 
	  {
	    this->resolve_(*g);
	    results.push_back(BasicHandle(g->product(), g->provenance()));
	  }
	++ib;
      }
  }

  void
  EventPrincipal::resolve_(const Group& g) const
  {
    if (!g.isAccessible())
      throw edm::Exception(errors::ProductNotFound,"InaccessibleProduct")
	<< "resolve_: product is not accessible\n"
	<< g.provenance();

    if (g.product()) return; // nothing to do.
    
    // must attempt to load from persistent store
    BranchKey bk(*g.provenance());
    auto_ptr<EDProduct> edp(store_->get(bk));

    // Now fixup the Group
    g.setProduct(edp);
  }
  

}
