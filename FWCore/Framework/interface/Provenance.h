#ifndef PROVENANCE_HH
#define PROVENANCE_HH

/*----------------------------------------------------------------------
  
Provenance: The full description of a product and how it came into
existence.

$Id: Provenance.h,v 1.7 2005/07/06 20:26:01 wmtan Exp $
----------------------------------------------------------------------*/
#include <ostream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/ConditionsID.h"
#include "FWCore/EDProduct/interface/EDP_ID.h"
#include "FWCore/Framework/interface/ModuleDescription.h"

/*
  Provenance

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Creator: The EDProducer that made the product.

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {
  // these parts of a provenance are used to compose a branch name
  struct BranchName
  {
    std::string module_label;
    std::string process_name;
    std::string friendly_product_type_name;
  };

  struct Provenance
  {
    enum CreatorStatus { Success = 0,
			 ApplicationFailure,
			 InfrastructureFailure };

    Provenance();
    explicit Provenance(const ModuleDescription& m);

    ~Provenance() {}

    ModuleDescription module;
    EDP_ID product_id;

    // The EDProduct IDs of the parents
    std::vector<EDP_ID> parents;

    // a single identifier that describes all the conditions used
    ConditionsID cid; // frame ID?

    // the full name of the type of product this is
    std::string full_product_type_name;

    // a readable name of the type of product this is
    std::string friendly_product_type_name;

    // a user-supplied name to distinguish multiple products of the same type
    // that are produced by the same producer
    std::string product_instance_name;
    // the last of these is not in the roadmap, but is on the board

    // if modules can or will place an object in the event
    // even though something not good occurred, like a timeout, then
    // this may be useful - or if the I/O system makes blank or default
    // constructed objects and we need to distinguish between zero
    // things in a collection between nothing was found and the case
    // where a failure caused nothing to be in the collection.
    // Should a provenance be inserted even if a module fails to 
    // create the output it promised?
    CreatorStatus status;


    void write(std::ostream& os) const;
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, const Provenance& p)
  {
    p.write(os);
    return os;
  }
  


//  this is broken gunk - it may not be needed at all
//
//   bool operator<(const Provenance& a, const Provenance& b)
//   {
//     // key order: product_type_name -> module_label -> process_name -> pass_
//     return
//       a.full_product_type_name < b.full_product_type_name ? true :
//       (a.full_product_type_name == b.full_product_type_name ?
//        (a.module_label < b.module_label ? true : 
// 	(a.module_label == b.module_label ?
// 	 (a.process_name < b.process_name ? true :
// 	  (a.process_name == b.process_name ?
// 	   (a.pass < b.pass ? true : false
//) : false
//) : false
//) : false
//) : false
//) : false
//)
//       ;
//   }
}
#endif
