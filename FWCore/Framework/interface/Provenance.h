#ifndef PROVENANCE_HH
#define PROVENANCE_HH

/*----------------------------------------------------------------------
  
Provenance: The full description of a product and how it came into
existence.

$Id: Provenance.h,v 1.3 2005/06/07 22:42:45 wmtan Exp $
----------------------------------------------------------------------*/
#include <ostream>
#include <string>
#include <vector>

#include "FWCore/CoreFramework/interface/ConditionsID.h"
#include "FWCore/EDProduct/interface/EDP_ID.h"
#include "FWCore/CoreFramework/interface/PassID.h"
#include "FWCore/CoreFramework/interface/PS_ID.h"
#include "FWCore/CoreFramework/interface/VersionNumber.h"

/*
  Provenance

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Creator: The EDProducer that made the product.

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {

  // once a module is born, these parts of the module's product provenance
  // are constant   (change to ModuleDescription)
  struct ModuleDescription
  {
    // ID of parameter set of the creator
    PS_ID pid;

    // The class name of the creator
    std::string module_name;    

    // A human friendly string that uniquely identifies the EDProducer
    // and becomes part of the identity of a product that it produces
    std::string module_label;

    // the release tag of the executable
    VersionNumber version_number;

    // the physical process that this program was part of (e.g. production)
    std::string process_name;

    // what the heck is this? I think its the version of the process_name
    // e.g. second production pass
    PassID pass;
  };

  inline
  bool 
  operator==(const ModuleDescription& a, const ModuleDescription& b)
  {
    return 
      a.pid==b.pid
      && a.module_name==b.module_name
      && a.module_label==b.module_label 
      && a.version_number==b.version_number
      && a.process_name==b.process_name
      && a.pass==b.pass;
  } 

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
  operator<< (std::ostream& os, const Provenance& p)
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
