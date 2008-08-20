#ifndef Streamer_StreamedProducts_h
#define Streamer_StreamedProducts_h

/*
  Simple packaging of all the event data that is needed to be serialized
  for transfer.

  The "other stuff in the SendEvent still needs to be
  populated.

  The product is paired with its provenance, and the entire event
  is captured in the SendEvent structure.
 */

#include <vector>
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

namespace edm {

  class EDProduct;
  // ------------------------------------------

  class StreamedProduct {
  public:
    StreamedProduct() :
      prod_(0), desc_(0), mod_(), productID_(), status_(), parents_(0) {}
    // explicit StreamedProduct(EDProduct const* p) : prod_(p), desc_(), productID_(), status_(), parents_(0) {}
    explicit StreamedProduct(BranchDescription const& desc) :
	prod_(0), desc_(&desc), mod_(), productID_(), status_(productstatus::neverCreated()), parents_(0) {}

    StreamedProduct(EDProduct const* prod,
		    BranchDescription const& desc,
		    ModuleDescriptionID const& mod,
		    ProductID pid,
		    ProductStatus status,
		    std::vector<BranchID> const* parents) :
      prod_(prod), desc_(&desc), mod_(mod), productID_(pid), status_(status), parents_(parents) {}

    EDProduct const* prod() const {return prod_;}
    BranchDescription const* desc() const {return desc_;}
    ModuleDescriptionID const& mod() const {return mod_;}
    BranchID branchID() const {return desc_->branchID();}
    ProductID productID() const {return productID_;}
    ProductStatus status() const {return status_;}
    std::vector<BranchID> const* parents() const {return parents_;}

   void clear() {
     prod_= 0;
     delete desc_;
     desc_= 0;
     productID_ = ProductID();
     status_ = 0;
     delete parents_;
     parents_ = 0;
  }

  private:
    EDProduct const* prod_;
    BranchDescription const* desc_;
    ModuleDescriptionID  mod_;
    ProductID productID_;
    ProductStatus status_;
    std::vector<BranchID> const* parents_;
  };

  // ------------------------------------------

  typedef std::vector<StreamedProduct> SendProds;

  // ------------------------------------------

  class SendEvent {
  public:
    SendEvent() { }
    SendEvent(EventAuxiliary const& aux, ProcessHistory const& processHistory) :
	aux_(aux), processHistory_(processHistory), products_() {}
    EventAuxiliary const& aux() const {return aux_;}
    SendProds const& products() const {return products_;}
    ProcessHistory const& processHistory() const {return processHistory_;}
    SendProds & products() {return products_;}
  private:
    EventAuxiliary aux_;
    ProcessHistory processHistory_;
    SendProds products_;

    // other tables necessary for provenance lookup
  };

  typedef std::vector<BranchDescription> SendDescs;

  class SendJobHeader {
  public:
    typedef std::map<ParameterSetID, ParameterSetBlob> ParameterSetMap;
    SendJobHeader() { }
    SendDescs const& descs() const {return descs_;}
    ParameterSetMap const& processParameterSet() const {return processParameterSet_;}
    ModuleDescriptionMap const& moduleDescriptionMap() const {return moduleDescriptionMap_;}
    void push_back(BranchDescription const& bd) {descs_.push_back(bd);}
    void setModuleDescriptionMap(ModuleDescriptionMap const& mdMap) {moduleDescriptionMap_ = mdMap;}
    void setParameterSetMap(ParameterSetMap const& psetMap) {processParameterSet_ = psetMap;}

  private:
    SendDescs descs_;
    ParameterSetMap processParameterSet_;
    ModuleDescriptionMap moduleDescriptionMap_;
    // trigger bit descriptions will be added here and permanent
    //  provenance values
  };


}
#endif

