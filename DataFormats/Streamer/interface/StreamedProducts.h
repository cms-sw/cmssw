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
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"

#include "TClassRef.h"

namespace edm {

  // ------------------------------------------

  class StreamedProduct {
  public:
    StreamedProduct() : desc_(0), present_(false), parents_(0), prod_(0), classRef_() {}
    explicit StreamedProduct(BranchDescription const& desc) :
      desc_(&desc), present_(false), parents_(0), prod_(0), classRef_() {}

    StreamedProduct(void const* prod,
                    BranchDescription const& desc,
                    bool present,
                    std::vector<BranchID> const* parents);

    BranchDescription const* desc() const {return desc_;}
    BranchID branchID() const {return desc_->branchID();}
    bool present() const {return present_;}
    std::vector<BranchID> const* parents() const {return parents_;}
    void* prod() {return prod_;}
    TClassRef const& classRef() const {return classRef_;}
    void allocateForReading();
    void setNewClassType();
    void clearClassType();

   void clear() {
     prod_= 0;
     delete desc_;
     desc_= 0;
     present_ = false;
     delete parents_;
     parents_ = 0;
  }

  private:
    BranchDescription const* desc_;
    bool present_;
    std::vector<BranchID> const* parents_;
    void* prod_;
    TClassRef classRef_;
  };

  // ------------------------------------------

  typedef std::vector<StreamedProduct> SendProds;

  // ------------------------------------------

  class SendEvent {
  public:
    SendEvent() { }
    SendEvent(EventAuxiliary const& aux,
              ProcessHistory const& processHistory,
              EventSelectionIDVector const& eventSelectionIDs,
              BranchListIndexes const& branchListIndexes) :
        aux_(aux),
        processHistory_(processHistory),
        eventSelectionIDs_(eventSelectionIDs),
        branchListIndexes_(branchListIndexes),
        products_() {}
    EventAuxiliary const& aux() const {return aux_;}
    SendProds const& products() const {return products_;}
    ProcessHistory const& processHistory() const {return processHistory_;}
    EventSelectionIDVector const& eventSelectionIDs() const {return eventSelectionIDs_;}
    BranchListIndexes const& branchListIndexes() const {return branchListIndexes_;}
    SendProds & products() {return products_;}
  private:
    EventAuxiliary aux_;
    ProcessHistory processHistory_;
    EventSelectionIDVector eventSelectionIDs_;
    BranchListIndexes branchListIndexes_;
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
    BranchIDLists const& branchIDLists() const {return branchIDLists_;}
    std::vector<ProcessConfiguration> const& processConfigurations() const {return processConfigurations_;}
    void push_back(BranchDescription const& bd) {descs_.push_back(bd);}
    void setParameterSetMap(ParameterSetMap const& psetMap) {processParameterSet_ = psetMap;}
    void setBranchIDLists(BranchIDLists const& bidlists) {branchIDLists_ = bidlists;}
    void setProcessConfigurations(std::vector<ProcessConfiguration> const& pcs) {processConfigurations_ = pcs;}

  private:
    SendDescs descs_;
    ParameterSetMap processParameterSet_;
    BranchIDLists branchIDLists_;
    std::vector<ProcessConfiguration> processConfigurations_;
    // trigger bit descriptions will be added here and permanent
    //  provenance values
  };


}
#endif

