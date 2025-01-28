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
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"

namespace edm {

  // ------------------------------------------

  class WrapperBase;
  class StreamedProduct {
  public:
    StreamedProduct() : prod_(nullptr), desc_(nullptr), present_(false), parents_(nullptr) {}
    explicit StreamedProduct(ProductDescription const& desc)
        : prod_(nullptr), desc_(&desc), present_(false), parents_(nullptr) {}

    StreamedProduct(WrapperBase const* prod,
                    ProductDescription const& desc,
                    bool present,
                    std::vector<BranchID> const* parents);

    ProductDescription const* desc() const { return desc_; }
    BranchID branchID() const { return desc_->branchID(); }
    bool present() const { return present_; }
    std::vector<BranchID> const* parents() const { return parents_; }
    WrapperBase const* prod() { return prod_; }

    void clear() {
      prod_ = nullptr;
      delete desc_;
      desc_ = nullptr;
      present_ = false;
      delete parents_;
      parents_ = nullptr;
    }

  private:
    WrapperBase const* prod_;
    ProductDescription const* desc_;
    bool present_;
    std::vector<BranchID> const* parents_;
  };

  // ------------------------------------------

  typedef std::vector<StreamedProduct> SendProds;

  // ------------------------------------------
  // Contains either Event data or meta data about an Event. The header of the
  // message contains which way an instance of this class is to be interpreted
  // via the use of EventMsgView::isEventMetaData()

  class SendEvent {
  public:
    SendEvent() {}
    SendEvent(EventAuxiliary const& aux,
              ProcessHistory const& processHistory,
              EventSelectionIDVector const& eventSelectionIDs,
              BranchListIndexes const& branchListIndexes,
              BranchIDLists const& branchIDLists,
              ThinnedAssociationsHelper const& thinnedAssociationsHelper,
              uint32_t metaDataChecksum)
        : aux_(aux),
          processHistory_(processHistory),
          eventSelectionIDs_(eventSelectionIDs),
          branchListIndexes_(branchListIndexes),
          branchIDLists_(branchIDLists),
          thinnedAssociationsHelper_(thinnedAssociationsHelper),
          products_(),
          metaDataChecksum_(metaDataChecksum) {}
    EventAuxiliary const& aux() const { return aux_; }
    SendProds const& products() const { return products_; }
    ProcessHistory const& processHistory() const { return processHistory_; }
    EventSelectionIDVector const& eventSelectionIDs() const { return eventSelectionIDs_; }
    BranchListIndexes const& branchListIndexes() const { return branchListIndexes_; }
    //This will only hold values for EventMetaData messages
    BranchIDLists const& branchIDLists() const { return branchIDLists_; }
    //This will only hold values for EventMetaData messages
    ThinnedAssociationsHelper const& thinnedAssociationsHelper() const { return thinnedAssociationsHelper_; }
    //This is the adler32 checksum of the EventMetaData associated with this Event
    uint32_t metaDataChecksum() const { return metaDataChecksum_; }
    SendProds& products() { return products_; }

  private:
    EventAuxiliary aux_;
    ProcessHistory processHistory_;
    EventSelectionIDVector eventSelectionIDs_;
    BranchListIndexes branchListIndexes_;
    BranchIDLists branchIDLists_;
    ThinnedAssociationsHelper thinnedAssociationsHelper_;
    SendProds products_;
    uint32_t metaDataChecksum_;

    // other tables necessary for provenance lookup
  };

  typedef std::vector<ProductDescription> SendDescs;

  class SendJobHeader {
  public:
    typedef std::map<ParameterSetID, ParameterSetBlob> ParameterSetMap;
    SendJobHeader() {}
    SendDescs const& descs() const { return descs_; }
    ParameterSetMap const& processParameterSet() const { return processParameterSet_; }
    void push_back(ProductDescription const& bd) { descs_.push_back(bd); }
    void setParameterSetMap(ParameterSetMap const& psetMap) { processParameterSet_ = psetMap; }
    void initializeTransients();

  private:
    SendDescs descs_;
    ParameterSetMap processParameterSet_;
  };

}  // namespace edm
#endif
