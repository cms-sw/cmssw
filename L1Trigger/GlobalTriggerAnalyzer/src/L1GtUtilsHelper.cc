#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtilsHelper.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

L1GtUtilsHelper::L1GtUtilsHelper(edm::ParameterSet const& pset, edm::ConsumesCollector& iC, bool useL1GtTriggerMenuLite)
    : m_l1GtRecordInputTag(pset.getParameter<edm::InputTag>("l1GtRecordInputTag")),
      m_l1GtReadoutRecordInputTag(pset.getParameter<edm::InputTag>("l1GtReadoutRecordInputTag")),
      m_l1GtTriggerMenuLiteInputTag(pset.getParameter<edm::InputTag>("l1GtTriggerMenuLiteInputTag")) {
  m_l1GtRecordToken = iC.consumes<L1GlobalTriggerRecord>(m_l1GtRecordInputTag);
  m_l1GtReadoutRecordToken = iC.consumes<L1GlobalTriggerReadoutRecord>(m_l1GtReadoutRecordInputTag);
  if (useL1GtTriggerMenuLite) {
    m_l1GtTriggerMenuLiteToken = iC.consumes<L1GtTriggerMenuLite, edm::InRun>(m_l1GtTriggerMenuLiteInputTag);
  }
}

void L1GtUtilsHelper::fillDescription(edm::ParameterSetDescription& desc) {
  desc.add<edm::InputTag>("l1GtRecordInputTag", edm::InputTag());
  desc.add<edm::InputTag>("l1GtReadoutRecordInputTag", edm::InputTag());
  desc.add<edm::InputTag>("l1GtTriggerMenuLiteInputTag", edm::InputTag());
}

namespace {
  template <typename C, typename T>
  void setConsumesAndCheckAmbiguities(edm::BranchDescription const& iDesc,
                                      C const& iPreferredTags,
                                      T& ioToken,
                                      edm::InputTag& ioTag,
                                      edm::ConsumesCollector& iCollector,
                                      const char* iTypeForErrorMessage) {
    if (ioTag.label().empty()) {
      //hasn't been set yet
      ioTag = edm::InputTag{iDesc.moduleLabel(), iDesc.productInstanceName(), iDesc.processName()};

      ioToken = iCollector.consumes(ioTag);
      LogDebug("L1GtUtils")
          << "Input tag found for " << iTypeForErrorMessage << " product.\n Input tag set to " << (ioTag) << "\n Tag is"
          << ((iPreferredTags.end() != std::find(iPreferredTags.begin(), iPreferredTags.end(), ioTag.label())) ? ""
                                                                                                               : " not")
          << " found in preferred tags list " << std::endl;

    } else {
      bool alreadyFoundPreferred =
          iPreferredTags.end() != std::find(iPreferredTags.begin(), iPreferredTags.end(), ioTag.label());
      if (alreadyFoundPreferred) {
        if (std::find(iPreferredTags.begin(), iPreferredTags.end(), iDesc.moduleLabel()) != iPreferredTags.end()) {
          edm::LogError("L1GtUtils") << "Found multiple preferred input tags for " << iTypeForErrorMessage
                                     << " product, "
                                     << "\nwith different instaces or processes."
                                     << "\nTag already found: " << (ioTag) << "\nOther tag: "
                                     << (edm::InputTag{
                                            iDesc.moduleLabel(), iDesc.productInstanceName(), iDesc.processName()})
                                     << "\nToken set to invalid." << std::endl;
          //another preferred also found
          ioToken = T{};
        }
      } else {
        //previous choice was not preferred

        auto itFound = std::find(iPreferredTags.begin(), iPreferredTags.end(), iDesc.moduleLabel());
        if (itFound != iPreferredTags.end()) {
          //reset to preferred
          auto oldTag = ioTag;
          ioTag = edm::InputTag{iDesc.moduleLabel(), iDesc.productInstanceName(), iDesc.processName()};

          ioToken = iCollector.consumes(ioTag);
          edm::LogWarning("L1GtUtils") << "Found preferred tag " << (ioTag) << "\n after having set unpreferred tag ("
                                       << oldTag << ") for " << iTypeForErrorMessage
                                       << ".\n Please change configuration to explicitly use the tag given above.\n "
                                          "This will avoid unnecessary prefetching of data not used.";
        } else {
          //hit an ambiguity
          edm::LogWarning("L1GtUtils") << "Found multiple input tags for " << iTypeForErrorMessage << " product."
                                       << "\nNone is in the preferred input tags - no safe choice."
                                       << "\nTag already found: " << (ioTag) << "\nOther tag: "
                                       << (edm::InputTag{
                                              iDesc.moduleLabel(), iDesc.productInstanceName(), iDesc.processName()})
                                       << "\nToken set to invalid." << std::endl;
          ioToken = T{};
        }
      }
    }
  }
}  // namespace
void L1GtUtilsHelper::checkToUpdateTags(edm::BranchDescription const& branchDescription,
                                        edm::ConsumesCollector consumesCollector,
                                        bool findRecord,
                                        bool findReadoutRecord,
                                        bool findMenuLite) {
  if (branchDescription.dropped()) {
    return;
  }
  // This is only used if required InputTags were not specified already.
  // This is called early in the process, once for each product in the ProductRegistry.
  // The callback is registered when callWhenNewProductsRegistered is called.
  // It finds products by type and sets the token so that it can be used
  // later when getting the product.

  // The code will look for the corresponding product in ProductRegistry.
  // If the product is found, it checks the product label in
  // a vector of preferred input tags (hardwired now to "gtDigis" and
  // "hltGtDigis"). The first input tag from the vector of preferred input tags, with the
  // same label as the input tag found from provenance, is kept as input tag, if there are no
  // multiple products with the same label.

  // If multiple products are found and no one has a label in the vector of preferred input tags,
  // or if multiple products are found with the label in the vector of preferred input tags
  // (with different instance or process) the input tag is set to empty input tag, and L1GtUtil
  // will produce an error, as it is not possible to safely choose a product. In this case, one must
  // provide explicitly the correct input tag via configuration or in the constructor.

  // TODO decide if the preferred input tags must be given as input parameters
  // or stay hardwired

  std::vector<edm::InputTag> preferredL1GtRecordInputTag = {edm::InputTag("gtDigis"), edm::InputTag("hltGtDigis")};

  std::vector<edm::InputTag> preferredL1GtReadoutRecordInputTag = {edm::InputTag("gtDigis"),
                                                                   edm::InputTag("hltGtDigis")};

  std::vector<edm::InputTag> preferredL1GtTriggerMenuLiteInputTag = {edm::InputTag("gtDigis"),
                                                                     edm::InputTag("hltGtDigis")};

  // L1GlobalTriggerRecord

  if (findRecord && (branchDescription.unwrappedTypeID() == edm::TypeID(typeid(L1GlobalTriggerRecord))) &&
      (branchDescription.branchType() == edm::InEvent)) {
    setConsumesAndCheckAmbiguities(branchDescription,
                                   preferredL1GtRecordInputTag,
                                   m_l1GtRecordToken,
                                   m_l1GtRecordInputTag,
                                   consumesCollector,
                                   "L1GlobalTriggerRecord");
  }

  // L1GlobalTriggerReadoutRecord

  if (findReadoutRecord && (branchDescription.unwrappedTypeID() == edm::TypeID(typeid(L1GlobalTriggerReadoutRecord))) &&
      (branchDescription.branchType() == edm::InEvent)) {
    setConsumesAndCheckAmbiguities(branchDescription,
                                   preferredL1GtReadoutRecordInputTag,
                                   m_l1GtReadoutRecordToken,
                                   m_l1GtReadoutRecordInputTag,
                                   consumesCollector,
                                   "L1GlobalTriggerReadoutRecord");
  }

  // L1GtTriggerMenuLite

  if (findMenuLite && (branchDescription.unwrappedTypeID() == edm::TypeID(typeid(L1GtTriggerMenuLite))) &&
      (branchDescription.branchType() == edm::InEvent)) {
    setConsumesAndCheckAmbiguities(branchDescription,
                                   preferredL1GtTriggerMenuLiteInputTag,
                                   m_l1GtTriggerMenuLiteToken,
                                   m_l1GtTriggerMenuLiteInputTag,
                                   consumesCollector,
                                   "L1GtTriggerMenuLite");
  }
}
