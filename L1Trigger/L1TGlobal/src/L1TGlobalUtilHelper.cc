#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtilHelper.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

l1t::L1TGlobalUtilHelper::L1TGlobalUtilHelper(edm::ParameterSet const& pset, edm::ConsumesCollector& iC)
    : m_l1tAlgBlkInputTag(pset.getParameter<edm::InputTag>("l1tAlgBlkInputTag")),
      m_l1tExtBlkInputTag(pset.getParameter<edm::InputTag>("l1tExtBlkInputTag")),
      m_readPrescalesFromFile(pset.getParameter<bool>("ReadPrescalesFromFile")) {
  m_l1tAlgBlkToken = iC.consumes<GlobalAlgBlkBxCollection>(m_l1tAlgBlkInputTag);
  m_l1tExtBlkToken = iC.consumes<GlobalExtBlkBxCollection>(m_l1tExtBlkInputTag);
}

void l1t::L1TGlobalUtilHelper::fillDescription(edm::ParameterSetDescription& desc,
                                               edm::InputTag const& iAlg,
                                               edm::InputTag const& iExt,
                                               bool readPrescalesFromFile) {
  desc.add<edm::InputTag>("l1tAlgBlkInputTag", iAlg);
  desc.add<edm::InputTag>("l1tExtBlkInputTag", iExt);
  desc.add<bool>("ReadPrescalesFromFile", readPrescalesFromFile);
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
          throw cms::Exception("L1GtUtils::TooManyChoices")
              << "Found multiple preferred input tags for " << iTypeForErrorMessage << " product, "
              << "\nwith different instaces or processes."
              << "\nTag already found: " << (ioTag) << "\nOther tag: "
              << (edm::InputTag{iDesc.moduleLabel(), iDesc.productInstanceName(), iDesc.processName()});
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

void l1t::L1TGlobalUtilHelper::checkToUpdateTags(edm::BranchDescription const& branchDescription,
                                                 edm::ConsumesCollector consumesCollector,
                                                 bool findL1TAlgBlk,
                                                 bool findL1TExtBlk) {
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

  if (branchDescription.dropped()) {
    return;
  }

  std::vector<edm::InputTag> preferredL1TAlgBlkInputTag = {edm::InputTag("gtStage2Digis"),
                                                           edm::InputTag("hltGtStage2Digis")};

  std::vector<edm::InputTag> preferredL1TExtBlkInputTag = {edm::InputTag("gtStage2Digis"),
                                                           edm::InputTag("hltGtStage2Digis")};

  // GlobalAlgBlkBxCollection

  if (findL1TAlgBlk && (branchDescription.unwrappedTypeID() == edm::TypeID(typeid(GlobalAlgBlkBxCollection))) &&
      (branchDescription.branchType() == edm::InEvent)) {
    setConsumesAndCheckAmbiguities(branchDescription,
                                   preferredL1TAlgBlkInputTag,
                                   m_l1tAlgBlkToken,
                                   m_l1tAlgBlkInputTag,
                                   consumesCollector,
                                   "GlobalAlgBlkBxCollection");
  }

  // GlobalExtBlkBxCollection

  if (findL1TExtBlk && (branchDescription.unwrappedTypeID() == edm::TypeID(typeid(GlobalExtBlkBxCollection))) &&
      (branchDescription.branchType() == edm::InEvent)) {
    setConsumesAndCheckAmbiguities(branchDescription,
                                   preferredL1TExtBlkInputTag,
                                   m_l1tExtBlkToken,
                                   m_l1tExtBlkInputTag,
                                   consumesCollector,
                                   "GlobalExtBlkBxCollection");
  }
}
