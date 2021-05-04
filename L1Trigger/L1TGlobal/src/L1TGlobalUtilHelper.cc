#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtilHelper.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

l1t::L1TGlobalUtilHelper::L1TGlobalUtilHelper(edm::ParameterSet const& pset, edm::ConsumesCollector& iC)
    : m_consumesCollector(iC),
      m_l1tAlgBlkInputTag(pset.getParameter<edm::InputTag>("l1tAlgBlkInputTag")),
      m_l1tExtBlkInputTag(pset.getParameter<edm::InputTag>("l1tExtBlkInputTag")),
      m_findL1TAlgBlk(false),
      m_findL1TExtBlk(false),
      m_readPrescalesFromFile(pset.getParameter<bool>("ReadPrescalesFromFile")),
      m_foundPreferredL1TAlgBlk(false),
      m_foundPreferredL1TExtBlk(false) {
  m_l1tAlgBlkToken = iC.consumes<GlobalAlgBlkBxCollection>(m_l1tAlgBlkInputTag);
  m_l1tExtBlkToken = iC.consumes<GlobalExtBlkBxCollection>(m_l1tExtBlkInputTag);
}

void l1t::L1TGlobalUtilHelper::fillDescription(edm::ParameterSetDescription& desc) {
  desc.add<edm::InputTag>("l1tAlgBlkInputTag", edm::InputTag());
  desc.add<edm::InputTag>("l1tExtBlkInputTag", edm::InputTag());
  desc.add<bool>("ReadPrescalesFromFile", false);
}

void l1t::L1TGlobalUtilHelper::operator()(edm::BranchDescription const& branchDescription) {
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

  if (m_findL1TAlgBlk && (!m_foundMultipleL1TAlgBlk) &&
      (branchDescription.unwrappedTypeID() == edm::TypeID(typeid(GlobalAlgBlkBxCollection))) &&
      (branchDescription.branchType() == edm::InEvent)) {
    edm::InputTag tag{
        branchDescription.moduleLabel(), branchDescription.productInstanceName(), branchDescription.processName()};

    if (m_foundPreferredL1TAlgBlk) {
      // check if a preferred input tag was already found and compare it with the actual tag
      // if the instance or the process names are different, one has incompatible tags - set
      // the tag to empty input tag and indicate that multiple preferred input tags are found
      // so it is not possibly to choose safely an input tag

      if ((m_l1tAlgBlkInputTag.label() == branchDescription.moduleLabel()) &&
          ((m_l1tAlgBlkInputTag.instance() != branchDescription.productInstanceName()) ||
           (m_l1tAlgBlkInputTag.process() != branchDescription.processName()))) {
        LogDebug("L1TGlobalUtil") << "\nWARNING: Found multiple preferred input tags for GlobalAlgBlkBxCollection, "
                                  << "\nwith different instaces or processes."
                                  << "\nInput tag already found: " << (m_l1tAlgBlkInputTag) << "\nActual tag: " << (tag)
                                  << "\nInput tag set to empty tag." << std::endl;

        m_foundMultipleL1TAlgBlk = true;
        m_l1tAlgBlkInputTag = edm::InputTag();
      }
    } else {
      // no preferred input tag found yet, check now with the actual tag
      for (std::vector<edm::InputTag>::const_iterator itPrefTag = preferredL1TAlgBlkInputTag.begin(),
                                                      itPrefTagEnd = preferredL1TAlgBlkInputTag.end();
           itPrefTag != itPrefTagEnd;
           ++itPrefTag) {
        if (branchDescription.moduleLabel() == itPrefTag->label()) {
          m_l1tAlgBlkInputTag = tag;
          m_l1tAlgBlkToken = m_consumesCollector.consumes<GlobalAlgBlkBxCollection>(tag);
          m_foundPreferredL1TAlgBlk = true;
          m_inputTagsL1TAlgBlk.push_back(tag);

          LogDebug("L1TGlobalUtil")
              << "\nWARNING: Input tag for GlobalAlgBlkBxCollection product set to preferred input tag" << (tag)
              << std::endl;
          break;
        }
      }
    }

    if (!m_foundPreferredL1TAlgBlk) {
      // check if other input tag was found - if true, there are multiple input tags in the event,
      // none in the preferred input tags, so it is not possibly to choose safely an input tag

      if (m_inputTagsL1TAlgBlk.size() > 1) {
        LogDebug("L1TGlobalUtil") << "\nWARNING: Found multiple input tags for GlobalAlgBlkBxCollection product."
                                  << "\nNone is in the preferred input tags - no safe choice."
                                  << "\nInput tag already found: " << (m_l1tAlgBlkInputTag) << "\nActual tag: " << (tag)
                                  << "\nInput tag set to empty tag." << std::endl;
        m_l1tAlgBlkInputTag = edm::InputTag();
        m_foundMultipleL1TAlgBlk = true;

      } else {
        if (m_l1tAlgBlkToken.isUninitialized()) {
          m_l1tAlgBlkInputTag = tag;
          m_inputTagsL1TAlgBlk.push_back(tag);
          m_l1tAlgBlkToken = m_consumesCollector.consumes<GlobalAlgBlkBxCollection>(tag);

          LogDebug("L1TGlobalUtil") << "\nWARNING: No preferred input tag found for GlobalAlgBlkBxCollection."
                                    << "\nInput tag set to " << (tag) << std::endl;
        }
      }
    }
  }

  // GlobalExtBlkBxCollection

  if (m_findL1TExtBlk && (!m_foundMultipleL1TExtBlk) &&
      (branchDescription.unwrappedTypeID() == edm::TypeID(typeid(GlobalExtBlkBxCollection))) &&
      (branchDescription.branchType() == edm::InEvent)) {
    edm::InputTag tag{
        branchDescription.moduleLabel(), branchDescription.productInstanceName(), branchDescription.processName()};

    if (m_foundPreferredL1TExtBlk) {
      // check if a preferred input tag was already found and compare it with the actual tag
      // if the instance or the process names are different, one has incompatible tags - set
      // the tag to empty input tag and indicate that multiple preferred input tags are found
      // so it is not possibly to choose safely an input tag

      if ((m_l1tExtBlkInputTag.label() == branchDescription.moduleLabel()) &&
          ((m_l1tExtBlkInputTag.instance() != branchDescription.productInstanceName()) ||
           (m_l1tExtBlkInputTag.process() != branchDescription.processName()))) {
        LogDebug("L1TGlobalUtil") << "\nWARNING: Found multiple preferred input tags for GlobalExtBlkBxCollection, "
                                  << "\nwith different instaces or processes."
                                  << "\nInput tag already found: " << (m_l1tExtBlkInputTag) << "\nActual tag: " << (tag)
                                  << "\nInput tag set to empty tag." << std::endl;

        m_foundMultipleL1TExtBlk = true;
        m_l1tExtBlkInputTag = edm::InputTag();
      }
    } else {
      // no preferred input tag found yet, check now with the actual tag

      for (std::vector<edm::InputTag>::const_iterator itPrefTag = preferredL1TExtBlkInputTag.begin(),
                                                      itPrefTagEnd = preferredL1TExtBlkInputTag.end();
           itPrefTag != itPrefTagEnd;
           ++itPrefTag) {
        if (branchDescription.moduleLabel() == itPrefTag->label()) {
          m_l1tExtBlkInputTag = tag;
          m_l1tExtBlkToken = m_consumesCollector.consumes<GlobalExtBlkBxCollection>(tag);
          m_foundPreferredL1TExtBlk = true;
          m_inputTagsL1TExtBlk.push_back(tag);

          LogDebug("L1TGlobalUtil")
              << "\nWARNING: Input tag for GlobalExtBlkBxCollection product set to preferred input tag" << (tag)
              << std::endl;
          break;
        }
      }
    }

    if (!m_foundPreferredL1TExtBlk) {
      // check if other input tag was found - if true, there are multiple input tags in the event,
      // none in the preferred input tags, so it is not possibly to choose safely an input tag

      if (m_inputTagsL1TExtBlk.size() > 1) {
        LogDebug("L1TGlobalUtil") << "\nWARNING: Found multiple input tags for GlobalExtBlkBxCollection."
                                  << "\nNone is in the preferred input tags - no safe choice."
                                  << "\nInput tag already found: " << (m_l1tExtBlkInputTag) << "\nActual tag: " << (tag)
                                  << "\nInput tag set to empty tag." << std::endl;
        m_l1tExtBlkInputTag = edm::InputTag();
        m_foundMultipleL1TExtBlk = true;

      } else {
        if (m_l1tExtBlkToken.isUninitialized()) {
          m_l1tExtBlkInputTag = tag;
          m_inputTagsL1TExtBlk.push_back(tag);
          m_l1tExtBlkToken = m_consumesCollector.consumes<GlobalExtBlkBxCollection>(tag);

          LogDebug("L1TGlobalUtil") << "\nWARNING: No preferred input tag found for GlobalExtBlkBxCollection product."
                                    << "\nInput tag set to " << (tag) << std::endl;
        }
      }
    }
  }
}
