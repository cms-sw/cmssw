#ifndef L1TGlobal_l1TGlobalUtilHelper_h
#define L1TGlobal_l1TGlobalUtilHelper_h

/**
 * \class L1TGlobalUtilHelper
 *
 * Does the same for L1TGlobalUtil as L1GtUtilsHelper does for L1GtUtils:
 * Description: Gets tokens for L1GtUtils to use when getting products
 *              from the Event and Run. This class was introduced
 *              when the consumes function calls were added for L1GtUtils.
 *              It preserves the special feature of L1GtUtils that allows
 *              it to run without configuration of InputTags, although it
 *              allows InputTags to be configured optionally or passed in
 *              via the constructor arguments.
 *
 * \author: W.David Dagenhart - Fermilab 30 April 2015
 *
 */

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <utility>

namespace edm {
  class BranchDescription;
  class ParameterSetDescription;
}  // namespace edm

namespace l1t {

  class L1TGlobalUtilHelper {
  public:
    // Using this constructor will require InputTags to be specified in the configuration
    L1TGlobalUtilHelper(edm::ParameterSet const& pset, edm::ConsumesCollector& iC);

    // Using this constructor will cause it to look for valid InputTags in
    // the following ways in the specified order until they are found.
    //   1. The configuration
    //   2. Search all products from the preferred input tags for the required type
    //   3. Search all products from any other process for the required type
    template <typename T>
    L1TGlobalUtilHelper(edm::ParameterSet const& pset, edm::ConsumesCollector& iC, T& module);

    // Using this constructor will cause it to look for valid InputTags in
    // the following ways in the specified order until they are found.
    //   1. The constructor arguments
    //   2. The configuration
    //   3. Search all products from the preferred input tags for the required type
    //   4. Search all products from any other process for the required type
    template <typename T>
    L1TGlobalUtilHelper(edm::ParameterSet const& pset,
                        edm::ConsumesCollector& iC,
                        T& module,
                        edm::InputTag const& l1tAlgBlkInputTag,
                        edm::InputTag const& l1tExtBlkInputTag);

    // A module defining its fillDescriptions function might want to use this
    static void fillDescription(edm::ParameterSetDescription& desc);

    // Callback which will be registered with the Framework if the InputTags
    // are not specified in the configuration or constructor arguments. It
    // will get called for each product in the ProductRegistry.
    void operator()(edm::BranchDescription const& branchDescription);

    edm::InputTag const& l1tAlgBlkInputTag() const { return m_l1tAlgBlkInputTag; }
    edm::InputTag const& l1tExtBlkInputTag() const { return m_l1tExtBlkInputTag; }

    bool const& readPrescalesFromFile() const { return m_readPrescalesFromFile; }

    edm::EDGetTokenT<GlobalAlgBlkBxCollection> const& l1tAlgBlkToken() const { return m_l1tAlgBlkToken; }
    edm::EDGetTokenT<GlobalExtBlkBxCollection> const& l1tExtBlkToken() const { return m_l1tExtBlkToken; }

  private:
    edm::ConsumesCollector m_consumesCollector;

    edm::InputTag m_l1tAlgBlkInputTag;
    edm::InputTag m_l1tExtBlkInputTag;

    edm::EDGetTokenT<GlobalAlgBlkBxCollection> m_l1tAlgBlkToken;
    edm::EDGetTokenT<GlobalExtBlkBxCollection> m_l1tExtBlkToken;

    bool m_findL1TAlgBlk;
    bool m_findL1TExtBlk;

    bool m_readPrescalesFromFile;

    bool m_foundPreferredL1TAlgBlk;
    bool m_foundPreferredL1TExtBlk;

    bool m_foundMultipleL1TAlgBlk;
    bool m_foundMultipleL1TExtBlk;

    // use vector here, InputTag has no '<' operator to use std::set
    std::vector<edm::InputTag> m_inputTagsL1TAlgBlk;
    std::vector<edm::InputTag> m_inputTagsL1TExtBlk;
  };

  template <typename T>
  L1TGlobalUtilHelper::L1TGlobalUtilHelper(edm::ParameterSet const& pset, edm::ConsumesCollector& iC, T& module)
      : L1TGlobalUtilHelper(pset, iC, module, edm::InputTag(), edm::InputTag()) {}

  template <typename T>
  L1TGlobalUtilHelper::L1TGlobalUtilHelper(edm::ParameterSet const& pset,
                                           edm::ConsumesCollector& iC,
                                           T& module,
                                           edm::InputTag const& l1tAlgBlkInputTag,
                                           edm::InputTag const& l1tExtBlkInputTag)
      : m_consumesCollector(iC),

        // Set InputTags from arguments
        m_l1tAlgBlkInputTag(l1tAlgBlkInputTag),
        m_l1tExtBlkInputTag(l1tExtBlkInputTag),

        m_findL1TAlgBlk(false),
        m_findL1TExtBlk(false),

        m_readPrescalesFromFile(false),

        m_foundPreferredL1TAlgBlk(false),
        m_foundPreferredL1TExtBlk(false),

        m_foundMultipleL1TAlgBlk(false),
        m_foundMultipleL1TExtBlk(false) {
    if (pset.existsAs<bool>("ReadPrescalesFromFile")) {
      m_readPrescalesFromFile = pset.getParameter<bool>("ReadPrescalesFromFile");
    }
    // If the InputTags are not set to valid values by the arguments, then
    // try to set them from the configuration.
    if (m_l1tAlgBlkInputTag.label().empty() && pset.existsAs<edm::InputTag>("l1tAlgBlkInputTag")) {
      m_l1tAlgBlkInputTag = pset.getParameter<edm::InputTag>("l1tAlgBlkInputTag");
    }
    if (m_l1tExtBlkInputTag.label().empty() && pset.existsAs<edm::InputTag>("l1tExtBlkInputTag")) {
      m_l1tExtBlkInputTag = pset.getParameter<edm::InputTag>("l1tExtBlkInputTag");
    }

    // If the InputTags were set to valid values, make the consumes calls.
    if (!m_l1tAlgBlkInputTag.label().empty()) {
      m_l1tAlgBlkToken = iC.consumes<GlobalAlgBlkBxCollection>(m_l1tAlgBlkInputTag);
    }
    if (!m_l1tExtBlkInputTag.label().empty()) {
      m_l1tExtBlkToken = iC.consumes<GlobalExtBlkBxCollection>(m_l1tExtBlkInputTag);
    }

    // Do we still need to search for each InputTag?
    m_findL1TAlgBlk = m_l1tAlgBlkInputTag.label().empty();
    m_findL1TExtBlk = m_l1tExtBlkInputTag.label().empty();

    // Register the callback function with the Framework
    // if any InputTags still need to be found.
    if (m_findL1TAlgBlk || m_findL1TExtBlk) {
      module.callWhenNewProductsRegistered(std::ref(*this));
    }
  }

}  // namespace l1t

#endif
