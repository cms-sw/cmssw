#ifndef GlobalTriggerAnalyzer_L1GtUtilsHelper_h
#define GlobalTriggerAnalyzer_L1GtUtilsHelper_h

/**
 * \class L1GtUtilsHelper
 *
 *
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

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtTriggerMenuLite.h"

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

class L1GtUtilsHelper {
public:
  // Using this constructor will require InputTags to be specified in the configuration
  L1GtUtilsHelper(edm::ParameterSet const& pset, edm::ConsumesCollector& iC, bool useL1GtTriggerMenuLite);

  // Using this constructor will cause it to look for valid InputTags in
  // the following ways in the specified order until they are found.
  //   1. The configuration
  //   2. Search all products from the preferred input tags for the required type
  //   3. Search all products from any other process for the required type
  template <typename T>
  L1GtUtilsHelper(edm::ParameterSet const& pset, edm::ConsumesCollector& iC, bool useL1GtTriggerMenuLite, T& module);

  // Using this constructor will cause it to look for valid InputTags in
  // the following ways in the specified order until they are found.
  //   1. The constructor arguments
  //   2. The configuration
  //   3. Search all products from the preferred input tags for the required type
  //   4. Search all products from any other process for the required type
  template <typename T>
  L1GtUtilsHelper(edm::ParameterSet const& pset,
                  edm::ConsumesCollector& iC,
                  bool useL1GtTriggerMenuLite,
                  T& module,
                  edm::InputTag const& l1GtRecordInputTag,
                  edm::InputTag const& l1GtReadoutRecordInputTag,
                  edm::InputTag const& l1GtTriggerMenuLiteInputTag);

  // A module defining its fillDescriptions function might want to use this
  static void fillDescription(edm::ParameterSetDescription& desc);

  edm::InputTag const& l1GtRecordInputTag() const { return m_l1GtRecordInputTag; }
  edm::InputTag const& l1GtReadoutRecordInputTag() const { return m_l1GtReadoutRecordInputTag; }
  edm::InputTag const& l1GtTriggerMenuLiteInputTag() const { return m_l1GtTriggerMenuLiteInputTag; }

  edm::EDGetTokenT<L1GlobalTriggerRecord> const& l1GtRecordToken() const { return m_l1GtRecordToken; }
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> const& l1GtReadoutRecordToken() const {
    return m_l1GtReadoutRecordToken;
  }
  edm::EDGetTokenT<L1GtTriggerMenuLite> const& l1GtTriggerMenuLiteToken() const { return m_l1GtTriggerMenuLiteToken; }

private:
  // Callback which will be registered with the Framework if the InputTags
  // are not specified in the configuration or constructor arguments. It
  // will get called for each product in the ProductRegistry.
  void checkToUpdateTags(edm::BranchDescription const& branchDescription,
                         edm::ConsumesCollector,
                         bool findRecord,
                         bool findReadoutRecord,
                         bool findMenuLite);

  edm::InputTag m_l1GtRecordInputTag;
  edm::InputTag m_l1GtReadoutRecordInputTag;
  edm::InputTag m_l1GtTriggerMenuLiteInputTag;

  edm::EDGetTokenT<L1GlobalTriggerRecord> m_l1GtRecordToken;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_l1GtReadoutRecordToken;
  edm::EDGetTokenT<L1GtTriggerMenuLite> m_l1GtTriggerMenuLiteToken;
};

template <typename T>
L1GtUtilsHelper::L1GtUtilsHelper(edm::ParameterSet const& pset,
                                 edm::ConsumesCollector& iC,
                                 bool useL1GtTriggerMenuLite,
                                 T& module)
    : L1GtUtilsHelper(pset, iC, useL1GtTriggerMenuLite, module, edm::InputTag(), edm::InputTag(), edm::InputTag()) {}

template <typename T>
L1GtUtilsHelper::L1GtUtilsHelper(edm::ParameterSet const& pset,
                                 edm::ConsumesCollector& iC,
                                 bool useL1GtTriggerMenuLite,
                                 T& module,
                                 edm::InputTag const& l1GtRecordInputTag,
                                 edm::InputTag const& l1GtReadoutRecordInputTag,
                                 edm::InputTag const& l1GtTriggerMenuLiteInputTag)
    :  // Set InputTags from arguments
      m_l1GtRecordInputTag(l1GtRecordInputTag),
      m_l1GtReadoutRecordInputTag(l1GtReadoutRecordInputTag),
      m_l1GtTriggerMenuLiteInputTag(l1GtTriggerMenuLiteInputTag) {
  // If the InputTags are not set to valid values by the arguments, then
  // try to set them from the configuration.
  if (m_l1GtRecordInputTag.label().empty() && pset.existsAs<edm::InputTag>("l1GtRecordInputTag")) {
    m_l1GtRecordInputTag = pset.getParameter<edm::InputTag>("l1GtRecordInputTag");
  }
  if (m_l1GtReadoutRecordInputTag.label().empty() && pset.existsAs<edm::InputTag>("l1GtReadoutRecordInputTag")) {
    m_l1GtReadoutRecordInputTag = pset.getParameter<edm::InputTag>("l1GtReadoutRecordInputTag");
  }
  if (useL1GtTriggerMenuLite && m_l1GtTriggerMenuLiteInputTag.label().empty() &&
      pset.existsAs<edm::InputTag>("l1GtTriggerMenuLiteInputTag")) {
    m_l1GtTriggerMenuLiteInputTag = pset.getParameter<edm::InputTag>("l1GtTriggerMenuLiteInputTag");
  }

  // If the InputTags were set to valid values, make the consumes calls.
  if (!m_l1GtRecordInputTag.label().empty()) {
    m_l1GtRecordToken = iC.consumes<L1GlobalTriggerRecord>(m_l1GtRecordInputTag);
  }
  if (!m_l1GtReadoutRecordInputTag.label().empty()) {
    m_l1GtReadoutRecordToken = iC.consumes<L1GlobalTriggerReadoutRecord>(m_l1GtReadoutRecordInputTag);
  }
  if (useL1GtTriggerMenuLite && !m_l1GtTriggerMenuLiteInputTag.label().empty()) {
    m_l1GtTriggerMenuLiteToken = iC.consumes<L1GtTriggerMenuLite, edm::InRun>(m_l1GtTriggerMenuLiteInputTag);
  }

  // Do we still need to search for each InputTag?
  bool findRecord = m_l1GtRecordInputTag.label().empty();
  bool findReadoutRecord = m_l1GtReadoutRecordInputTag.label().empty();
  bool findMenuLite = m_l1GtTriggerMenuLiteInputTag.label().empty() && useL1GtTriggerMenuLite;

  // Register the callback function with the Framework
  // if any InputTags still need to be found.
  if (findRecord || findReadoutRecord || findMenuLite) {
    module.callWhenNewProductsRegistered([this, findRecord, findReadoutRecord, findMenuLite, iC](auto iBranch) {
      checkToUpdateTags(iBranch, iC, findRecord, findReadoutRecord, findMenuLite);
    });
  }
}
#endif
