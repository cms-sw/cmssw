#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtilsHelper.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/TypeID.h"

L1GtUtilsHelper::L1GtUtilsHelper(edm::ParameterSet const& pset,
                                 edm::ConsumesCollector& iC,
                                 bool useL1GtTriggerMenuLite) :
  m_consumesCollector(std::move(iC)),
  m_l1GtRecordInputTag(pset.getParameter<edm::InputTag>("l1GtRecordInputTag")),
  m_l1GtReadoutRecordInputTag(pset.getParameter<edm::InputTag>("l1GtReadoutRecordInputTag")),
  m_l1GtTriggerMenuLiteInputTag(pset.getParameter<edm::InputTag>("l1GtTriggerMenuLiteInputTag")),
  m_findRecord(false),
  m_findReadoutRecord(false),
  m_findMenuLite(false),
  m_foundRECORecord(false),
  m_foundRECOReadoutRecord(false),
  m_foundRECOMenuLite(false)
{
  m_l1GtRecordToken = iC.consumes<L1GlobalTriggerRecord>(m_l1GtRecordInputTag);
  m_l1GtReadoutRecordToken = iC.consumes<L1GlobalTriggerReadoutRecord>(m_l1GtReadoutRecordInputTag);
  if(useL1GtTriggerMenuLite) {
    m_l1GtTriggerMenuLiteToken = iC.consumes<L1GtTriggerMenuLite,edm::InRun>(m_l1GtTriggerMenuLiteInputTag);
  }
}

void L1GtUtilsHelper::fillDescription(edm::ParameterSetDescription & desc) {
  desc.add<edm::InputTag>("l1GtRecordInputTag", edm::InputTag());
  desc.add<edm::InputTag>("l1GtReadoutRecordInputTag", edm::InputTag());
  desc.add<edm::InputTag>("l1GtTriggerMenuLiteInputTag", edm::InputTag());
}

void L1GtUtilsHelper::operator()(edm::BranchDescription const& branchDescription) {

  // This is only used if required InputTags were not specified already.
  // This is called early in the process, once for each product in the ProductRegistry.
  // The callback is registered when callWhenNewProductsRegistered is called.
  // It finds products by type and sets the token so that it can be used
  // later when getting the product.
  // This code assumes there is at most one product from the RECO process
  // and at most one product from some other process.  It selects a product
  // from the RECO process if it is present and a product from some other process
  // if it is not. This is a bit unsafe because if there is more than one from
  // RECO or none from RECO and more than one from other processes it is somewhat
  // arbitrary which one is selected. I'm leaving this behavior in to maintain
  // consistency with the previous behavior. It is supposed to not happen and if
  // it does the products might be identical anyway. To avoid this risk or select
  // different products, specify the InputTags either in the configuration or the
  // arguments to the constructor.

  if (m_findRecord &&
      !m_foundRECORecord &&
      branchDescription.unwrappedTypeID() == edm::TypeID(typeid(L1GlobalTriggerRecord)) &&
      branchDescription.branchType() == edm::InEvent) {
    edm::InputTag tag{branchDescription.moduleLabel(),
                      branchDescription.productInstanceName(),
                      branchDescription.processName()};
    if(branchDescription.processName() == "RECO") {
      m_l1GtRecordInputTag = tag;
      m_l1GtRecordToken = m_consumesCollector.consumes<L1GlobalTriggerRecord>(tag);
      m_foundRECORecord = true;
    } else if (m_l1GtRecordToken.isUninitialized()) {
      m_l1GtRecordInputTag = tag;
      m_l1GtRecordToken = m_consumesCollector.consumes<L1GlobalTriggerRecord>(tag);
    }
  }

  if (m_findReadoutRecord &&
      !m_foundRECOReadoutRecord &&
      branchDescription.unwrappedTypeID() == edm::TypeID(typeid(L1GlobalTriggerReadoutRecord)) &&
      branchDescription.branchType() == edm::InEvent) {
    edm::InputTag tag{branchDescription.moduleLabel(),
                      branchDescription.productInstanceName(),
                      branchDescription.processName()};
    if(branchDescription.processName() == "RECO") {
      m_l1GtReadoutRecordInputTag = tag;
      m_l1GtReadoutRecordToken = m_consumesCollector.consumes<L1GlobalTriggerReadoutRecord>(tag);
      m_foundRECOReadoutRecord = true;
    } else if (m_l1GtReadoutRecordToken.isUninitialized()) {
      m_l1GtReadoutRecordInputTag = tag;
      m_l1GtReadoutRecordToken = m_consumesCollector.consumes<L1GlobalTriggerReadoutRecord>(tag);
    }
  }

  if (m_findMenuLite &&
      !m_foundRECOMenuLite &&
      branchDescription.branchType() == edm::InRun &&
      branchDescription.unwrappedTypeID() == edm::TypeID(typeid(L1GtTriggerMenuLite))) {
    edm::InputTag tag{branchDescription.moduleLabel(),
                      branchDescription.productInstanceName(),
                      branchDescription.processName()};
    if(branchDescription.processName() == "RECO") {
      m_l1GtTriggerMenuLiteInputTag = tag;
      m_l1GtTriggerMenuLiteToken = m_consumesCollector.consumes<L1GtTriggerMenuLite,edm::InRun>(tag);
      m_foundRECOMenuLite = true;
    } else if (m_l1GtTriggerMenuLiteToken.isUninitialized()) {
      m_l1GtTriggerMenuLiteInputTag = tag;
      m_l1GtTriggerMenuLiteToken = m_consumesCollector.consumes<L1GtTriggerMenuLite,edm::InRun>(tag);
    }
  }
}
