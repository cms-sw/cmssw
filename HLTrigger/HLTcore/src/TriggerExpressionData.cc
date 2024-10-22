#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"

namespace triggerExpression {

  void Data::setPathStatusToken(edm::BranchDescription const& branch, edm::ConsumesCollector&& iC) {
    m_pathStatusTokens[branch.moduleLabel()] = iC.consumes<edm::HLTPathStatus>(
        edm::InputTag(branch.moduleLabel(), branch.productInstanceName(), branch.processName()));
  }

  bool Data::setEvent(const edm::Event& event, const edm::EventSetup& setup) {
    // cache the event number
    m_eventNumber = event.id().event();

    // access L1 objects only if L1 is used
    if (hasL1T()) {
      // cache the L1 GT results objects
      auto const& l1t = edm::get(event, m_l1tResultsToken);
      if (l1t.size() == 0 or l1t.isEmpty(0)) {
        m_l1tResults = nullptr;
        return false;
      }
      if (m_l1tIgnoreMaskAndPrescale)
        m_l1tResults = &l1t.at(0, 0).getAlgoDecisionInitial();
      else
        m_l1tResults = &l1t.at(0, 0).getAlgoDecisionFinal();

      // cache the L1 trigger menu
      unsigned long long l1tCacheID = setup.get<L1TUtmTriggerMenuRcd>().cacheIdentifier();
      if (m_l1tCacheID == l1tCacheID) {
        m_l1tUpdated = false;
      } else {
        m_l1tMenu = &setup.getData(m_l1tUtmTriggerMenuToken);
        m_l1tCacheID = l1tCacheID;
        m_l1tUpdated = true;
      }
    }

    // access HLT objects only if HLT is used
    if (usePathStatus()) {
      m_pathStatus.clear();
      std::vector<std::string> triggerNames;
      m_pathStatus.reserve(m_pathStatusTokens.size());
      triggerNames.reserve(m_pathStatusTokens.size());
      for (auto const& p : m_pathStatusTokens) {
        auto const& handle = event.getHandle(p.second);
        if (handle.isValid()) {
          m_pathStatus.push_back(handle->accept());
          triggerNames.push_back(p.first);
        } else {
          edm::LogError("MissingHLTPathStatus")
              << "invalid handle for requested edm::HLTPathStatus with label \"" << p.first << "\"";
        }
      }
      m_hltUpdated = m_triggerNames != triggerNames;
      m_triggerNames = triggerNames;
    } else if (hasHLT()) {
      // cache the HLT TriggerResults
      m_hltResults = &edm::get(event, m_hltResultsToken);
      if (not m_hltResults)
        return false;

      // access the TriggerNames, and check if it has changed
      m_hltMenu = &event.triggerNames(*m_hltResults);
      if (m_hltMenu->parameterSetID() == m_hltCacheID) {
        m_hltUpdated = false;
      } else {
        m_hltCacheID = m_hltMenu->parameterSetID();
        m_hltUpdated = true;
      }
    }

    return true;
  }

}  // namespace triggerExpression
