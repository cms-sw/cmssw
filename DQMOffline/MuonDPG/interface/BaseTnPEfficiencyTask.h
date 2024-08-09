#ifndef DQMOffline_MuonDPG_BaseTnPEfficiencyTask_H
#define DQMOffline_MuonDPG_BaseTnPEfficiencyTask_H

/*
 * \file BaseTnPEfficiencyTask.h
 *
 * \author L. Lunerti - INFN Bologna
 *
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DataFormats/GEMDigi/interface/GEMVFATStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMOHStatusCollection.h"
//#include "DataFormats/GEMDigi/interface/GEMAMCStatusCollection.h"

#include <vector>
#include <string>
#include <map>

class BaseTnPEfficiencyTask : public DQMEDAnalyzer {
public:
  /// Constructor
  BaseTnPEfficiencyTask(const edm::ParameterSet& config);

  /// Destructor
  ~BaseTnPEfficiencyTask() override;

protected:
  /// BeginRun
  void dqmBeginRun(const edm::Run& run, const edm::EventSetup& context) override;  //final ?

  void bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& context) override;

  /// Return the top folder
  virtual std::string topFolder() const = 0;

  /// Analyze
  void analyze(const edm::Event& event, const edm::EventSetup& context) override;  // final?
  bool hasTrigger(std::vector<int>& trigIndices,
                  const trigger::TriggerObjectCollection& trigObjs,
                  edm::Handle<trigger::TriggerEvent>& trigEvent,
                  const reco::Muon& muon);

  std::vector<std::vector<unsigned>> m_probeIndices;
  std::vector<std::vector<unsigned>> m_tagIndices;

  std::map<std::string, MonitorElement*> m_histos;

  int m_nEvents;

  const edm::EDGetTokenT<reco::MuonCollection> m_muToken;

  const double m_borderCut;
  const double m_dxCut;
  const bool m_detailedAnalysis;

  const bool kMaskChamberWithError_;

  const edm::EDGetTokenT<GEMOHStatusCollection> kGEMOHStatusCollectionToken_;
  const edm::EDGetTokenT<GEMVFATStatusCollection> kGEMVFATStatusCollectionToken_;
  //const edm::EDGetTokenT<GEMAMCStatusCollection> kGEMAMCStatusCollectionToken_;

private:
  const edm::EDGetTokenT<std::vector<reco::Vertex>> m_primaryVerticesToken;
  const edm::EDGetTokenT<edm::TriggerResults> m_triggerResultsToken;
  const edm::EDGetTokenT<trigger::TriggerEvent> m_triggerEventToken;

  const std::string m_trigName;
  HLTConfigProvider m_hltConfig;

  //Probe selectors
  const StringCutObjectSelector<reco::Candidate, true> m_probeSelector;
  const double m_dxyCut;
  const double m_dzCut;

  //Tag selectors
  const StringCutObjectSelector<reco::Muon, true> m_tagSelector;

  //Trigger indices
  std::vector<int> m_trigIndices;

  const double m_lowPairMassCut;
  const double m_highPairMassCut;
};

#endif
