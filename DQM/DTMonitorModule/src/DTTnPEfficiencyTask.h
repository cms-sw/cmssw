#ifndef DTTnPEfficiencyTask_H
#define DTTnPEfficiencyTask_H

/*
 * \file DTTnPEfficiencyTask.h
 *
 * \author L. Lunerti - INFN Bologna
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/AnySelector.h"

#include <vector>
#include <string>
#include <map>

class DTTnPEfficiencyTask : public DQMEDAnalyzer
{

public:
  /// Constructor
  DTTnPEfficiencyTask(const edm::ParameterSet& config);

  /// Destructor
  ~DTTnPEfficiencyTask() override;

protected:

  /// BeginRun
  void dqmBeginRun(const edm::Run& run, const edm::EventSetup& context) override;

  void bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& context) override;

  /// Book wheel granularity histograms
  void bookWheelHistos(DQMStore::IBooker& iBooker, int wheel, std::string folder = "");

  /// Return the top folder
  inline std::string topFolder() const { return "DT/10-Segment_TnP/"; };

  /// Analyze
  void analyze(const edm::Event& event, const edm::EventSetup& context) override;

  /// To reset the MEs

private:

  int m_nEvents;

  edm::EDGetTokenT<reco::MuonCollection> m_muToken;

  bool m_detailedAnalysis;

  StringCutObjectSelector<reco::Candidate,true> m_selector;

  std::map<std::string, MonitorElement*> m_histos;

  float m_borderCut;

};

#endif
