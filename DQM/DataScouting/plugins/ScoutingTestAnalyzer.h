#ifndef ScoutingTestAnalyzer_h
#define ScoutingTestAnalyzer_h

// This class is used to test the functionalities of the package
#include "DQM/DataScouting/interface/ScoutingAnalyzerBase.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

class ScoutingTestAnalyzer : public ScoutingAnalyzerBase {
public:
  explicit ScoutingTestAnalyzer(const edm::ParameterSet &);
  ~ScoutingTestAnalyzer() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  // histograms
  edm::InputTag m_pfJetsCollectionTag;
  MonitorElement *m_jetPt;
  MonitorElement *m_jetEtaPhi;
  // define Token(-s)
  edm::EDGetTokenT<reco::CaloJetCollection> m_pfJetsCollectionTagToken_;
};
#endif
