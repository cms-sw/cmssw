#ifndef MiniAODTaggerAnalyzer_H
#define MiniAODTaggerAnalyzer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DQMOffline/RecoB/interface/JetTagPlotter.h"

/** \class MiniAODTaggerAnalyzer
 *
 *  Tagger analyzer to run on MiniAOD
 *
 */

class MiniAODTaggerAnalyzer : public DQMEDAnalyzer {
public:
  explicit MiniAODTaggerAnalyzer(const edm::ParameterSet& pSet);
  ~MiniAODTaggerAnalyzer() override;

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  typedef std::vector<std::string> vstring;

  // using JetTagPlotter object for all the hard work ;)
  std::unique_ptr<JetTagPlotter> jetTagPlotter_;

  const edm::EDGetTokenT<std::vector<pat::Jet> > jetToken_;
  const edm::ParameterSet discrParameters_;

  const std::string folder_;
  const vstring discrNumerator_;
  const vstring discrDenominator_;

  const int mclevel_;
  const bool doCTagPlots_;
  const bool dodifferentialPlots_;
  const double discrCut_;

  const bool etaActive_;
  const double etaMin_;
  const double etaMax_;
  const bool ptActive_;
  const double ptMin_;
  const double ptMax_;
};

#endif
