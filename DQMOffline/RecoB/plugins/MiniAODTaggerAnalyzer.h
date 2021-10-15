#ifndef MiniAODTaggerAnalyzer_H
#define MiniAODTaggerAnalyzer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
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

  edm::EDGetTokenT<std::vector<pat::Jet> > jetToken_;
  edm::ParameterSet disrParameters_;

  std::string folder_;
  vstring discrNumerator_;
  vstring discrDenominator_;

  int mclevel_;
  bool doCTagPlots_;
  bool dodifferentialPlots_;
  double discrCut_;

  bool etaActive_;
  double etaMin_;
  double etaMax_;
  bool ptActive_;
  double ptMin_;
  double ptMax_;
};

#endif
