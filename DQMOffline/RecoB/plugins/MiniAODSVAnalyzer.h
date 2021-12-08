#ifndef MiniAODSVAnalyzer_H
#define MiniAODSVAnalyzer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

/** \class MiniAODSVAnalyzer
 *
 *  Secondary Vertex Analyzer to run on MiniAOD
 *
 */

class MiniAODSVAnalyzer : public DQMEDAnalyzer {
public:
  explicit MiniAODSVAnalyzer(const edm::ParameterSet& pSet);
  ~MiniAODSVAnalyzer() override;

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  const edm::EDGetTokenT<std::vector<pat::Jet> > jetToken_;
  const std::string svTagInfo_;
  const double jetPtMin_;
  const double etaMax_;

  MonitorElement* n_sv_;

  MonitorElement* sv_mass_;
  MonitorElement* sv_pt_;
  MonitorElement* sv_ntracks_;
  MonitorElement* sv_chi2norm_;
  MonitorElement* sv_chi2prob_;

  // relation to jet
  MonitorElement* sv_ptrel_;
  MonitorElement* sv_energyratio_;
  MonitorElement* sv_deltaR_;

  MonitorElement* sv_dxy_;
  MonitorElement* sv_dxysig_;
  MonitorElement* sv_d3d_;
  MonitorElement* sv_d3dsig_;
};

#endif
