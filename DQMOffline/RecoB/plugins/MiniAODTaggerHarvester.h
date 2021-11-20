#ifndef MiniAODTaggerHarvester_H
#define MiniAODTaggerHarvester_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMOffline/RecoB/interface/JetTagPlotter.h"

/** \class MiniAODTaggerHarvester
 *
 *  Tagger harvester to run on MiniAOD
 *
 */

class MiniAODTaggerHarvester : public DQMEDHarvester {
public:
  explicit MiniAODTaggerHarvester(const edm::ParameterSet& pSet);
  ~MiniAODTaggerHarvester() override;

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

  std::unique_ptr<JetTagPlotter> jetTagPlotter_;

  const std::string folder_;
  const edm::ParameterSet discrParameters_;

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
