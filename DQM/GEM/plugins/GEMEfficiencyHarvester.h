#ifndef DQM_GEM_GEMEfficiencyHarvester_h
#define DQM_GEM_GEMEfficiencyHarvester_h

/** \class GEMEfficiencyHarvester
 * 
 * DQM monitoring client for GEM efficiency and resolution
 * based on Validation/MuonGEMHits/MuonGEMBaseHarvestor
 *
 * TODO bookSummaryPlot
 *
 * \author Seungjin Yang <seungjin.yang@cern.ch>
 */

#include "DQM/GEM/interface/GEMDQMEfficiencyClientBase.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class GEMEfficiencyHarvester : public GEMDQMEfficiencyClientBase {
public:
  GEMEfficiencyHarvester(const edm::ParameterSet&);
  ~GEMEfficiencyHarvester() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  void bookDetector1DEfficiency(DQMStore::IBooker&, DQMStore::IGetter&, const std::string&);

  const std::vector<std::string> kFolders_;
};

#endif  // DQM_GEM_GEMEfficiencyHarvester_h
