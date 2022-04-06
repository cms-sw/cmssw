#ifndef DQM_GEM_GEMDQMEfficiencyCalculator_h
#define DQM_GEM_GEMDQMEfficiencyCalculator_h

/** GEMDQMEfficiencyCalculator
 * 
 * \author Seungjin Yang <seungjin.yang@cern.ch>
 */

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <vector>
#include <string>

class GEMDQMEfficiencyCalculator {
public:
  typedef dqm::harvesting::DQMStore DQMStore;
  typedef dqm::harvesting::MonitorElement MonitorElement;

  GEMDQMEfficiencyCalculator();
  ~GEMDQMEfficiencyCalculator();

  void drawEfficiency(DQMStore::IBooker&, DQMStore::IGetter&, const std::string&);

private:
  TProfile* computeEfficiency(const TH1F*, const TH1F*, const char*, const char*);
  TH2F* computeEfficiency(const TH2F*, const TH2F*, const char*, const char*);

  const float kConfidenceLevel_ = 0.683;
  const std::string kMatchedSuffix_ = "_matched";
  const std::string kLogCategory_ = "GEMDQMEfficiencyCalculator";
};

#endif  // DQM_GEM_GEMDQMEfficiencyCalculator_h
