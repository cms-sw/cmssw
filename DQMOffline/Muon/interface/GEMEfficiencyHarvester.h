#ifndef DQMOffline_Muon_GEMEfficiencyHarvester_h
#define DQMOffline_Muon_GEMEfficiencyHarvester_h

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <string>

class GEMEfficiencyHarvester : public DQMEDHarvester {
public:
  GEMEfficiencyHarvester(const edm::ParameterSet&);
  ~GEMEfficiencyHarvester() override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  TProfile* computeEfficiency(const TH1F*, const TH1F*, const char*, const char*, const double confidence_level = 0.683);

  TH2F* computeEfficiency(const TH2F*, const TH2F*, const char*, const char*);

  std::vector<std::string> splitString(std::string, const std::string);
  std::tuple<std::string, int, bool, int> parseResidualName(std::string, const std::string);

  void doEfficiency(DQMStore::IBooker&, DQMStore::IGetter&);
  void doResolution(DQMStore::IBooker&, DQMStore::IGetter&, const std::string);

  std::string folder_;
  std::string log_category_;
};

#endif  // DQMOffline_Muon_GEMEfficiencyHarvester_h
