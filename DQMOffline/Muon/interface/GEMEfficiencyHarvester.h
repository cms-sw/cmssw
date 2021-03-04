#ifndef DQMOffline_Muon_GEMEfficiencyHarvester_h
#define DQMOffline_Muon_GEMEfficiencyHarvester_h

/** \class GEMEfficiencyAnalyzer
 * 
 * DQM monitoring client for GEM efficiency and resolution
 * based on Validation/MuonGEMHits/MuonGEMBaseHarvestor
 *
 * \author Seungjin Yang <seungjin.yang@cern.ch>
 */

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <vector>
#include <string>

class GEMEfficiencyHarvester : public DQMEDHarvester {
public:
  GEMEfficiencyHarvester(const edm::ParameterSet&);
  ~GEMEfficiencyHarvester() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  TProfile* computeEfficiency(const TH1F*, const TH1F*, const char*, const char*, const double confidence_level = 0.683);

  TH2F* computeEfficiency(const TH2F*, const TH2F*, const char*, const char*);

  std::vector<std::string> splitString(std::string, const std::string);
  std::tuple<std::string, int, int> parseResidualName(std::string, const std::string);

  void doEfficiency(DQMStore::IBooker&, DQMStore::IGetter&);
  void doResolution(DQMStore::IBooker&, DQMStore::IGetter&, const std::string);

  template <typename T>
  int findResolutionBin(const T&, const std::vector<T>&);

  std::string folder_;
  std::string log_category_;
};

template <typename T>
int GEMEfficiencyHarvester::findResolutionBin(const T& elem, const std::vector<T>& vec) {
  auto iter = std::find(vec.begin(), vec.end(), elem);
  int bin = (iter != vec.end()) ? std::distance(vec.begin(), iter) + 1 : -1;
  return bin;
}

#endif  // DQMOffline_Muon_GEMEfficiencyHarvester_h
