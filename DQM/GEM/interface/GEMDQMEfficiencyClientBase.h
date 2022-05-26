#ifndef DQM_GEM_GEMDQMEfficiencyClientBase_h
#define DQM_GEM_GEMDQMEfficiencyClientBase_h

/** \class GEMDQMEfficiencyClientBase
 * 
 * \author Seungjin Yang <seungjin.yang@cern.ch>
 */

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

class GEMDQMEfficiencyClientBase : public DQMEDHarvester {
public:
  using MEPair = std::pair<const MonitorElement*, const MonitorElement*>;

  GEMDQMEfficiencyClientBase(const edm::ParameterSet&);

  std::tuple<bool, std::string, std::string, bool> parseEfficiencySourceName(std::string);
  GEMDetId parseGEMLabel(const std::string, const std::string delimiter = "-");

  std::map<std::string, MEPair> makeEfficiencySourcePair(DQMStore::IBooker&,
                                                         DQMStore::IGetter&,
                                                         const std::string&,
                                                         const std::string prefix = "");
  void setBins(TH1F*, const TAxis*);
  TH1F* projectHistogram(const TH2F*, const unsigned int);
  bool checkConsistency(const TH1&, const TH1&);
  TH1F* makeEfficiency(const TH1F*, const TH1F*, const char* name = nullptr, const char* title = nullptr);
  TH2F* makeEfficiency(const TH2F*, const TH2F*, const char* name = nullptr, const char* title = nullptr);
  void bookEfficiencyAuto(DQMStore::IBooker&, DQMStore::IGetter&, const std::string&);

  const double kConfidenceLevel_;
  const std::string kLogCategory_;
};

#endif  // DQM_GEM_GEMDQMEfficiencyClientBase_h
