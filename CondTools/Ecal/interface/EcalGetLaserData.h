#ifndef CondTools_Ecal_EcalGetLaserData_h
#define CondTools_Ecal_EcalGetLaserData_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"

#include <string>
#include <map>
#include <vector>
#include <ctime>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class EcalLaserAPDPNRatios;
class EcalLaserAPDPNRatiosRcd;
class EcalLaserAPDPNRatiosRefRcd;
class EcalLaserAlphasRcd;

class EcalGetLaserData : public edm::one::EDAnalyzer<> {
public:
  explicit EcalGetLaserData(const edm::ParameterSet& iConfig);
  ~EcalGetLaserData() override;

  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;

private:
  //std::string m_timetype;
  std::map<std::string, unsigned long long> m_cacheIDs;
  std::map<std::string, std::string> m_records;
  //unsigned long m_firstRun ;
  //unsigned long m_lastRun ;

  void beginJob() override;
  void endJob() override;
  edm::ESGetToken<EcalLaserAPDPNRatios, EcalLaserAPDPNRatiosRcd> ecalLaserAPDPNRatiosToken_;
  edm::ESGetToken<EcalLaserAPDPNRatiosRef, EcalLaserAPDPNRatiosRefRcd> ecalLaserAPDPNRatiosRefToken_;
  edm::ESGetToken<EcalLaserAlphas, EcalLaserAlphasRcd> ecalLaserAlphasToken_;
};

#endif
