#ifndef ECALTPGDBCOPY_H
#define ECALTPGDBCOPY_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CondCore/CondDB/interface/Exception.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"

#include <string>
#include <map>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class  EcalTPGDBCopy : public edm::EDAnalyzer {
 public:
  explicit  EcalTPGDBCopy(const edm::ParameterSet& iConfig );
  ~EcalTPGDBCopy();

  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);

 private:
  bool shouldCopy(const edm::EventSetup& evtSetup, std::string container);
  void copyToDB(const edm::EventSetup& evtSetup, std::string container);

  std::string m_timetype;
  std::map<std::string, unsigned long long> m_cacheIDs;
  std::map<std::string, std::string> m_records;
};

#endif
