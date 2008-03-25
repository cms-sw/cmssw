#ifndef UpdaterService_H
#define UpdaterService_H

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {
  class ParameterSet;
  class EventID;
  class TimeStamp;
}

#include <map>
#include <string>

class UpdaterService {
 public:
  UpdaterService(const edm::ParameterSet & cfg, edm::ActivityRegistry & r );
  ~UpdaterService();

  void init(const edm::EventID&, const edm::Timestamp&);
  bool checkOnce(std::string);
  bool check(std::string, std::string);

 private:
  void theInit();
  std::map< std::string, uint > theCounts;
  const edm::EventID * theEventId;
};

#endif
