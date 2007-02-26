#ifndef ProfilerService_H
#define ProfilerService_H


//FIXME only forward declarations???
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <vector>
#include <string>


/* A Service to start and stop calgrind profiling on demand...
 * act also as profiler watchdog 
 * (in the same service to avoid dependency between service)
 */
class ProfilerService {
public:
  ProfilerService(edm::ParameterSet const& pset, 
		  edm::ActivityRegistry  & activity);

  ~ProfilerService();

  bool startInstrumentation();
  bool stopInstrumentation();
  bool forceStopInstrumentation();
  void dumpStat();

  void beginEventI(const edm::EventID&, const edm::Timestamp&) {
    beginEvent();
  }
  void endEventI(const edm::Event&, const edm::EventSetup&) {
    endEvent();
  }
  void beginPathI(std::string const & path) {
    beginPath(path);
  }
  void endPathI(std::string const & path,  const edm::HLTPathStatus&) {
    endPath(path);
  }

  bool doEvent() const { return m_doEvent;}
  bool active() const { return m_active>0;}

private:

  void beginEvent();
  void endEvent();
  
  void beginPath(std::string const & path);
  void endPath(std::string const & path);


  // configurable
  int m_firstEvent; 
  int m_lastEvent;
  std::vector<std::string> m_paths; 
  bool m_allPaths;

  // internal state
  int m_evtCount;
  bool m_doEvent;
  int m_active;
  std::string m_activePath;

}; 

#endif // ProfilerService_H
