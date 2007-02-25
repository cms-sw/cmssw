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

  void beginEvent(const edm::EventID&, const edm::Timestamp&);
  void endEvent(const edm::Event&, const edm::EventSetup&);
  
  void beginPath(std::string const & path);
  void endPath(std::string const & path,  const edm::HLTPathStatus&);

  bool doEvent() const { return m_doEvent;}
  bool active() const { return m_active>0;}

private:

  // configurable
  int m_firstEvent; 
  int m_lastEvent;
  std::vector<std::string> m_paths; 

  // internal state
  int m_evtCount;
  bool m_doEvent;
  int m_active;
  std::string m_activePath;

}; 

#endif // ProfilerService_H
