#ifndef GeneratorInterface_MCatNLOInterface_MCatNLOSource_h
#define GeneratorInterface_MCatNLOInterface_MCatNLOSource_h

#include <memory>

#include <boost/shared_ptr.hpp>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Sources/interface/ProducerSourceFromFiles.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

#include "GeneratorInterface/Herwig6Interface/interface/Herwig6Instance.h"

#include <fstream>


// Common Block for HERWIG parameters set by UPINIT
extern "C" {
  extern struct MCPARS_ {
    double emmin;
    double emmax;
    double gammax;
    double rmass[1000];
    double gamw;
    double gamz;
    int emmins;
    int emmaxs;
    int gammaxs;
    int rmasss[1000];
    int gamws;
    int gamzs;
  } mcpars_; 
}    

namespace lhef {
  class LHERunInfo;
  class LHEEvent;
}

class MCatNLOSource : public edm::ProducerSourceFromFiles,
                   public gen::Herwig6Instance {
public:
  explicit MCatNLOSource(const edm::ParameterSet &params,
		      const edm::InputSourceDescription &desc);
  virtual ~MCatNLOSource();
  
private:
  virtual void endJob();
  virtual void beginRun(edm::Run &run);
  virtual bool setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&);
  virtual void produce(edm::Event &event);
  
  void nextEvent();

  virtual bool hwwarn(const std::string &fn, int code);

  /// Name of the input file
  std::string                   fileName;
  
  /// Pointer to the input file
  std::unique_ptr<std::ifstream>  inputFile;
  
  /// Number of events to skip
  unsigned int                  skipEvents;
  
  /// Number of events
  unsigned int                  nEvents;

  int                           ihpro;
  
  int                           processCode;

  std::unique_ptr<std::ifstream>          reader;
  
  boost::shared_ptr<lhef::LHERunInfo>	runInfo;
  boost::shared_ptr<lhef::LHEEvent>	event;
};

#endif // GeneratorInterface_MCatNLOInterface_MCatNLOSource_h
