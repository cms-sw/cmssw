#ifndef FastSimulation_PileUpProducer_PUSource_h
#define FastSimulation_PileUpProducer_PUSource_h

/*----------------------------------------------------------------------

PUSource: This is an InputSource

$Id: PoolSource.h,v 1.18 2006/04/04 22:19:16 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <string>
#include <map>

#include "IOPool/Input/src/Inputfwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/VectorInputSource.h"

#include "boost/shared_ptr.hpp"

namespace edm {

  class RootFile;

  class PUSource : public VectorInputSource {

  public:
    explicit PUSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~PUSource();

  private:
    typedef boost::shared_ptr<RootFile> RootFileSharedPtr;
    typedef std::map<std::string, RootFileSharedPtr> RootFileMap;
    PUSource(PUSource const&); // disable copy construction
    PUSource & operator=(PUSource const&); // disable assignment
    virtual std::auto_ptr<EventPrincipal> read();
    virtual std::auto_ptr<EventPrincipal> readIt(int entry);
    virtual void readMany_(int number, EventPrincipalVector& result);
    void init();

    RootFileSharedPtr rootFile_;
    std::map<RootFileSharedPtr,int> eventsInRootFiles; 
    int totalNbEvents;
  }; // class PUSource
}
#endif
