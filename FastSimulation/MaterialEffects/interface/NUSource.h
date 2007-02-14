#ifndef FastSimulation_MaterialEffects_NUSource_h
#define FastSimulation_MaterialEffects_NUSource_h

/*----------------------------------------------------------------------

NUSource: This is an InputSource for NUclear interactions

$Id: NUSource.h,v 1.0 2007/01/26 13:02:39 pjanot Exp $

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

  class NUSource : public VectorInputSource {

  public:
    explicit NUSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~NUSource();


  private:
    typedef boost::shared_ptr<RootFile> RootFileSharedPtr;
    typedef std::map<std::string, RootFileSharedPtr> RootFileMap;
    NUSource(NUSource const&); // disable copy construction
    NUSource & operator=(NUSource const&); // disable assignment
    virtual std::auto_ptr<EventPrincipal> read();
    virtual std::auto_ptr<EventPrincipal> readIt(int entry);
    virtual void readMany_(int number, EventPrincipalVector& result);
    void init();

    RootFileSharedPtr rootFile_;
    RootFileMap rootFiles_;
    std::map<RootFileSharedPtr,int> eventsInRootFiles; 
    int totalNbEvents;
    int localEntry;

  }; // class NUSource
}
#endif
