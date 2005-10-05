#ifndef Input_MCFileSource_h
#define Input_MCFileSource_h

/** \class MCFileSource
 *
 * Reads in HepMC events
 * Joanna Weng & Filip Moortgat 08/2005 
 ***************************************/

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "IOMC/Input/interface/HepMCFileReader.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/BranchDescription.h"
#include "FWCore/EDProduct/interface/EventID.h"
#include <map>
#include <string>

class HepMCFileReader;

namespace edm
{
  class MCFileSource : public InputSource {
  public:
    MCFileSource(const ParameterSet &, const InputSourceDescription &  );
   virtual ~MCFileSource();
// the following cannot be used anymore since an explicit InputSourceDescription is needed ?? FM
/*
    MCFileSource(const std::string& processName);
    /// Specify the file to be read. FIXME: should be done by the "configuration"
    MCFileSource(const std::string& filename, const std::string& processName);
*/

  private:
   
   virtual std::auto_ptr<EventPrincipal> read();
    void clear();
    
    int remainingEvents_;
    unsigned long numberEventsInRun_;
    unsigned long presentRun_;
    unsigned long nextTime_;
    unsigned long timeBetweenEvents_;
    unsigned long numberEventsInThisRun_;
    EventID nextID_;
    
    HepMCFileReader * reader_;
    
    HepMC::GenEvent  *evt;
    
    std::string filename_;
    BranchDescription branchDesc_;
	
    
    	
  };
} 

#endif
