#ifndef Input_MCFileSource_h
#define Input_MCFileSource_h

/** \class MCFileSource
 *
 * Reads in HepMC events
 * Joanna Weng & Filip Moortgat 08/2005 
 ***************************************/

#include "FWCore/Framework/interface/ExternalInputSource.h"
#include "IOMC/Input/interface/HepMCFileReader.h"
#include <map>
#include <string>

class HepMCFileReader;

namespace edm
{
  class MCFileSource : public ExternalInputSource {
  public:
    MCFileSource(const ParameterSet &, const InputSourceDescription &);
   virtual ~MCFileSource();
// the following cannot be used anymore since an explicit InputSourceDescription is needed ?? FM
/*
    MCFileSource(const std::string& processName);
    /// Specify the file to be read. FIXME: should be done by the "configuration"
    MCFileSource(const std::string& filename, const std::string& processName);
*/

  private:
   
   virtual bool produce(Event &e);
    void clear();
    
    HepMCFileReader * reader_;
    
    HepMC::GenEvent  *evt;
    	
  };
} 

#endif
