#ifndef Input_MCFileSource_h
#define Input_MCFileSource_h

// $Id: MCFileSource.h,v 1.7 2007/06/19 13:48:03 weng Exp $

/** \class MCFileSource
 *
 * Reads in HepMC events
 * Joanna Weng & Filip Moortgat 08/2005 
 ***************************************/

#include "FWCore/Sources/interface/ExternalInputSource.h"

#include "IOMC/Input/interface/HepMCFileReader.h"


namespace HepMC{
  class GenEvent;
}


namespace edm
{
  class Event;
  class ParameterSet;
  class InputSourceDescription;

  class MCFileSource : public ExternalInputSource {
  public:
    MCFileSource(const ParameterSet& pset, const InputSourceDescription& desc);
    virtual ~MCFileSource();

  private:
    virtual bool produce(Event &e);
    void clear();
    
    HepMCFileReader *reader_;
    HepMC::GenEvent *evt_;
    HepMCFileReader::FileMode mode_;
  };
} 

#endif
