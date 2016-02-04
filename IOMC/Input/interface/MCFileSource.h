#ifndef Input_MCFileSource_h
#define Input_MCFileSource_h

// $Id: MCFileSource.h,v 1.9 2009/12/01 19:23:11 fabstoec Exp $

/** \class MCFileSource
 *
 * Reads in HepMC events
 * Joanna Weng & Filip Moortgat 08/2005 
 ***************************************/

#include "FWCore/Sources/interface/ExternalInputSource.h"


class HepMCFileReader;

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
    bool useExtendedAscii_;
  };
} 

#endif
