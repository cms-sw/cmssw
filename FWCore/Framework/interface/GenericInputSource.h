#ifndef Framework_GenericInputSource_h
#define Framework_GenericInputSource_h

/*----------------------------------------------------------------------
$Id: GenericInputSource.h,v 1.2 2006/02/07 07:51:41 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/InputSource.h"
#include <vector>
#include <string>

namespace edm {
  class InputSourceDescription;
  class ParameterSet;
  class GenericInputSource : public InputSource {
  public:
    explicit GenericInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~GenericInputSource();

    int remainingEvents() const {return remainingEvents_;}
    std::vector<std::string> const& fileNames() const{return fileNames_;}

  protected:

    void repeat() {remainingEvents_ = maxEvents();}
    virtual std::auto_ptr<EventPrincipal> read();


  private:
    virtual std::auto_ptr<EventPrincipal> readOneEvent() = 0;
    
    std::vector<std::string> fileNames_;
    int remainingEvents_;
  };
}
#endif
