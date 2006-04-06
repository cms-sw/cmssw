#ifndef Framework_EDInputSource_h
#define Framework_EDInputSource_h

/*----------------------------------------------------------------------
$Id: EDInputSource.h,v 1.7 2006/04/04 22:15:21 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/FileCatalog.h"
#include <vector>
#include <string>

namespace edm {
  class InputSourceDescription;
  class ParameterSet;
  class EDInputSource : public InputSource {
  public:
    explicit EDInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~EDInputSource();

    int remainingEvents() const {return remainingEvents_;}
    std::vector<std::string> const& fileNames() const {return catalog_.fileNames();}
    InputFileCatalog& catalog() {return catalog_;}

  protected:

    void repeat() {remainingEvents_ = maxEvents();}
    virtual std::auto_ptr<EventPrincipal> read();


  private:
    virtual std::auto_ptr<EventPrincipal> readOneEvent() = 0;
    
    InputFileCatalog catalog_;
    int remainingEvents_;
  };
}
#endif
