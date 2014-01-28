#ifndef Modules_DummySource_h
#define Modules_DummySource_h

/*----------------------------------------------------------------------
$Id: DummySource.h,v 1.3 2005/12/28 21:49:52 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/GeneratedInputSource.h"

namespace edm {
  class DummySource : public GeneratedInputSource {
  public:
    explicit DummySource(ParameterSet const&, InputSourceDescription const&);
    ~DummySource();
  private:
    virtual bool produce(Event &);
  };
}
#endif
