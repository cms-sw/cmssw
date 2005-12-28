#ifndef Modules_EmptySource_h
#define Modules_EmptySource_h

/*----------------------------------------------------------------------
$Id: EmptySource.h,v 1.1 2005/10/17 19:22:41 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/GeneratedInputSource.h"

namespace edm {
  class EmptySource : public GeneratedInputSource {
  public:
    explicit EmptySource(ParameterSet const&, InputSourceDescription const&);
    ~EmptySource();
  private:
    virtual void produce(Event &);
  };
}
#endif
