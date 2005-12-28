#ifndef Modules_EmptySource_h
#define Modules_EmptySource_h

/*----------------------------------------------------------------------
$Id: EmptySource.h,v 1.2 2005/12/28 00:52:54 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/GeneratedInputSource.h"

namespace edm {
  class EmptySource : public GeneratedInputSource {
  public:
    explicit EmptySource(ParameterSet const&, InputSourceDescription const&);
    ~EmptySource();
  private:
    virtual bool produce(Event &);
  };
}
#endif
