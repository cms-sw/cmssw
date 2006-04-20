#ifndef Common_HLTPathStatus_h
#define Common_HLTPathStatus_h

/** \class HLTPathStatus
 *
 *  The status of a single HLT trigger (single trigger path consisting
 *  of modules on the path).  Initially, the status is Ready (meaning
 *  that this trigger path has not run yet for this event).  If all
 *  modules on the path pass (accept) the event, then the state is
 *  Pass. If any module on the path fails (rejects) the event, then
 *  the state of the whole trigger path is Fail. If any module on the
 *  path throws an unhandled error, then the trigger state is
 *  Exception.  In the latter two cases, the position of the module in
 *  the path issuing the (first) fail/error is recorded.  The Fw skips
 *  further processing of modules along this path, ie, path processing
 *  is aborted.
 *
 *  $Date: 2006/04/19 20:12:04 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/HLTenums.h"
#include <cassert>

namespace edm
{
  class HLTPathStatus {

  private:
    unsigned char status_; // packed status: bits 0-1: state, 
                           // bits 2-n: index of aborting module

  public:

    HLTPathStatus(const hlt::HLTState state = hlt::Ready, const unsigned int abort = 0)
    : status_(abort*4+state) { assert(abort<64); }

    hlt::HLTState state() const {return ((hlt::HLTState) (status_ % 4));}
    unsigned int  abort() const {return ((unsigned int)  (status_ / 4));}

    void reset() {status_=0;}

    bool wasrun() const {return (state() != hlt::Ready);}
    bool accept() const {return (state() == hlt::Pass );}
    bool error()  const {return (state() == hlt::Exception);}

  };
}

#endif // Common_HLTPathStatus_h
