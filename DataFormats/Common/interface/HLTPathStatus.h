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
 *  Exception.  For the latter two cases, the Fw skips further
 *  processing of modules along this path, ie, path processing is
 *  aborted.
 *
 *  The index of the module on the path, 0 to n-1 for a path with n
 *  modules issuing the decision for the path is recorded.  For
 *  accepted events, this is simply the index of the last module on
 *  the path, ie, n-1.
 *
 *  Note that n is limited, due to packing, to at most 2^(16-2)=16384.
 *
 *  $Date: 2006/07/13 15:12:07 $
 *  $Revision: 1.3 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/HLTenums.h"
#include <boost/cstdint.hpp>
#include <cassert>

namespace edm
{
  class HLTPathStatus {

  private:
    uint16_t status_;    // packed status [unsigned char is too small!]
    // bits 0- 1 (0-    3): HLT state
    // bits 2-16 (0-16383): index of module on path making path decision

  public:

    HLTPathStatus(const hlt::HLTState state = hlt::Ready, const unsigned int index = 0)
    : status_(index*4+state) {
      assert (state<4);
      assert (index<16384);
    }

    hlt::HLTState state() const {return (static_cast<hlt::HLTState>(status_ % 4));}
    unsigned int  index() const {return (static_cast<unsigned int >(status_ / 4));}

    void reset() {status_=0;}

    bool wasrun() const {return (state() != hlt::Ready);}
    bool accept() const {return (state() == hlt::Pass );}
    bool error()  const {return (state() == hlt::Exception);}

  };
}

#endif // Common_HLTPathStatus_h
