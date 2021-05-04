#ifndef DataFormats_Common_HLTPathStatus_h
#define DataFormats_Common_HLTPathStatus_h

/** \class edm::HLTPathStatus
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
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/HLTenums.h"
#include <cassert>
#include <cstdint>

namespace edm {
  class HLTPathStatus {
  private:
    /// packed status of trigger path [unsigned char is too small]
    uint16_t status_;
    // bits 0- 1 (0-    3): HLT state
    // bits 2-16 (0-16383): index of module on path making path decision

  public:
    /// constructor
    HLTPathStatus(const hlt::HLTState state = hlt::Ready, const unsigned int index = 0) : status_(index * 4 + state) {
      assert(((int)state) < 4);
      assert(index < 16384);
    }

    /// get state of path
    hlt::HLTState state() const { return (static_cast<hlt::HLTState>(status_ % 4)); }
    /// get index of module giving the status of this path
    /// Nota Bene: if a Path or EndPath is empty (that is, it does not contain any ED module),
    /// index will be 0, even if there is no "0th module" responsible for the status of the path
    unsigned int index() const { return (static_cast<unsigned int>(status_ / 4)); }
    /// reset this path
    void reset() { status_ = 0; }

    /// was this path run?
    bool wasrun() const { return (state() != hlt::Ready); }
    /// has this path accepted the event?
    bool accept() const { return (state() == hlt::Pass); }
    /// has this path encountered an error (exception)?
    bool error() const { return (state() == hlt::Exception); }
  };
}  // namespace edm

#endif  // DataFormats_Common_HLTPathStatus_h
