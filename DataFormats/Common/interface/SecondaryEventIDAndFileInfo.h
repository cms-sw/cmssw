#ifndef DataFormats_Common_SecondaryEventIDAndFileInfo_h
#define DataFormats_Common_SecondaryEventIDAndFileInfo_h

#include "DataFormats/Provenance/interface/EventID.h"

// forward declarations
namespace edm {
  class SecondaryEventIDAndFileInfo {
  public:
    SecondaryEventIDAndFileInfo() : eventID_(), fileNameHash_(0U) {}
    SecondaryEventIDAndFileInfo(EventID const& evID, size_t fNameHash) : eventID_(evID), fileNameHash_(fNameHash) {}
    EventID const& eventID() const {
      return eventID_;
    }
    size_t fileNameHash() const {
      return fileNameHash_;
    }
  private:
    EventID eventID_;
    size_t fileNameHash_;
  };
}
#endif
