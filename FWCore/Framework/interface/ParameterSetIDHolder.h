
// The only purpose of this is to make it easier to use
// forward declarations when passing around ParameterSetID's.
// ParameterSetID itself is a typedef and hard to forward
// declare.

#include "DataFormats/Provenance/interface/ParameterSetID.h"

namespace edm {
   namespace eventsetup {

      class ParameterSetIDHolder {
      public:
         ParameterSetIDHolder(ParameterSetID const& psetID) : psetID_(psetID) { }
         ParameterSetID const& psetID() const { return psetID_; }
         bool operator<(ParameterSetIDHolder const& other) const { return psetID() < other.psetID(); }
         bool operator==(ParameterSetIDHolder const& other) const { return psetID() == other.psetID(); }
      private:
         ParameterSetID psetID_;
      };
   }
}
