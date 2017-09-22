#include "Rtypes.h" 

#include <PhysicsTools/NanoAOD/interface/FlatTable.h>
#include <PhysicsTools/NanoAOD/interface/MergableCounterTable.h>
#include <PhysicsTools/NanoAOD/interface/UniqueString.h>
#include "DataFormats/Common/interface/Wrapper.h"

namespace PhysicsTools_NanoAOD {
    struct dictionary {
        edm::Wrapper<FlatTable> w_table;
        edm::Wrapper<MergableCounterTable> w_mtable;
        edm::Wrapper<UniqueString> w_ustr;
    };
}
