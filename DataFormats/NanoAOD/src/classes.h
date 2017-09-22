#include "Rtypes.h" 

#include <DataFormats/NanoAOD/interface/FlatTable.h>
#include <DataFormats/NanoAOD/interface/MergeableCounterTable.h>
#include <DataFormats/NanoAOD/interface/UniqueString.h>
#include "DataFormats/Common/interface/Wrapper.h"

namespace DataFormats_NanoAOD {
    struct dictionary {
        edm::Wrapper<FlatTable> w_table;
        edm::Wrapper<MergeableCounterTable> w_mtable;
        edm::Wrapper<UniqueString> w_ustr;
    };
}
