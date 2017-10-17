#include "Rtypes.h" 

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/NanoAOD/interface/MergeableCounterTable.h"
#include "DataFormats/NanoAOD/interface/UniqueString.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace DataFormats_NanoAOD {
    struct dictionary {
        nanoaod::FlatTable table;
        edm::Wrapper<nanoaod::FlatTable> w_table;
        edm::Wrapper<nanoaod::MergeableCounterTable> w_mtable;
        edm::Wrapper<nanoaod::UniqueString> w_ustr;
    };
}
