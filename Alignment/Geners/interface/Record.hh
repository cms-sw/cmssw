#ifndef GENERS_RECORD_HH_
#define GENERS_RECORD_HH_

#include "Alignment/Geners/interface/ArchiveRecord.hh"

namespace gs {
    template <typename T>
    inline ArchiveRecord<T> Record(const T& object, const char* name,
                                   const char* category)
    {
        return ArchiveRecord<T>(object, name, category);
    }

    template <typename T>
    inline ArchiveValueRecord<T> ValueRecord(const T& object, const char* name,
                                             const char* category)
    {
        return ArchiveValueRecord<T>(object, name, category);
    }
}

#endif // GENERS_RECORD_HH_

