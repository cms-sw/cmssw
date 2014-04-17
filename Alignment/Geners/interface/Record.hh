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

#ifndef SWIG
    template <typename T>
    inline ArchiveRecord<T> Record(const T& object, const std::string& name,
                                   const char* category)
    {
        return ArchiveRecord<T>(object, name.c_str(), category);
    }

    template <typename T>
    inline ArchiveRecord<T> Record(const T& object, const char* name,
                                   const std::string& category)
    {
        return ArchiveRecord<T>(object, name, category.c_str());
    }

    template <typename T>
    inline ArchiveRecord<T> Record(const T& object, const std::string& name,
                                   const std::string& category)
    {
        return ArchiveRecord<T>(object, name.c_str(), category.c_str());
    }
#endif

    //
    // ValueRecord makes a copy of the object and stores it internally
    //
    template <typename T>
    inline ArchiveValueRecord<T> ValueRecord(const T& object, const char* name,
                                             const char* category)
    {
        return ArchiveValueRecord<T>(object, name, category);
    }

#ifndef SWIG
    template <typename T>
    inline ArchiveValueRecord<T> ValueRecord(const T& object,
                                             const std::string& name,
                                             const char* category)
    {
        return ArchiveValueRecord<T>(object, name.c_str(), category);
    }

    template <typename T>
    inline ArchiveValueRecord<T> ValueRecord(const T& object, const char* name,
                                             const std::string& category)
    {
        return ArchiveValueRecord<T>(object, name, category.c_str());
    }

    template <typename T>
    inline ArchiveValueRecord<T> ValueRecord(const T& object,
                                             const std::string& name,
                                             const std::string& category)
    {
        return ArchiveValueRecord<T>(object, name.c_str(), category.c_str());
    }
#endif
}

#endif // GENERS_RECORD_HH_

