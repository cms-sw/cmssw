#ifndef GENERS_TUPLEIO_HH_
#define GENERS_TUPLEIO_HH_

#include "Alignment/Geners/interface/CPP11_config.hh"
#ifdef CPP11_STD_AVAILABLE

#include <tuple>
#include <cassert>

#include "Alignment/Geners/interface/GenericIO.hh"

#define gs_specialize_template_help_tuple(qualifyer, name, version) /**/      \
    template<typename... Args>                                                \
    struct ClassIdSpecialization<qualifyer name <Args...> >                   \
    {inline static ClassId classId(const bool pt=false)                       \
    {return ClassId(tuple_class_name< name <Args...> >(#name), version, pt);}};

#define gs_specialize_template_id_tuple(name, version) /**/                         \
namespace gs {                                                                      \
    gs_specialize_template_help_tuple(GENERS_EMPTY_TYPE_QUALIFYER_, name, version)\
    gs_specialize_template_help_tuple(const, name, version)                         \
    gs_specialize_template_help_tuple(volatile, name, version)                      \
    gs_specialize_template_help_tuple(const volatile, name, version)                \
}

namespace gs {
    template<typename Item>
    Item notImplementedItemCreator();

    template <typename... Args>
    struct IOIsTuple<std::tuple<Args...> >
    {
        enum {value = 1};
    };

    namespace Private
    {
        template<typename Tuple, unsigned long N>
        struct TupleClassIdCycler
        {
            inline static void collectClassIds(std::string& os)
            {
                TupleClassIdCycler<Tuple, N-1>::collectClassIds(os);
                if (N > 1) os += ',';
                os += ClassIdSpecialization<typename 
                    std::tuple_element<N-1, Tuple>::type>::classId().id();
            }

            inline static bool dumpClassIds(std::ostream& os)
            {
                return TupleClassIdCycler<Tuple, N-1>::dumpClassIds(os) &&
                  ClassIdSpecialization<typename 
                    std::tuple_element<N-1, Tuple>::type>::classId().write(os);
            }

            inline static void fillClassIdVector(std::vector<ClassId>* vec)
            {
                TupleClassIdCycler<Tuple, N-1>::fillClassIdVector(vec);
                vec->push_back(ClassIdSpecialization<
                    typename std::tuple_element<N-1, Tuple>::type>::classId());
            }
        };

        template<typename Tuple>
        struct TupleClassIdCycler<Tuple, 0UL>
        {
            inline static void collectClassIds(std::string&) {}
            inline static bool dumpClassIds(std::ostream&) {return true;}
            inline static void fillClassIdVector(std::vector<ClassId>* vec)
            {
                assert(vec);
                vec->clear();
            }
        };

        template<typename Tuple, unsigned long N>
        struct TupleIOCycler
        {
            template <typename Stream, typename State>
            inline static bool write(const Tuple& s, Stream& os, State* st,
                                     const bool processClassId)
            {
                return TupleIOCycler<Tuple, N-1>::write(
                    s, os, st, processClassId) &&
                    process_const_item<GenericWriter2>(
                        std::get<N-1>(s), os, st, processClassId);
            }

            template <typename Stream, typename StateVec>
            inline static bool read(Tuple* s, Stream& os, StateVec& st,
                                    const bool processClassId)
            {
                return TupleIOCycler<Tuple, N-1>::read(
                    s, os, st, processClassId) &&
                    process_item<GenericReader2>(
                        std::get<N-1>(*s), os, &st[N-1], processClassId);
            }

            inline static void clearPointers(Tuple* s)
            {
                TupleIOCycler<Tuple, N-1>::clearPointers(s);
                clearIfPointer(std::get<N-1>(*s));
            }
        };

        template<typename Tuple>
        struct TupleIOCycler<Tuple, 0UL>
        {
            template <typename Stream, typename State>
            inline static bool write(const Tuple&, Stream&, State*, bool)
                {return true;}

            template <typename Stream, typename StateVec>
            inline static bool read(Tuple*, Stream&, StateVec&, bool)
                {return true;}

            inline static void clearPointers(Tuple*) {}
        };

        inline std::vector<std::string> make_default_tuple_columns(
            const unsigned long N)
        {
            std::vector<std::string> names_;
            names_.reserve(N);
            for (unsigned long i=0; i<N; ++i)
            {
                std::ostringstream os;
                os << 'c';
                os << i;
                names_.push_back(os.str());
            }
            return names_;
        }
    }

    template<class T>
    std::string tuple_class_name(const char* templateName)
    {
        assert(templateName);
        std::string os(templateName);
        if (std::tuple_size<T>::value)
        {
            os += '<';
            Private::TupleClassIdCycler<
                T, std::tuple_size<T>::value>::collectClassIds(os);
            os += '>';
        }
        return os;
    }

    template<unsigned long N>
    const std::vector<std::string>& default_tuple_columns()
    {
        static const std::vector<std::string> names_(
            Private::make_default_tuple_columns(N));
        return names_;
    }
}

gs_specialize_template_id_tuple(std::tuple, 0)

namespace gs {
    template <class Stream, class State, class T>
    struct GenericWriter<Stream, State, T,
                         Int2Type<IOTraits<int>::ISTUPLE> >
    {
        inline static bool process(const T& s, Stream& os, State* st,
                                   const bool processClassId)
        {
            static const ClassId current(ClassId::makeId<T>());
            return (processClassId ? current.write(os) : true) &&
                Private::TupleIOCycler<T, std::tuple_size<T>::value>::write(
                    s, os, st, false);
        }
    };

    template <class Stream, class State, class T>
    struct GenericReader<Stream, State, T,
                         Int2Type<IOTraits<int>::ISTUPLE> >
    {
        inline static bool readIntoPtr(T*& ptr, Stream& str, State* s,
                                       const bool processClassId)
        {
            std::unique_ptr<T> myptr;
            if (ptr == 0)
            {
                myptr = std::unique_ptr<T>(new T());
                Private::TupleIOCycler<
                    T, std::tuple_size<T>::value>::clearPointers(myptr.get());
            }
            std::vector<std::vector<ClassId> > itemIds;
            if (processClassId)
            {
                static const ClassId current(ClassId::makeId<T>());
                ClassId id(str, 1);
                current.ensureSameName(id);
                id.templateParameters(&itemIds);
                assert(itemIds.size() == std::tuple_size<T>::value);
            }
            else
            {
                assert(!s->empty());
                s->back().templateParameters(&itemIds);
                if (itemIds.size() != std::tuple_size<T>::value)
                {
                    std::string err("In gs::GenericReader::readIntoPtr: "
                                    "bad class id for std::tuple on the "
                                    "class id stack: ");
                    err += s->back().id();
                    throw IOInvalidData(err);
                }
            }
            const bool status = Private::TupleIOCycler<
                T, std::tuple_size<T>::value>::read(
                    ptr ? ptr : myptr.get(), str, itemIds, false);
            if (status && ptr == 0)
                ptr = myptr.release();
            return status;
        }

        inline static bool process(T& s, Stream& os, State* st,
                                   const bool processClassId)
        {
            T* ps = &s;
            return readIntoPtr(ps, os, st, processClassId);
        }
    };
}

#endif // CPP11_STD_AVAILABLE
#endif // GENERS_TUPLEIO_HH_

