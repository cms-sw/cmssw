#ifndef GENERS_ARRAYIO_HH_
#define GENERS_ARRAYIO_HH_

#include "Alignment/Geners/interface/CPP11_array.hh"
#include "Alignment/Geners/interface/GenericIO.hh"

namespace gs {
    template<class T, std::size_t N>
    struct ClassIdSpecialization<CPP11_array<T,N> >
    {inline static ClassId classId(const bool isPtr=false)
    {return ClassId(stack_container_name<T,N>("std::array"), 0, isPtr);}};

    template<class T, std::size_t N>
    struct ClassIdSpecialization<const CPP11_array<T,N> >
    {inline static ClassId classId(const bool isPtr=false)
    {return ClassId(stack_container_name<T,N>("std::array"), 0, isPtr);}};

    template<class T, std::size_t N>
    struct ClassIdSpecialization<volatile CPP11_array<T,N> >
    {inline static ClassId classId(const bool isPtr=false)
    {return ClassId(stack_container_name<T,N>("std::array"), 0, isPtr);}};

    template<class T, std::size_t N>
    struct ClassIdSpecialization<const volatile CPP11_array<T,N> >
    {inline static ClassId classId(const bool isPtr=false)
    {return ClassId(stack_container_name<T,N>("std::array"), 0, isPtr);}};

    template <class T, std::size_t N>
    struct IOIsContiguous<CPP11_array<T,N> >
    {enum {value = 1};};

    template <class T, std::size_t N>
    struct IOIsContiguous<const CPP11_array<T,N> >
    {enum {value = 1};};

    template <class T, std::size_t N>
    struct IOIsContiguous<volatile CPP11_array<T,N> >
    {enum {value = 1};};

    template <class T, std::size_t N>
    struct IOIsContiguous<const volatile CPP11_array<T,N> >
    {enum {value = 1};};

    // Need to specialize behavior in the header because
    // there is no "clear" method in std::array
    template <class Stream, class State, class T, std::size_t N>
    struct GenericReader<Stream, State, CPP11_array<T,N>, InContainerHeader>
    {
        typedef CPP11_array<T,N> Container;
        inline static bool process(Container& a, Stream& is, State* state,
                                   const bool processClassId)
        {
            if (processClassId)
            {
                static const ClassId current(ClassId::makeId<Container>());
                ClassId id(is, 1);
                current.ensureSameName(id);
            }
            if (!IOTraits<T>::IsPOD)
            {
                ClassId id(is, 1);
                state->push_back(id);
            }
            return true;
        }
    };

    // Ignore array size I/O. The size is built into
    // the type definition anyway
    template <class Stream, class State, class T, std::size_t N>
    struct GenericWriter<Stream, State, CPP11_array<T,N>, InContainerSize>
    {
        inline static bool process(std::size_t, Stream& os, State*,
                                   const bool processClassId)
            {return true;}
    };

    template <class Stream, class State, class T, std::size_t N>
    struct GenericReader<Stream, State, CPP11_array<T,N>, InContainerSize>
    {
        inline static bool process(std::size_t, Stream& os, State*,
                                   const bool processClassId)
            {return true;}
    };

    template <class Stream, class State, class T, std::size_t N>
    struct GenericWriter<Stream, State, CPP11_array<T,N>, InPODArray>
    {
        typedef CPP11_array<T,N> Array;
        inline static bool process(const Array& a, Stream& os, State*, bool)
        {
            write_pod_array(os, &a[0], N);
            return !os.fail();
        }
    };

    template <class Stream, class State, class T, std::size_t N>
    struct GenericReader<Stream, State, CPP11_array<T,N>, InPODArray>
    {
        typedef CPP11_array<T,N> Array;
        inline static bool process(Array& a, Stream& is, State*, bool)
        {
            read_pod_array(is, &a[0], N);
            return !is.fail();
        }
    };

    template <class T, std::size_t N>
    struct InsertContainerItem<CPP11_array<T,N> >
    {
        typedef CPP11_array<T,N> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t itemNumber)
            {obj.at(itemNumber) = item;}
    };

    template <class T, std::size_t N>
    struct InsertContainerItem<volatile CPP11_array<T,N> >
    {
        typedef CPP11_array<T,N> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t itemNumber)
            {obj.at(itemNumber) = item;}
    };
}

#endif // GENERS_ARRAYIO_HH_

