#ifndef GENERS_VALARRAYIO_HH_
#define GENERS_VALARRAYIO_HH_

#include <valarray>
#include "Alignment/Geners/interface/GenericIO.hh"

namespace gs {
    template <class T>
    struct IOIsContiguous<std::valarray<T> >
    {enum {value = 1};};

    template <class T>
    struct IOIsContiguous<const std::valarray<T> >
    {enum {value = 1};};

    template <class T>
    struct IOIsContiguous<volatile std::valarray<T> >
    {enum {value = 1};};

    template <class T>
    struct IOIsContiguous<const volatile std::valarray<T> >
    {enum {value = 1};};

    template <class T>
    struct InsertContainerItem<std::valarray<T> >
    {
        typedef std::valarray<T> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t itemNumber)
            {obj[itemNumber] = item;}
    };

    template <class T>
    struct InsertContainerItem<volatile std::valarray<T> >
    {
        typedef std::valarray<T> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t itemNumber)
            {obj[itemNumber] = item;}
    };

    // Need to specialize behavior in the header because
    // there is no "clear" method in std::valarray
    template <class Stream, class State, class T>
    struct GenericReader<Stream, State, std::valarray<T>, InContainerHeader>
    {
        typedef std::valarray<T> Container;
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

    namespace Private {
        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct iterate_const_container<Visitor,std::valarray<T>,Arg1,Arg2>
        {
            static bool process(const std::valarray<T>& v, Arg1& a1,
                                Arg2* p2, const std::size_t len)
            {
                bool itemStatus = true;
                for (std::size_t i=0; i<len && itemStatus; ++i)
                    itemStatus = process_const_item<Visitor>(v[i],a1,p2,false);
                return itemStatus;
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct iterate_container<Visitor,std::valarray<T>,Arg1,Arg2>
        {
            static bool process(std::valarray<T>& obj, Arg1& a1,
                                Arg2* p2, const std::size_t newSize)
            {
                obj.resize(newSize);
                bool itemStatus = true;
                for (std::size_t i=0; i<newSize && itemStatus; ++i)
                    itemStatus = Visitor<
                        Arg1,Arg2,std::valarray<T>,InContainerCycle>::process(
                            obj, a1, p2, i);
                return itemStatus;
            }
        };
    }
}

gs_specialize_template_id_T(std::valarray, 0, 1)

#endif // GENERS_VALARRAYIO_HH_

