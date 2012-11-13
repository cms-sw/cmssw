#ifndef GENERS_ARRAYADAPTOR_HH_
#define GENERS_ARRAYADAPTOR_HH_

#include <cstddef>
#include <cassert>
#include "Alignment/Geners/interface/IOException.hh"

#include "Alignment/Geners/interface/ProcessItem.hh"
#include "Alignment/Geners/interface/InsertContainerItem.hh"
#include "Alignment/Geners/interface/IOIsContiguous.hh"
#include "Alignment/Geners/interface/binaryIO.hh"

namespace gs {
    template <class Stream, class State, class Item, class Stage>
    struct GenericWriter;

    template <class Stream, class State, class Item, class Stage>
    struct GenericReader;

    template<typename T>
    class ArrayAdaptor
    {
    public:
        typedef T value_type;
        typedef const T* const_iterator;

        inline ArrayAdaptor(const T* indata, const std::size_t sz,
                            const bool writeItemClassId = true)
            : data_(indata), size_(sz), writetemCl_(writeItemClassId)
              {if (sz) assert(data_);}

        inline std::size_t size() const {return size_;}
        inline const_iterator begin() const {return data_;}
        inline const_iterator end() const {return data_ + size_;}
        inline bool writeItemClassId() const {return writetemCl_;}
        inline const T& operator[](const std::size_t index) const
            {return data_[index];}
        inline T& operator[](const std::size_t index)
            {return (const_cast<T*>(data_))[index];}
        inline T& at(const std::size_t index)
        {
            if (index >= size_) throw gs::IOOutOfRange(
                "gs::ArrayAdaptor::at: index out of range");
            return (const_cast<T*>(data_))[index];
        }

    private:
        ArrayAdaptor();

        const T* data_;
        std::size_t size_;
        bool writetemCl_;
    };

    template <class T>
    struct IOIsContiguous<ArrayAdaptor<T> >
    {enum {value = 1};};

    template <class T>
    struct IOIsContiguous<const ArrayAdaptor<T> >
    {enum {value = 1};};

    template <class T>
    struct IOIsContiguous<volatile ArrayAdaptor<T> >
    {enum {value = 1};};

    template <class T>
    struct IOIsContiguous<const volatile ArrayAdaptor<T> >
    {enum {value = 1};};

    template <class T>
    struct InsertContainerItem<ArrayAdaptor<T> >
    {
        typedef ArrayAdaptor<T> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t itemNumber)
            {obj.at(itemNumber) = item;}
    };

    template <class T>
    struct InsertContainerItem<volatile ArrayAdaptor<T> >
    {
        typedef ArrayAdaptor<T> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t itemNumber)
            {obj.at(itemNumber) = item;}
    };

    // Ignore array size I/O. The size is provided in the constructor.
    // Of course, it still has to be written somewhere, but that code
    // is external w.r.t. the array adaptor.
    template <class Stream, class State, class T>
    struct GenericWriter<Stream, State, ArrayAdaptor<T>, InContainerSize>
    {
        inline static bool process(std::size_t, Stream& os, State*,
                                   const bool processClassId)
            {return true;}
    };

    template <class Stream, class State, class T>
    struct GenericReader<Stream, State, ArrayAdaptor<T>, InContainerSize>
    {
        inline static bool process(std::size_t, Stream& os, State*,
                                   const bool processClassId)
            {return true;}
    };

    template <class Stream, class State, class T>
    struct GenericWriter<Stream, State, ArrayAdaptor<T>, InPODArray>
    {
        inline static bool process(const ArrayAdaptor<T>& a, Stream& os,
                                   State*, bool)
        {
            const std::size_t len = a.size();
            if (len)
                write_pod_array(os, &a[0], len);
            return !os.fail();
        }
    };

    template <class Stream, class State, class T>
    struct GenericReader<Stream, State, ArrayAdaptor<T>, InPODArray>
    {
        inline static bool process(ArrayAdaptor<T>& a, Stream& s, State*, bool)
        {
            const std::size_t len = a.size();
            if (len)
                read_pod_array(s, &a[0], len);
            return !s.fail();
        }
    };

    template <class Stream, class State, class T>
    struct GenericWriter<Stream, State, ArrayAdaptor<T>, InContainerHeader>
    {
        typedef ArrayAdaptor<T> Container;

        inline static bool process(const Container& c, Stream& os, State*,
                                   const bool processClassId)
        {
            bool status = processClassId ?
                ClassId::makeId<Container>().write(os) : true;
            if (status && !(IOTraits<T>::IsPOD && 
                            IOTraits<Container>::IsContiguous) && 
                c.writeItemClassId())
                status = ClassId::makeId<T>().write(os);
            return status;
        }
    };

    template <class Stream, class State, class T>
    struct GenericReader<Stream, State, ArrayAdaptor<T>, InContainerHeader>
    {
        typedef ArrayAdaptor<T> Container;

        inline static bool process(Container& a, Stream& is, State* state,
                                   const bool processClassId)
        {
            bool status = true;
            if (processClassId)
            {
                ClassId id(is, 1);
                const ClassId& current = ClassId::makeId<Container>();
                status = (id.name() == current.name());
            }
            if (status)
            {
                if (!(IOTraits<T>::IsPOD && IOTraits<Container>::IsContiguous))
                    if (a.writeItemClassId())
                    {
                        ClassId id(is, 1);
                        state->push_back(id);
                    }
            }
            return status;
        }
    };

    template <class Stream, class State, class T>
    struct GenericReader<Stream, State, ArrayAdaptor<T>, InContainerFooter>
    {
        typedef ArrayAdaptor<T> Container;

        inline static bool process(Container& a, Stream&, State* state, bool)
        {
            if (!(IOTraits<T>::IsPOD && IOTraits<Container>::IsContiguous))
                if (a.writeItemClassId())
                    state->pop_back();
            return true;
        }
    };
}

gs_specialize_template_id_T(gs::ArrayAdaptor, 0, 1)

#endif // GENERS_ARRAYADAPTOR_HH_

