#ifndef GENERS_FORWARD_LISTIO_HH_
#define GENERS_FORWARD_LISTIO_HH_

#include "Alignment/Geners/interface/CPP11_config.hh"
#ifdef CPP11_STD_AVAILABLE

#include <forward_list>
#include "Alignment/Geners/interface/GenericIO.hh"

// std::forward_list does not have the size() method. Because of this,
// we can not use the I/O machinery developed in "GenericIO.hh" for
// standard containers. Instead, we will designate std::forward_list
// as an external type and will handle it separately.
//
gs_declare_template_external_TT(std::forward_list)
gs_specialize_template_id_TT(std::forward_list, 0, 1)

namespace gs {
    // Assuming that we want to write the list once and potentially
    // read it back many times, we will write it out in the reverse
    // order. This is because it is easy to extend the list from the
    // front but not from the back.
    //
    template <class Stream, class State, class T>
    struct GenericWriter<Stream, State, std::forward_list<T>,
                         Int2Type<IOTraits<int>::ISEXTERNAL> >
    {
        inline static bool process(const std::forward_list<T>& s, Stream& os,
                                   State* p2, const bool processClassId)
        {
            typedef typename std::forward_list<T>::const_iterator Iter;

            bool status = processClassId ? 
                ClassId::makeId<std::forward_list<T> >().write(os) : true;
            if (status)
            {
                const Iter listend = s.end();
                std::size_t sz = 0;
                for (Iter it=s.begin(); it!=listend; ++it, ++sz) {;}
                write_pod(os, sz);
                if (sz)
                {
                    status = ClassId::makeId<T>().write(os);
                    std::vector<Iter> iters(sz);
                    sz = 0;
                    for (Iter it=s.begin(); it!=listend; ++it, ++sz)
                        iters[sz] = it;
                    for (long long number=sz-1; number>=0 && status; --number)
                        status = process_const_item<GenericWriter2>(
                            *iters[number], os, p2, false);
                }
            }
            return status && !os.fail();
        }
    };

    template <class T>
    struct InsertContainerItem<std::forward_list<T> >
    {
        inline static void insert(std::forward_list<T>& obj, const T& item,
                                  const std::size_t /* itemNumber */)
            {obj.push_front(item);}
    };

    template <class Stream, class State, class T>
    struct GenericReader<Stream, State, std::forward_list<T>,
                         Int2Type<IOTraits<int>::ISEXTERNAL> >
    {
        inline static bool readIntoPtr(std::forward_list<T>*& ptr, Stream& is,
                                       State* p2, const bool processClassId)
        {
            if (processClassId)
            {
                ClassId id(is, 1);
                const ClassId& curr = ClassId::makeId<std::forward_list<T> >();
                curr.ensureSameName(id);
            }
            CPP11_auto_ptr<std::forward_list<T> > myptr;
            if (ptr == 0)
                myptr = CPP11_auto_ptr<std::forward_list<T> >(
                    new std::forward_list<T>());
            else
                ptr->clear();
            std::size_t sz = 0;
            read_pod(is, &sz);
            bool itemStatus = true;
            if (sz)
            {
                ClassId itemId(is, 1);
                p2->push_back(itemId);
                std::forward_list<T>* nzptr = ptr ? ptr : myptr.get();
                try
                {
                    for (std::size_t i=0; i < sz && itemStatus; ++i)
                        itemStatus = GenericReader<
                             Stream,State,std::forward_list<T>,InContainerCycle
                        >::process(*nzptr, is, p2, i);
                }
                catch (...)
                {
                    p2->pop_back();
                    throw;
                }
            }
            const bool success = itemStatus && !is.fail();
            if (success && ptr == 0)
                ptr = myptr.release();
            return success;
        }

        inline static bool process(std::forward_list<T>& s, Stream& is,
                                   State* st, const bool processClassId)
        {
            std::forward_list<T>* ps = &s;
            return readIntoPtr(ps, is, st, processClassId);
        }
    };
}

#endif // CPP11_STD_AVAILABLE
#endif // GENERS_FORWARD_LISTIO_HH_

