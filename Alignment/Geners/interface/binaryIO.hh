//=========================================================================
// binaryIO.hh
//
// "Low-level" utility API for reading/writing data in a binary form
// (this particular form is not platform independent). All "geners" I/O
// should be performed through one of the functions defined in this file,
// there should be no direct user manipulation of C++ streams. In this
// way possible future I/O modifications will be isolated and restricted
// to this facility.
//
// I. Volobouev
// April 2009
//=========================================================================

#ifndef GENERS_BINARYIO_HH_
#define GENERS_BINARYIO_HH_

#include <vector>
#include <cassert>
#include <iostream>

#include "Alignment/Geners/interface/CPP11_shared_ptr.hh"
#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"
#include "Alignment/Geners/interface/ClassId.hh"
#include "Alignment/Geners/interface/IOException.hh"

namespace gs {
    // The following functions perform binary I/O of built-in types.
    // Note that all of them are "void". It is assumed that the
    // top-level code will check the status of the stream after
    // several I/O operations.
    template <typename T>
    inline void write_pod(std::ostream& of, const T& pod)
    {
        of.write(reinterpret_cast<const char*>(&pod), sizeof(T));
    }

    template <typename T>
    inline void read_pod(std::istream& in, T* pod)
    {
        assert(pod);
        in.read(reinterpret_cast<char*>(pod), sizeof(T));
    }

    template <typename T>
    inline void write_pod_array(std::ostream& of, const T* pod,
                                const unsigned long len)
    {
        if (len)
        {
            assert(pod);
            of.write(reinterpret_cast<const char*>(pod), len*sizeof(T));
        }
    }

    template <typename T>
    inline void read_pod_array(std::istream& in, T* pod,
                               const unsigned long len)
    {
        if (len)
        {
            assert(pod);
            in.read(reinterpret_cast<char*>(pod), len*sizeof(T));
        }
    }

    // String is treated as a pod vector. This will be guaranteed
    // to work correctly in the C++11 standard (the current standard
    // does not specify that the characters must be stored contuguously
    // inside the string -- however, this is always true in practice).
    template <typename T>
    inline void write_string(std::ostream& of, const std::basic_string<T>& v)
    {
        const unsigned long sz = v.size();
        write_pod(of, sz);
        if (sz)
            write_pod_array(of, v.data(), sz);
    }

    template <typename T>
    inline void read_string(std::istream& in, std::basic_string<T>* pv)
    {
        assert(pv);
        unsigned long vlen = 0UL;
        read_pod(in, &vlen);
        if (vlen)
        {
            pv->resize(vlen);
            read_pod_array(in, const_cast<T*>(pv->data()), vlen);
        }
        else
            pv->clear();
    }

    // Specialization of POD-based I/O functions so that they work
    // as if std::string is a POD
    template<>
    inline void write_pod<std::string>(std::ostream& of, const std::string& s)
    {
        write_string<char>(of, s);
    }

    template<>
    inline void read_pod<std::string>(std::istream& in, std::string* ps)
    {
        read_string<char>(in, ps);
    }

    template<>
    inline void write_pod_array<std::string>(std::ostream& of,
                                             const std::string* pod,
                                             const unsigned long len)
    {
        if (len)
        {
            assert(pod);
            for (unsigned long i=0; i<len; ++i)
                write_string<char>(of, pod[i]);
        }
    }

    template<>
    inline void read_pod_array<std::string>(std::istream& in,
                                            std::string* pod,
                                            const unsigned long len)
    {
        if (len)
        {
            assert(pod);
            for (unsigned long i=0; i<len; ++i)
                read_string<char>(in, pod + i);
        }
    }

    template <typename T>
    inline void write_pod_vector(std::ostream& of, const std::vector<T>& v)
    {
        const unsigned long sz = v.size();
        write_pod(of, sz);
        if (sz)
            write_pod_array(of, &v[0], sz);
    }

    template <typename T>
    inline void read_pod_vector(std::istream& in, std::vector<T>* pv)
    {
        assert(pv);
        unsigned long vlen = 0UL;
        read_pod(in, &vlen);
        if (in.fail()) throw IOReadFailure(
            "In gs::read_pod_vector: input stream failure");
        if (vlen)
        {
            pv->resize(vlen);
            read_pod_array(in, &((*pv)[0]), vlen);
        }
        else
            pv->clear();
    }

    // The following functions perform binary I/O on objects
    // which have write/read/restore functions. Compared to
    // calling the corresponding class methods directly, these
    // function take care of writing out the class identifier.
    //
    template <typename T>
    inline bool write_obj(std::ostream& of, const T& obj)
    {
        return obj.classId().write(of) && obj.write(of);
    }

    template <typename T>
    inline CPP11_auto_ptr<T> read_obj(std::istream& in)
    {
        const ClassId id(in, 1);
        return CPP11_auto_ptr<T>(T::read(id, in));
    }

    template <typename T>
    inline void restore_obj(std::istream& in, T* obj)
    {
        assert(obj);
        const ClassId id(in, 1);
        T::restore(id, in, obj);
    }

    // The following function is templated upon the reader factory
    template <typename Reader>
    inline CPP11_auto_ptr<typename Reader::value_type>
    read_base_obj(std::istream& in, const Reader& f)
    {
        typedef typename Reader::value_type T;
        const ClassId id(in, 1);
        return CPP11_auto_ptr<T>(f.read(id, in));
    }

    // The following function assumes that the array contains actual
    // objects rather than pointers
    template <typename T>
    inline bool write_obj_array(std::ostream& of, const T* arr,
                                const unsigned long len)
    {
        bool status = true;
        if (len)
        {
            assert(arr);
            status = arr[0].classId().write(of);
            for (unsigned long i=0; i<len && status; ++i)
                status = arr[i].write(of);
        }
        return status;
    }

    // The following assumes that the array contains actual objects
    // and that class T has the "restore" function
    template <typename T>
    inline void read_placed_obj_array(std::istream& in, T* arr,
                                      const unsigned long len)
    {
        if (len)
        {
            assert(arr);
            const ClassId id(in, 1);
            for (unsigned long i=0; i<len; ++i)
                T::restore(id, in, arr + i);
        }
    }

    // The following assumes that the array contains a bunch of
    // shared pointers and that class T has the "read" function
    template <typename T>
    inline void read_heap_obj_array(std::istream& in,
                                    CPP11_shared_ptr<T>* arr,
                                    const unsigned long len)
    {
        if (len)
        {
            assert(arr);
            const ClassId id(in, 1);
            for (unsigned long i=0; i<len; ++i)
            {
                T* obj = T::read(id, in);
                arr[i] = CPP11_shared_ptr<T>(obj);
            }
        }
    }

    // The following function is templated upon the reader factory
    template <typename Reader>
    inline void read_base_obj_array(
        std::istream& in, const Reader& f,
        CPP11_shared_ptr<typename Reader::value_type>* arr,
        const unsigned long len)
    {
        typedef typename Reader::value_type T;
        if (len)
        {
            assert(arr);
            const ClassId id(in, 1);
            for (unsigned long i=0; i<len; ++i)
            {
                T* obj = f.read(id, in);
                arr[i] = CPP11_shared_ptr<T>(obj);
            }
        }
    }

    // The following assumes that the vector contains actual objects
    template <typename T>
    inline bool write_obj_vector(std::ostream& of, const std::vector<T>& v)
    {
        const unsigned long sz = v.size();
        write_pod(of, sz);
        bool status = !of.fail();
        if (sz && status)
            status = write_obj_array(of, &v[0], sz);
        return status;
    }

    // The following assumes that the vector contains actual objects
    template <typename T>
    inline void read_placed_obj_vector(std::istream& in, std::vector<T>* pv)
    {
        unsigned long vlen = 0UL;
        read_pod(in, &vlen);
        if (in.fail()) throw IOReadFailure(
            "In gs::read_placed_obj_vector: input stream failure");
        assert(pv);
        if (vlen)
        {
            pv->resize(vlen);
            read_placed_obj_array(in, &(*pv)[0], vlen);
        }
        else
            pv->clear();
    }

    // The following assumes that the vector contains a bunch of
    // shared pointers
    template <typename T>
    inline void read_heap_obj_vector(std::istream& in,
                                     std::vector<CPP11_shared_ptr<T> >* pv)
    {
        unsigned long vlen = 0UL;
        read_pod(in, &vlen);
        if (in.fail()) throw IOReadFailure(
            "In gs::read_heap_obj_vector: input stream failure");
        assert(pv);
        if (vlen)
        {
            pv->resize(vlen);
            return read_heap_obj_array(in, &(*pv)[0], vlen);
        }
        else
            pv->clear();
    }

    // The following assumes that the vector contains actual objects.
    // This function is less efficient than others, but it has to be
    // used sometimes if one wants to have a vector of objects without
    // default constructors.
    //
    template <typename T>
    inline void read_heap_obj_vector_as_placed(std::istream& in,
                                               std::vector<T>* pv)
    {
        unsigned long vlen = 0UL;
        read_pod(in, &vlen);
        if (in.fail()) throw IOReadFailure(
            "In gs::read_heap_obj_vector_as_placed: input stream failure");
        assert(pv);
        pv->clear();
        if (vlen)
        {
            const ClassId id(in, 1);
            pv->reserve(vlen);
            for (unsigned long i=0; i<vlen; ++i)
            {
                CPP11_auto_ptr<T> obj(T::read(id, in));
                pv->push_back(*obj);
            }
        }
    }

    // The following function is templated upon the reader factory
    template <typename Reader>
    inline void read_base_obj_vector(
        std::istream& in, const Reader& f,
        std::vector<CPP11_shared_ptr<typename Reader::value_type> >* pv)
    {
        unsigned long vlen = 0UL;
        read_pod(in, &vlen);
        if (in.fail()) throw IOReadFailure(
            "In gs::read_base_obj_vector: input stream failure");
        assert(pv);
        if (vlen)
        {
            pv->resize(vlen);
            read_base_obj_array(in, f, &(*pv)[0], vlen);
        }
        else
            pv->clear();
    }
}

#endif // GENERS_BINARYIO_HH_

