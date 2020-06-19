/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5UTILS_HPP
#define H5UTILS_HPP

// internal utilities functions
#include <cstddef> // __GLIBCXX__
#include <exception>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#ifdef H5_USE_BOOST
#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#endif

#include <H5public.h>

#ifndef H5_USE_CXX11
#if ___cplusplus >= 201103L
#define H5_USE_CXX11 1
#else
#define H5_USE_CXX11 0
#endif
#endif

namespace HighFive {

namespace details {

// determine at compile time number of dimensions of in memory datasets
template <typename T>
struct array_dims {
    static const size_t value = 0;
};

template <typename T>
struct array_dims<std::vector<T> > {
    static const size_t value = 1 + array_dims<T>::value;
};

template <typename T>
struct array_dims<T*> {
    static const size_t value = 1 + array_dims<T>::value;
};

template <typename T, std::size_t N>
struct array_dims<T[N]> {
    static const size_t value = 1 + array_dims<T>::value;
};

#ifdef H5_USE_BOOST
template <typename T, std::size_t Dims>
struct array_dims<boost::multi_array<T, Dims> > {
    static const size_t value = Dims;
};

template <typename T>
struct array_dims<boost::numeric::ublas::matrix<T> > {
    static const size_t value = 2;
};
#endif

// determine recursively the size of each dimension of a N dimension vector
template <typename T>
void get_dim_vector_rec(const T& vec, std::vector<size_t>& dims) {
    (void)dims;
    (void)vec;
}

template <typename T>
void get_dim_vector_rec(const std::vector<T>& vec, std::vector<size_t>& dims) {
    dims.push_back(vec.size());
    get_dim_vector_rec(vec[0], dims);
}

template <typename T>
std::vector<size_t> get_dim_vector(const std::vector<T>& vec) {
    std::vector<size_t> dims;
    get_dim_vector_rec(vec, dims);
    return dims;
}

// determine at compile time recursively the basic type of the data
template <typename T>
struct type_of_array {
    typedef T type;
};

template <typename T>
struct type_of_array<std::vector<T> > {
    typedef typename type_of_array<T>::type type;
};

#ifdef H5_USE_BOOST
template <typename T, std::size_t Dims>
struct type_of_array<boost::multi_array<T, Dims> > {
    typedef typename type_of_array<T>::type type;
};

template <typename T>
struct type_of_array<boost::numeric::ublas::matrix<T> > {
    typedef typename type_of_array<T>::type type;
};
#endif

template <typename T>
struct type_of_array<T*> {
    typedef typename type_of_array<T>::type type;
};

template <typename T, std::size_t N>
struct type_of_array<T[N]> {
    typedef typename type_of_array<T>::type type;
};

// check if the type is a container ( only vector supported for now )
template <typename>
struct is_container {
    static const bool value = false;
};

template <typename T>
struct is_container<std::vector<T> > {
    static const bool value = true;
};

// check if the type is a basic C-Array
// check if the type is a container ( only vector supported for now )
template <typename>
struct is_c_array {
    static const bool value = false;
};

template <typename T>
struct is_c_array<T*> {
    static const bool value = true;
};

template <typename T, std::size_t N>
struct is_c_array<T[N]> {
    static const bool value = true;
};



// convertor function for hsize_t -> size_t when hsize_t != size_t
template<typename Size>
inline std::vector<std::size_t> to_vector_size_t(std::vector<Size> vec){
    static_assert(std::is_same<Size, std::size_t>::value == false, " hsize_t != size_t mandatory here");
    std::vector<size_t> res(vec.size());
    std::copy(vec.begin(), vec.end(), res.begin());
    return res;
}

// convertor function for hsize_t -> size_t when size_t == hsize_t
inline std::vector<std::size_t> to_vector_size_t(std::vector<std::size_t> vec){
    return vec;
}

// shared ptr portability
// was used pre-C++11, kept for compatibility
namespace Mem {
    using namespace std;
} // end Mem

} // end details
}

#endif // H5UTILS_HPP
