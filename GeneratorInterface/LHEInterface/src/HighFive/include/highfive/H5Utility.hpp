/*
 *  Copyright (c), 2017, Juan Hernando <juan.hernando@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */

#ifndef H5UTILITY_HPP
#define H5UTILITY_HPP

#include <H5Epublic.h>

namespace HighFive {

/// \brief Utility class to disable HDF5 stack printing inside a scope.
class SilenceHDF5 {
public:
    inline SilenceHDF5()
        : _client_data(0)
    {
        H5Eget_auto2(H5E_DEFAULT, &_func, &_client_data);
        H5Eset_auto2(H5E_DEFAULT, 0, 0);
    }

    inline ~SilenceHDF5() { H5Eset_auto2(H5E_DEFAULT, _func, _client_data); }
private:
    H5E_auto2_t _func;
    void* _client_data;
};
}

#endif // H5UTIL_HPP
