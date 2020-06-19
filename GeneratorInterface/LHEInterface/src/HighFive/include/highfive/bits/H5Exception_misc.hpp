/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5EXCEPTION_MISC_HPP
#define H5EXCEPTION_MISC_HPP

#include "../H5Exception.hpp"

#include <H5Epublic.h>
#include <cstdlib>

namespace HighFive {

struct HDF5ErrMapper {

    template <typename ExceptionType>
    static inline herr_t stackWalk(unsigned n, const H5E_error2_t* err_desc,
                                   void* client_data) {
        ExceptionType** e_iter = static_cast<ExceptionType**>(client_data);
        (void)n;

        char* major_err = H5Eget_major(err_desc->maj_num);
        char* minor_err = H5Eget_minor(err_desc->min_num);

        std::string err_string("(");
        err_string += major_err;
        err_string += ") ";
        err_string += minor_err;

        free(major_err);
        free(minor_err);

        ExceptionType* e = new ExceptionType(err_string);
        e->_err_major = err_desc->maj_num;
        e->_err_minor = err_desc->min_num;
        (*e_iter)->_next.reset(e);
        *e_iter = e;
        return 0;
    }

    template <typename ExceptionType>
    static inline void ToException(const std::string& prefix_msg) {

        hid_t err_stack = H5Eget_current_stack();
        if (err_stack >= 0) {
            ExceptionType e("");
            ExceptionType* e_iter = &e;

            H5Ewalk2(err_stack, H5E_WALK_UPWARD,
                     &HDF5ErrMapper::stackWalk<ExceptionType>, &e_iter);
            H5Eclear2(err_stack);

            const char* next_err_msg = (e.nextException() != NULL)
                                           ? (e.nextException()->what())
                                           : ("");

            e.setErrorMsg(prefix_msg + " " + next_err_msg);
            throw e;
        }
        // throw generic error, unrecognized error
        throw ExceptionType(prefix_msg + ": Unknown HDF5 error");
    }
};
}

#endif // H5OBJECT_MISC_HPP
