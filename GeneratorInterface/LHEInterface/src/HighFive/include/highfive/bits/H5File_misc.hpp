/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5FILE_MISC_HPP
#define H5FILE_MISC_HPP

#include <string>
#include "../H5Exception.hpp"
#include "../H5File.hpp"
#include "../H5Utility.hpp"

#include <H5Fpublic.h>

namespace HighFive {

namespace {

// libhdf5 uses a preprocessor trick on their oflags
// we can not declare them constant without a mapper
inline int convert_open_flag(int openFlags) {
    int res_open = 0;
    if (openFlags & File::ReadOnly)
        res_open |= H5F_ACC_RDONLY;
    if (openFlags & File::ReadWrite)
        res_open |= H5F_ACC_RDWR;
    if (openFlags & File::Create)
        res_open |= H5F_ACC_CREAT;
    if (openFlags & File::Truncate)
        res_open |= H5F_ACC_TRUNC;
    if (openFlags & File::Excl)
        res_open |= H5F_ACC_EXCL;
    return res_open;
}
}  // namespace

inline File::File(const std::string& filename, int openFlags,
                  const Properties& fileAccessProps)
    : _filename(filename) {

    openFlags = convert_open_flag(openFlags);

    int createMode = openFlags & (H5F_ACC_TRUNC | H5F_ACC_EXCL);
    int openMode = openFlags & (H5F_ACC_RDWR | H5F_ACC_RDONLY);
    bool mustCreate = createMode > 0;
    bool openOrCreate = (openFlags & H5F_ACC_CREAT) > 0;

    // open is default. It's skipped only if flags require creation
    // If open fails it will try create() if H5F_ACC_CREAT is set
    if (!mustCreate) {
        // Silence open errors if create is allowed
        std::unique_ptr<SilenceHDF5> silencer;
        if (openOrCreate) silencer.reset(new SilenceHDF5());

        _hid = H5Fopen(_filename.c_str(), openMode, fileAccessProps.getId());

        if (isValid()) return;  // Done

        if (openOrCreate) {
            // Will attempt to create ensuring wont clobber any file
            createMode = H5F_ACC_EXCL;
        } else {
            HDF5ErrMapper::ToException<FileException>(
                std::string("Unable to open file " + _filename));
        }
    }

    if ((_hid = H5Fcreate(_filename.c_str(), createMode, H5P_DEFAULT,
                          fileAccessProps.getId())) < 0) {
        HDF5ErrMapper::ToException<FileException>(
            std::string("Unable to create file " + _filename));
    }
}

inline const std::string& File::getName() const {
    return _filename;
}

inline void File::flush() {
    if (H5Fflush(_hid, H5F_SCOPE_GLOBAL) < 0) {
        HDF5ErrMapper::ToException<FileException>(
            std::string("Unable to flush file " + _filename));
    }
}
}  // namespace HighFive

#endif  // H5FILE_MISC_HPP
