/*
 *  Copyright (c), 2017, Adrien Devresse
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#include <iostream>

#undef H5_USE_BOOST
#define H5_USE_BOOST

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <highfive/H5File.hpp>

using namespace HighFive;

const std::string FILE_NAME("boost_ublas_double.h5");
const std::string DATASET_NAME("dset");

const size_t size_x = 10;
const size_t size_y = 10;

int main(void) {

    try {
        typedef typename boost::numeric::ublas::matrix<double> Matrix;

        // create a 10x10 matrix
        Matrix mat(size_x, size_y);

        // fill it
        for (std::size_t i = 0; i < size_x; ++i) {
            mat(i, i) = i;
        }

        // Create a new HDF5 file
        File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);

        // create a new dataset with the 10x10 Matrix dimension
        DataSet dataset =
            file.createDataSet<double>(DATASET_NAME, DataSpace::From(mat));

        // write it
        dataset.write(mat);

        // now, let read it back
        Matrix result;
        dataset.read(result);

        // print what we read
        std::cout << "Matrix result:\n" << result << std::endl;

    } catch (Exception& err) {
        // catch and print any HDF5 error
        std::cerr << err.what() << std::endl;
    }

    return 0; // successfully terminated
}
