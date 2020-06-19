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

#include <boost/multi_array.hpp>
#include <highfive/H5File.hpp>

using namespace HighFive;

const std::string FILE_NAME("boost_multiarray_example.h5");
const std::string DATASET_NAME("dset");
const size_t size_x = 10;
const size_t size_y = 3;

// Create a 2D dataset 10x3 of double with boost multi array
// and write it to a file
int main(void) {

    try {

        boost::multi_array<double, 2> my_array(boost::extents[size_x][size_y]);

        for (size_t i = 0; i < size_x; ++i) {
            for (size_t j = 0; j < size_y; ++j) {
                my_array[i][j] = double(j + i * size_y);
            }
        }

        // we create a new hdf5 file
        File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);

        // let's create our dataset of the size of the boost::multi_array
        DataSet dataset =
            file.createDataSet<double>(DATASET_NAME, DataSpace::From(my_array));

        // we fill it
        dataset.write(my_array);

    } catch (Exception& err) {
        // catch and print any HDF5 error
        std::cerr << err.what() << std::endl;
    }

    return 0; // successfully terminated
}
