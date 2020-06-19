/*
 *  Copyright (c), 2017, Adrien Devresse
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#include <iostream>
#include <string>
#include <vector>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

const std::string FILE_NAME("create_dataset_example.h5");
const std::string DATASET_NAME("dset");

// Create a dataset name "dset" of double 4x6
//
int main(void) {
    using namespace HighFive;
    try {
        // Create a new file using the default property lists.
        File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);

        // Define the size of our dataset: 2x6
        std::vector<size_t> dims(2);
        dims[0] = 2;
        dims[1] = 6;

        // Create the dataset
        DataSet dataset =
            file.createDataSet<double>(DATASET_NAME, DataSpace(dims));

        double data[2][6] = {{1.1, 2.2, 3.3, 4.4, 5.5, 6.6},
                             {11.11, 12.12, 13.13, 14.14, 15.15, 16.16}};

        // write it
        dataset.write(data);

    } catch (Exception& err) {
        // catch and print any HDF5 error
        std::cerr << err.what() << std::endl;
    }

    return 0; // successfully terminated
}
