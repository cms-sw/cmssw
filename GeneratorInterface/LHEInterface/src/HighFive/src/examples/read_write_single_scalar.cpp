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

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>

const std::string FILE_NAME("read_write_scalar.h5");
const std::string DATASET_NAME("single_scalar");

// Create a dataset name "single_scalar"
// which contains only the perfect integer number "42"
//
int main(void) {
    using namespace HighFive;
    try {
        // Create a new file using the default property lists.
        File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);

        int perfect_number = 42;

        // Create the dataset
        DataSet dataset = file.createDataSet<double>(
            DATASET_NAME, DataSpace::From(perfect_number));

        // write it
        dataset.write(perfect_number);

        // flush everything
        file.flush();

        // let's read it back
        int potentially_perfect_number;

        dataset.read(potentially_perfect_number);

        std::cout << "perfect number: " << potentially_perfect_number
                  << std::endl;

    } catch (Exception& err) {
        // catch and print any HDF5 error
        std::cerr << err.what() << std::endl;
    }

    return 0; // successfully terminated
}
