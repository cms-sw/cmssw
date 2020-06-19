/*
 *  Copyright (c), 2017, Adrien Devresse
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>

const std::string FILE_NAME("select_partial_string.h5");
const std::string DATASET_NAME("message");

// Create a dataset name "dset" of double 4x6
//
int main(void) {
    using namespace HighFive;
    try {
        // Create a new file using the default property lists.
        File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);

        {
            // We have a set of string
            std::vector<std::string> values = {
                "Cat", "Dog",  "Hello", "Tree", "World",     "Plane",
                ", ",  "你好", "Tea",   "Moon", "صباح جميل", "Spaceship",
            };

            // let's create a dataset
            DataSet dataset = file.createDataSet<std::string>(
                DATASET_NAME, DataSpace::From(values));

            // and write them
            dataset.write(values);
        }

        {
            DataSet dataset = file.getDataSet(DATASET_NAME);

            // now let's read back by cherry pick our interesting string
            std::vector<std::string> result;
            // we select only element N° 2 and 5
            dataset.select(ElementSet({2, 4, 6, 7, 6, 10})).read(result);

            // and display it
            for (auto i : result) {
                std::cout << i << " ";
            }
            std::cout << "\n";
        }

    } catch (Exception& err) {
        // catch and print any HDF5 error
        std::cerr << err.what() << std::endl;
    }

    return 0; // successfully terminated
}
