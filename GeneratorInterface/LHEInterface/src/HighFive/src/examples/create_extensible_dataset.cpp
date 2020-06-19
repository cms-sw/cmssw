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

const std::string FILE_NAME("create_extensible_dataset_example.h5");
const std::string DATASET_NAME("dset");

// Create a dataset name "dset" of double 4x6
//
int main(void) {
    using namespace HighFive;
    try {
        // Create a new file using the default property lists.
        File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);

        // Create a dataspace with initial shape and max shape
        DataSpace dataspace = DataSpace({4, 5}, {17, DataSpace::UNLIMITED});

        // Use chunking
        DataSetCreateProps props;
        props.add(Chunking(std::vector<hsize_t>{2, 2}));

        // Create the dataset
        DataSet dataset = file.createDataSet(DATASET_NAME, dataspace,
                                             AtomicType<double>(), props);

        // Write into the initial part of the dataset
        double t1[3][1] = {{2.0}, {2.0}, {4.0}};
        dataset.select({0, 0}, {3, 1}).write(t1);

        // Resize the dataset to a larger size
        dataset.resize({4, 6});

        // Write into the new part of the dataset
        double t2[1][3] = {{4.0, 8.0, 6.0}};
        dataset.select({3, 3}, {1, 3}).write(t2);

        // now we read it back
        std::vector<std::vector<double>> result;
        dataset.read(result);

        // we print it out and see:
        // 2 0 0 0 0 0
        // 2 0 0 0 0 0
        // 4 0 0 0 0 0
        // 0 0 0 4 8 6
        for (auto row : result) {
            for (auto col : row)
                std::cout << " " << col;
            std::cout << std::endl;
        }

    } catch (const Exception& err) {
        // catch and print any HDF5 error
        std::cerr << err.what() << std::endl;
    }

    return 0; // successfully terminated
}
