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

using namespace HighFive;

const std::string FILE_NAME("dataset_integer.h5");
const std::string DATASET_NAME("dset");
const size_t size_dataset = 20;

// create a dataset 1D from a vector of string
void write_dataset() {

    // we create a new hdf5 file
    File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);

    std::vector<int> data(size_dataset);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = int(i);
    }

    // let's create a dataset of native integer with the size of the vector
    // 'data'
    DataSet dataset =
        file.createDataSet<int>(DATASET_NAME, DataSpace::From(data));

    // let's write our vector of int to the HDF5 dataset
    dataset.write(data);
}

// read our data back
void read_dataset() {
    // we open the existing hdf5 file we created before
    File file(FILE_NAME, File::ReadOnly);

    std::vector<int> read_data;

    // we get the dataset
    DataSet dataset = file.getDataSet(DATASET_NAME);

    // we convert the hdf5 dataset to a single dimension vector
    dataset.read(read_data);

    for (size_t i = 0; i < read_data.size(); ++i) {
        std::cout << read_data[i] << " ";
    }
    std::cout << "\n";
}

int main(void) {

    try {
        write_dataset();
        read_dataset();

    } catch (Exception& err) {
        // catch and print any HDF5 error
        std::cerr << err.what() << std::endl;
    }

    return 0; // successfully terminated
}
