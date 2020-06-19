/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5Group.hpp>

#define BOOST_TEST_MAIN HighFiveTest
#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>

using namespace HighFive;

int argc = boost::unit_test::framework::master_test_suite().argc;
char** argv = boost::unit_test::framework::master_test_suite().argv;

struct MpiFixture {
    MpiFixture() { MPI_Init(&argc, &argv); }

    ~MpiFixture() { MPI_Finalize(); }
};

BOOST_GLOBAL_FIXTURE(MpiFixture);

typedef boost::mpl::list<int, unsigned int, long, unsigned long, unsigned char,
                         char, float, double, long long, unsigned long long>
    numerical_test_types;

template <typename T>
struct ContentGenerate {
    ContentGenerate(T init_val = T(0), T inc_val = T(1) + T(1) / T(10))
        : _init(init_val), _inc(inc_val) {}

    T operator()() {
        T ret = _init;
        _init += _inc;
        return ret;
    }

    T _init, _inc;
};

template <>
struct ContentGenerate<char> {
    ContentGenerate() : _init('a') {}

    char operator()() {
        char ret = _init;
        if (++_init >= ('a' + 26))
            _init = 'a';
        return ret;
    }

    char _init;
};

template <>
struct ContentGenerate<std::string> {
    ContentGenerate() {}

    std::string operator()() {
        ContentGenerate<char> gen;
        std::string random_string;
        const size_t size_string = std::rand() % 1000;
        random_string.resize(size_string);
        std::generate(random_string.begin(), random_string.end(), gen);
        return random_string;
    }
};

template <typename T>
void selectionArraySimpleTestParallel() {

    int mpi_rank, mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    typedef typename std::vector<T> Vector;

    std::ostringstream filename;
    filename << "h5_rw_select_parallel_test_" << typeid(T).name() << "_test.h5";

    const size_t size_x = mpi_size;
    const size_t offset_x = mpi_rank, count_x = mpi_size - mpi_rank;

    const std::string DATASET_NAME("dset");

    Vector values(size_x);

    ContentGenerate<T> generator;
    std::generate(values.begin(), values.end(), generator);

    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate,
              MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

    DataSet dataset =
        file.createDataSet<T>(DATASET_NAME, DataSpace::From(values));

    dataset.write(values);

    file.flush();

    // read it back
    Vector result;
    std::vector<size_t> offset;
    offset.push_back(offset_x);
    std::vector<size_t> size;
    size.push_back(count_x);

    Selection slice = dataset.select(offset, size);

    BOOST_CHECK_EQUAL(slice.getSpace().getDimensions()[0], size_x);
    BOOST_CHECK_EQUAL(slice.getMemSpace().getDimensions()[0], count_x);

    slice.read(result);

    BOOST_CHECK_EQUAL(result.size(), count_x);

    for (size_t i = offset_x; i < count_x; ++i) {
        // std::cout << result[i] << " ";
        BOOST_CHECK_EQUAL(values[i + offset_x], result[i]);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(selectionArraySimple, T, numerical_test_types) {

    selectionArraySimpleTestParallel<T>();
}
