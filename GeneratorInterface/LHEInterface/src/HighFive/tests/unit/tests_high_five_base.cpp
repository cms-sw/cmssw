/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>

#ifdef HIGHFIVE_CPP11_ENABLE
#include <memory>
#endif

#include <stdio.h>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>
#include <highfive/H5Utility.hpp>

#define BOOST_TEST_MAIN HighFiveTest

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

#ifdef H5_USE_BOOST
#include <boost/multi_array.hpp>
#endif

using namespace HighFive;

typedef boost::mpl::list<float, double> floating_numerics_test_types;
typedef boost::mpl::list<int, unsigned int, long, unsigned long, unsigned char,
                         char, float, double, long long, unsigned long long>
    numerical_test_types;
typedef boost::mpl::list<int, unsigned int, long, unsigned long, unsigned char,
                         char, float, double, std::string>
    dataset_test_types;

template <typename T, typename Func>
void generate2D(T* table, size_t x, size_t y, Func& func) {
    for (size_t i = 0; i < x; i++) {
        for (size_t j = 0; j < y; j++) {
            table[i][j] = func();
        }
    }
}

template <typename T, typename Func>
void generate2D(std::vector<std::vector<T>>& vec, size_t x, size_t y,
                Func& func) {
    vec.resize(x);
    for (size_t i = 0; i < x; i++) {
        vec[i].resize(y);
        for (size_t j = 0; j < y; j++) {
            vec[i][j] = func();
        }
    }
}

template <typename T>
struct ContentGenerate {
    explicit ContentGenerate(T init_val = T(0),
                             T inc_val = T(1) + T(1) / T(10))
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
        std::mt19937_64 rgen;
        rgen.seed(88);
        std::uniform_int_distribution<int> int_dist(0, 1000);
        const size_t size_string = int_dist(rgen);

        random_string.resize(size_string);
        std::generate(random_string.begin(), random_string.end(), gen);
        return random_string;
    }
};

BOOST_AUTO_TEST_CASE(HighFiveBasic) {
    const std::string FILE_NAME("h5tutr_dset.h5");
    const std::string DATASET_NAME("dset");

    // Create a new file using the default property lists.
    File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);

    BOOST_CHECK_EQUAL(file.getName(), FILE_NAME);

    // Create the data space for the dataset.
    std::vector<size_t> dims;
    dims.push_back(4);
    dims.push_back(6);

    DataSpace dataspace(dims);

    // check if the dataset exist
    bool dataset_exist = file.exist(DATASET_NAME + "_double");
    BOOST_CHECK_EQUAL(dataset_exist, false);

    // Create a dataset with double precision floating points
    DataSet dataset_double = file.createDataSet(
        DATASET_NAME + "_double", dataspace, AtomicType<double>());

    BOOST_CHECK_EQUAL(file.getObjectName(0), DATASET_NAME + "_double");

    {
        // check if it exist again
        dataset_exist = file.exist(DATASET_NAME + "_double");
        BOOST_CHECK_EQUAL(dataset_exist, true);

        // and also try to recreate it to the sake of exception testing
        BOOST_CHECK_THROW(
            {
                DataSet fail_duplicated = file.createDataSet(
                    DATASET_NAME + "_double", dataspace, AtomicType<double>());
            },
            DataSetException);
    }

    DataSet dataset_size_t =
        file.createDataSet<size_t>(DATASET_NAME + "_size_t", dataspace);
}

BOOST_AUTO_TEST_CASE(HighFiveSilence) {
    // Setting up a buffer for stderr so we can detect if the stack trace
    // was disabled
    fflush(stderr);
    char buffer[1024];
    memset(buffer, 0, sizeof(char) * 1024);
    setvbuf(stderr, buffer, _IOLBF, 1023);

    try {
        SilenceHDF5 silence;
        File file("nonexistent", File::ReadOnly);
    } catch (const FileException&) {
    }
    BOOST_CHECK_EQUAL(buffer[0], '\0');

    // restore the dyn allocated buffer
    // or using stderr will segfault when buffer get out of scope
    fflush(stderr);
    setvbuf(stderr, NULL, _IONBF, 0);
}

BOOST_AUTO_TEST_CASE(HighFiveOpenMode) {
    const std::string FILE_NAME("openmodes.h5");
    const std::string DATASET_NAME("dset");

    std::remove(FILE_NAME.c_str());

    // Attempt open file only ReadWrite should fail (wont create)
    BOOST_CHECK_THROW(
        { File file(FILE_NAME, File::ReadWrite); },
        FileException);

    // But with Create flag should be fine
    { File file(FILE_NAME, File::ReadWrite | File::Create); }

    // But if its there and exclusive is given, should fail
    BOOST_CHECK_THROW(
        { File file(FILE_NAME, File::ReadWrite | File::Excl); },
        FileException);
    // ReadWrite and Excl flags are fine together (posix)
    std::remove(FILE_NAME.c_str());
    { File file(FILE_NAME, File::ReadWrite | File::Excl); }
    // All three are fine as well (as long as the file does not exist)
    std::remove(FILE_NAME.c_str());
    { File file(FILE_NAME, File::ReadWrite | File::Create | File::Excl); }

    // Just a few combinations are incompatible, detected by hdf5lib
    BOOST_CHECK_THROW(
        { File file(FILE_NAME, File::Truncate | File::Excl); },
        FileException);

    std::remove(FILE_NAME.c_str());
    BOOST_CHECK_THROW(
        { File file(FILE_NAME, File::Truncate | File::Excl); },
        FileException);

    // But in most cases we will truncate and that should always work
    { File file(FILE_NAME, File::Truncate); }
    std::remove(FILE_NAME.c_str());
    { File file(FILE_NAME, File::Truncate); }

    // Last but not least, defaults should be ok
    { File file(FILE_NAME); }     // ReadOnly
    { File file(FILE_NAME, 0); }  // force empty-flags, does open without flags
}


BOOST_AUTO_TEST_CASE(HighFiveGroupAndDataSet) {
    const std::string FILE_NAME("h5_group_test.h5");
    const std::string DATASET_NAME("dset");
    const std::string CHUNKED_DATASET_NAME("chunked_dset");
    const std::string CHUNKED_DATASET_SMALL_NAME("chunked_dset_small");
    const std::string GROUP_NAME1("/group1");
    const std::string GROUP_NAME2("group2");
    const std::string GROUP_NESTED_NAME("group_nested");

    {
        // Create a new file using the default property lists.
        File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);

        // absolute group
        file.createGroup(GROUP_NAME1);
        // nested group absolute
        file.createGroup(GROUP_NAME1 + "/" + GROUP_NESTED_NAME);
        // relative group
        Group g1 = file.createGroup(GROUP_NAME2);
        // relative group
        Group nested = g1.createGroup(GROUP_NESTED_NAME);

        // Create the data space for the dataset.
        std::vector<size_t> dims;
        dims.push_back(4);
        dims.push_back(6);

        DataSpace dataspace(dims);

        DataSet dataset_absolute = file.createDataSet(
            GROUP_NAME1 + "/" + GROUP_NESTED_NAME + "/" + DATASET_NAME,
            dataspace, AtomicType<double>());

        DataSet dataset_relative =
            nested.createDataSet(DATASET_NAME, dataspace, AtomicType<double>());

        DataSetCreateProps goodChunking;
        goodChunking.add(Chunking(std::vector<hsize_t>{2, 2}));

        // will fail because exceeds dimensions
        DataSetCreateProps badChunking0;
        badChunking0.add(Chunking(std::vector<hsize_t>{10, 10}));

        DataSetCreateProps badChunking1;
        badChunking1.add(Chunking(std::vector<hsize_t>{1, 1, 1}));

        BOOST_CHECK_THROW(file.createDataSet(CHUNKED_DATASET_NAME, dataspace,
                                             AtomicType<double>(),
                                             badChunking0),
                          DataSetException);

        BOOST_CHECK_THROW(file.createDataSet(CHUNKED_DATASET_NAME, dataspace,
                                             AtomicType<double>(),
                                             badChunking1),
                          DataSetException);

        // here we use the other signature
        DataSet dataset_chunked = file.createDataSet<float>(
            CHUNKED_DATASET_NAME, dataspace, goodChunking);

        // Here we resize to smaller than the chunking size
        DataSet dataset_chunked_small = file.createDataSet<float>(
            CHUNKED_DATASET_SMALL_NAME, dataspace, goodChunking);

        dataset_chunked_small.resize({1, 1});
    }
    // read it back
    {
        File file(FILE_NAME, File::ReadOnly);
        Group g1 = file.getGroup(GROUP_NAME1);
        Group g2 = file.getGroup(GROUP_NAME2);
        Group nested_group2 = g2.getGroup(GROUP_NESTED_NAME);

        DataSet dataset_absolute = file.getDataSet(
            GROUP_NAME1 + "/" + GROUP_NESTED_NAME + "/" + DATASET_NAME);
        BOOST_CHECK_EQUAL(4, dataset_absolute.getSpace().getDimensions()[0]);

        DataSet dataset_relative = nested_group2.getDataSet(DATASET_NAME);
        BOOST_CHECK_EQUAL(4, dataset_relative.getSpace().getDimensions()[0]);

        DataSet dataset_chunked = file.getDataSet(CHUNKED_DATASET_NAME);
        BOOST_CHECK_EQUAL(4, dataset_chunked.getSpace().getDimensions()[0]);

        DataSet dataset_chunked_small =
            file.getDataSet(CHUNKED_DATASET_SMALL_NAME);
        BOOST_CHECK_EQUAL(1,
                          dataset_chunked_small.getSpace().getDimensions()[0]);
    }
}

BOOST_AUTO_TEST_CASE(HighFiveExtensibleDataSet) {
    const std::string FILE_NAME("create_extensible_dataset_example.h5");
    const std::string DATASET_NAME("dset");
    constexpr double t1[3][1] = {{2.0}, {2.0}, {4.0}};
    constexpr double t2[1][3] = {{4.0, 8.0, 6.0}};

    {
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
        dataset.select({0, 0}, {3, 1}).write(t1);

        // Resize the dataset to a larger size
        dataset.resize({4, 6});

        BOOST_CHECK_EQUAL(4, dataset.getSpace().getDimensions()[0]);
        BOOST_CHECK_EQUAL(6, dataset.getSpace().getDimensions()[1]);

        // Write into the new part of the dataset
        dataset.select({3, 3}, {1, 3}).write(t2);

        // Try resize out of bounds
        BOOST_CHECK_THROW(dataset.resize({18, 1}), DataSetException);
        // Try resize invalid dimensions
        BOOST_CHECK_THROW(dataset.resize({1, 2, 3}), DataSetException);
    }

    // read it back
    {
        File file(FILE_NAME, File::ReadOnly);

        DataSet dataset_absolute = file.getDataSet("/" + DATASET_NAME);
        const auto dims = dataset_absolute.getSpace().getDimensions();
        double values[4][6];
        dataset_absolute.read(values);
        BOOST_CHECK_EQUAL(4, dims[0]);
        BOOST_CHECK_EQUAL(6, dims[1]);

        BOOST_CHECK_EQUAL(t1[0][0], values[0][0]);
        BOOST_CHECK_EQUAL(t1[1][0], values[1][0]);
        BOOST_CHECK_EQUAL(t1[2][0], values[2][0]);

        BOOST_CHECK_EQUAL(t2[0][0], values[3][3]);
        BOOST_CHECK_EQUAL(t2[0][1], values[3][4]);
        BOOST_CHECK_EQUAL(t2[0][2], values[3][5]);
    }
}

#ifdef HIGHFIVE_CPP11_ENABLE
BOOST_AUTO_TEST_CASE(HighFiveRefCountMove) {
    const std::string FILE_NAME("h5_ref_count_test.h5");
    const std::string DATASET_NAME("dset");
    const std::string GROUP_NAME1("/group1");
    const std::string GROUP_NAME2("/group2");

    // Create a new file using the default property lists.
    File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);

    std::unique_ptr<DataSet> d1_ptr;
    std::unique_ptr<Group> g_ptr;

    {
        // create group
        Group g1 = file.createGroup(GROUP_NAME1);

        // override object
        g1 = file.createGroup(GROUP_NAME2);

        // Create the data space for the dataset.
        std::vector<size_t> dims = {10, 10};

        DataSpace dataspace(dims);

        DataSet d1 = file.createDataSet(GROUP_NAME1 + DATASET_NAME, dataspace,
                                        AtomicType<double>());

        double values[10][10] = {0};
        values[5][0] = 1;
        d1.write(values);

        // force move
        d1_ptr.reset(new DataSet(std::move(d1)));

        // force copy
        g_ptr.reset(new Group(g1));
    }
    // read it back
    {
        DataSet d2(std::move(*d1_ptr));
        d1_ptr.reset();

        double values[10][10];
        memset(values, 255, 10 * 10);

        d2.read(values);

        for (std::size_t i = 0; i < 10; ++i) {
            for (std::size_t j = 0; j < 10; ++j) {
                double v = values[i][j];

                if (i == 5 && j == 0) {
                    BOOST_CHECK_EQUAL(v, 1);
                } else {
                    BOOST_CHECK_EQUAL(v, 0);
                }
            }
        }

        // force copy
        Group g2 = *g_ptr;

        // add a subgroup
        g2.createGroup("blabla");
    }
}

#endif

BOOST_AUTO_TEST_CASE(HighFiveSimpleListing) {
    const std::string FILE_NAME("h5_list_test.h5");
    const std::string GROUP_NAME_CORE("group_name");
    const std::string GROUP_NESTED_NAME("/group_nested");

    // Create a new file using the default property lists.
    File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);

    {
        // absolute group
        for (int i = 0; i < 2; ++i) {
            std::ostringstream ss;
            ss << "/" << GROUP_NAME_CORE << "_" << i;
            file.createGroup(ss.str());
        }

        size_t n_elem = file.getNumberObjects();
        BOOST_CHECK_EQUAL(2, n_elem);

        std::vector<std::string> elems = file.listObjectNames();
        BOOST_CHECK_EQUAL(2, elems.size());
        std::vector<std::string> reference_elems;
        for (int i = 0; i < 2; ++i) {
            std::ostringstream ss;
            ss << GROUP_NAME_CORE << "_" << i;
            reference_elems.push_back(ss.str());
        }

        BOOST_CHECK_EQUAL_COLLECTIONS(elems.begin(), elems.end(),
                                      reference_elems.begin(),
                                      reference_elems.end());
    }

    {
        file.createGroup(GROUP_NESTED_NAME);
        Group g_nest = file.getGroup(GROUP_NESTED_NAME);

        for (int i = 0; i < 50; ++i) {
            std::ostringstream ss;
            ss << GROUP_NAME_CORE << "_" << i;
            g_nest.createGroup(ss.str());
        }

        size_t n_elem = g_nest.getNumberObjects();
        BOOST_CHECK_EQUAL(50, n_elem);

        std::vector<std::string> elems = g_nest.listObjectNames();
        BOOST_CHECK_EQUAL(50, elems.size());
        std::vector<std::string> reference_elems;

        for (int i = 0; i < 50; ++i) {
            std::ostringstream ss;
            ss << GROUP_NAME_CORE << "_" << i;
            reference_elems.push_back(ss.str());
        }
        // there is no guarantee on the order of the hdf5 index, let's sort it
        // to put them in order
        std::sort(elems.begin(), elems.end());
        std::sort(reference_elems.begin(), reference_elems.end());

        BOOST_CHECK_EQUAL_COLLECTIONS(elems.begin(), elems.end(),
                                      reference_elems.begin(),
                                      reference_elems.end());
    }
}

BOOST_AUTO_TEST_CASE(DataTypeEqualSimple) {
    using namespace HighFive;

    AtomicType<double> d_var;
    AtomicType<size_t> size_var;
    AtomicType<double> d_var_test;
    AtomicType<size_t> size_var_cpy(size_var);
    AtomicType<int> int_var;
    AtomicType<unsigned int> uint_var;

    // check different type matching
    BOOST_CHECK(d_var == d_var_test);
    BOOST_CHECK(d_var != size_var);

    // check type copy matching
    BOOST_CHECK(size_var_cpy == size_var);

    // check sign change not matching
    BOOST_CHECK(int_var != uint_var);
}

BOOST_AUTO_TEST_CASE(DataTypeEqualTakeBack) {
    using namespace HighFive;

    const std::string FILE_NAME("h5tutr_dset.h5");
    const std::string DATASET_NAME("dset");

    // Create a new file using the default property lists.
    File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);

    // Create the data space for the dataset.
    std::vector<size_t> dims;
    dims.push_back(10);
    dims.push_back(1);

    DataSpace dataspace(dims);

    // Create a dataset with double precision floating points
    DataSet dataset =
        file.createDataSet<size_t>(DATASET_NAME + "_double", dataspace);

    AtomicType<size_t> s;
    AtomicType<double> d;

    BOOST_CHECK(s == dataset.getDataType());
    BOOST_CHECK(d != dataset.getDataType());
}

BOOST_AUTO_TEST_CASE(DataSpaceTest) {
    using namespace HighFive;

    const std::string FILE_NAME("h5tutr_space.h5");
    const std::string DATASET_NAME("dset");

    // Create a new file using the default property lists.
    File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);

    // Create the data space for the dataset.
    std::vector<size_t> dims;
    dims.push_back(10);
    dims.push_back(1);

    DataSpace dataspace(dims);

    // Create a dataset with size_t type
    DataSet dataset = file.createDataSet<size_t>(DATASET_NAME, dataspace);

    DataSpace space = dataset.getSpace();
    DataSpace space2 = dataset.getSpace();

    // verify space id are different
    BOOST_CHECK_NE(space.getId(), space2.getId());

    // verify space id are consistent
    BOOST_CHECK_EQUAL(space.getDimensions().size(), 2);
    BOOST_CHECK_EQUAL(space.getDimensions()[0], 10);
    BOOST_CHECK_EQUAL(space.getDimensions()[1], 1);
}

template <typename T>
void readWrite2DArrayTest() {
    std::ostringstream filename;
    filename << "h5_rw_2d_array_" << typeid(T).name() << "_test.h5";
    const std::string DATASET_NAME("dset");
    const size_t x_size = 100;
    const size_t y_size = 10;

    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

    // Create the data space for the dataset.
    std::vector<size_t> dims;
    dims.push_back(x_size);
    dims.push_back(y_size);

    DataSpace dataspace(dims);

    // Create a dataset with arbitrary type
    DataSet dataset = file.createDataSet<T>(DATASET_NAME, dataspace);

    T array[x_size][y_size];

    ContentGenerate<T> generator;
    generate2D(array, x_size, y_size, generator);

    dataset.write(array);

    T result[x_size][y_size];

    dataset.read(result);

    for (size_t i = 0; i < x_size; ++i) {
        for (size_t j = 0; j < y_size; ++j) {
            BOOST_CHECK_EQUAL(result[i][j], array[i][j]);
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReadWrite2DArray, T, numerical_test_types) {
    readWrite2DArrayTest<T>();
}

template <typename T>
void readWriteVectorTest() {
    using namespace HighFive;

    std::ostringstream filename;
    filename << "h5_rw_vec_" << typeid(T).name() << "_test.h5";

    std::srand((unsigned int)std::time(0));
    const size_t x_size = 800;
    const std::string DATASET_NAME("dset");
    typename std::vector<T> vec;

    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

    // Create a dataset with double precision floating points
    DataSet dataset = file.createDataSet<T>(DATASET_NAME, DataSpace::From(vec));

    dataset.write(vec);

    typename std::vector<T> result;

    dataset.read(result);

    BOOST_CHECK_EQUAL(vec.size(), x_size);
    BOOST_CHECK_EQUAL(result.size(), x_size);

    for (size_t i = 0; i < x_size; ++i) {
        // std::cout << result[i] << " " << vec[i] << "  ";
        BOOST_CHECK_EQUAL(result[i], vec[i]);
    }
}

template <typename T>
void readWriteAttributeVectorTest() {
    using namespace HighFive;

    std::ostringstream filename;
    filename << "h5_rw_attribute_vec_" << typeid(T).name() << "_test.h5";

    std::srand((unsigned int)std::time(0));
    const size_t x_size = 25;
    const std::string DATASET_NAME("dset");
    typename std::vector<T> vec;

    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

    vec.resize(x_size);
    ContentGenerate<T> generator;

    std::generate(vec.begin(), vec.end(), generator);

    {
        // Create a dummy group to annotate with an attribute
        Group g = file.createGroup("dummy_group");

        // check that no attributes are there
        std::size_t n = g.getNumberAttributes();
        BOOST_CHECK_EQUAL(n, 0);

        std::vector<std::string> all_attribute_names = g.listAttributeNames();
        BOOST_CHECK_EQUAL(all_attribute_names.size(), 0);

        bool has_attribute = g.hasAttribute("my_attribute");
        BOOST_CHECK_EQUAL(has_attribute, false);

        Attribute a1 =
            g.createAttribute<T>("my_attribute", DataSpace::From(vec));
        a1.write(vec);

        // check now that we effectively have an attribute listable
        std::size_t n2 = g.getNumberAttributes();
        BOOST_CHECK_EQUAL(n2, 1);

        has_attribute = g.hasAttribute("my_attribute");
        BOOST_CHECK_EQUAL(has_attribute, true);

        all_attribute_names = g.listAttributeNames();
        BOOST_CHECK_EQUAL(all_attribute_names.size(), 1);
        BOOST_CHECK_EQUAL(all_attribute_names[0], std::string("my_attribute"));

        // Create the same attribute on a newly created dataset
        DataSet s =
            g.createDataSet("dummy_dataset", DataSpace(1), AtomicType<int>());

        Attribute a2 =
            s.createAttribute<T>("my_attribute_copy", DataSpace::From(vec));
        a2.write(vec);
    }

    typename std::vector<T> result1, result2;

    {
        Attribute a1_read =
            file.getGroup("dummy_group").getAttribute("my_attribute");
        a1_read.read(result1);

        BOOST_CHECK_EQUAL(vec.size(), x_size);
        BOOST_CHECK_EQUAL(result1.size(), x_size);

        for (size_t i = 0; i < x_size; ++i) {
            // std::cout << result[i] << " " << vec[i] << "  ";
            BOOST_CHECK_EQUAL(result1[i], vec[i]);
        }

        Attribute a2_read = file.getDataSet("/dummy_group/dummy_dataset")
                                .getAttribute("my_attribute_copy");
        a2_read.read(result2);

        BOOST_CHECK_EQUAL(vec.size(), x_size);
        BOOST_CHECK_EQUAL(result2.size(), x_size);

        for (size_t i = 0; i < x_size; ++i) {
            // std::cout << result[i] << " " << vec[i] << "  ";
            BOOST_CHECK_EQUAL(result2[i], vec[i]);
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReadWriteAttributeVector, T, dataset_test_types) {
    readWriteAttributeVectorTest<T>();
}

template <typename T>
void readWriteVector2DTest() {
    using namespace HighFive;

    std::ostringstream filename;
    filename << "h5_rw_vec_2d_" << typeid(T).name() << "_test.h5";

    const size_t x_size = 10;
    const size_t y_size = 10;
    const std::string DATASET_NAME("dset");
    typename std::vector<std::vector<T>> vec;

    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

    vec.resize(x_size);
    ContentGenerate<T> generator;

    generate2D(vec, x_size, y_size, generator);

    // Create a dataset with double precision floating points
    DataSet dataset = file.createDataSet<T>(DATASET_NAME, DataSpace::From(vec));

    dataset.write(vec);

    typename std::vector<std::vector<T>> result;

    dataset.read(result);

    BOOST_CHECK_EQUAL(vec.size(), x_size);
    BOOST_CHECK_EQUAL(result.size(), x_size);

    BOOST_CHECK_EQUAL(vec[0].size(), y_size);
    BOOST_CHECK_EQUAL(result[0].size(), y_size);

    for (size_t i = 0; i < x_size; ++i) {
        for (size_t j = 0; i < y_size; ++i) {
            BOOST_CHECK_EQUAL(result[i][j], vec[i][j]);
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(readWriteVector2D, T, numerical_test_types) {
    readWriteVector2DTest<T>();
}

BOOST_AUTO_TEST_CASE(datasetOffset) {
    std::string filename = "datasetOffset.h5";
    std::string dsetname = "dset";
    const size_t size_dataset = 20;

    File file(filename, File::ReadWrite | File::Create | File::Truncate);
    std::vector<int> data(size_dataset);
    DataSet ds = file.createDataSet<int>(dsetname, DataSpace::From(data));
    ds.write(data);
    DataSet ds_read = file.getDataSet(dsetname);
    BOOST_CHECK(ds_read.getOffset() > 0);
}

#ifdef H5_USE_BOOST

template <typename T>
void MultiArray3DTest() {
    typedef typename boost::multi_array<T, 3> MultiArray;

    std::ostringstream filename;
    filename << "h5_rw_multiarray_" << typeid(T).name() << "_test.h5";

    const size_t size_x = 10, size_y = 10, size_z = 10;
    const std::string DATASET_NAME("dset");
    MultiArray array(boost::extents[size_x][size_y][size_z]);

    ContentGenerate<T> generator;
    std::generate(array.data(), array.data() + array.num_elements(), generator);

    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

    DataSet dataset =
        file.createDataSet<T>(DATASET_NAME, DataSpace::From(array));

    dataset.write(array);

    // read it back
    MultiArray result;

    dataset.read(result);

    for (size_t i = 0; i < size_x; ++i) {
        for (size_t j = 0; j < size_y; ++j) {
            for (size_t k = 0; k < size_z; ++k) {
                // std::cout << array[i][j][k] << " ";
                BOOST_CHECK_EQUAL(array[i][j][k], result[i][j][k]);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MultiArray3D, T, numerical_test_types) {
    MultiArray3DTest<T>();
}

template <typename T>
void ublas_matrix_Test() {
    typedef typename boost::numeric::ublas::matrix<T> Matrix;

    std::ostringstream filename;
    filename << "h5_rw_multiarray_" << typeid(T).name() << "_test.h5";

    const size_t size_x = 10, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (std::size_t i = 0; i < mat.size1(); ++i) {
        for (std::size_t j = 0; j < mat.size2(); ++j) {
            mat(i, j) = generator();
        }
    }

    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

    DataSet dataset = file.createDataSet<T>(DATASET_NAME, DataSpace::From(mat));

    dataset.write(mat);

    // read it back
    Matrix result;

    dataset.read(result);

    for (size_t i = 0; i < size_x; ++i) {
        for (size_t j = 0; j < size_y; ++j) {
            // std::cout << array[i][j][k] << " ";
            BOOST_CHECK_EQUAL(mat(i, j), result(i, j));
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ublas_matrix, T, numerical_test_types) {
    ublas_matrix_Test<T>();
}

#endif

template <typename T>
void selectionArraySimpleTest() {
    typedef typename std::vector<T> Vector;

    std::ostringstream filename;
    filename << "h5_rw_select_test_" << typeid(T).name() << "_test.h5";

    const size_t size_x = 10;
    const size_t offset_x = 2, count_x = 5;

    const std::string DATASET_NAME("dset");

    Vector values(size_x);

    ContentGenerate<T> generator;
    std::generate(values.begin(), values.end(), generator);

    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

    DataSet dataset =
        file.createDataSet<T>(DATASET_NAME, DataSpace::From(values));

    dataset.write(values);

    file.flush();

    // select slice
    {
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

        BOOST_CHECK_EQUAL(result.size(), 5);

        for (size_t i = 0; i < count_x; ++i) {
            BOOST_CHECK_EQUAL(values[i + offset_x], result[i]);
        }
    }

    // select cherry pick
    {
        // read it back
        Vector result;
        std::vector<size_t> ids;
        ids.push_back(1);
        ids.push_back(3);
        ids.push_back(4);
        ids.push_back(7);

        Selection slice = dataset.select(ElementSet(ids));

        BOOST_CHECK_EQUAL(slice.getSpace().getDimensions()[0], size_x);
        BOOST_CHECK_EQUAL(slice.getMemSpace().getDimensions()[0], ids.size());

        slice.read(result);

        BOOST_CHECK_EQUAL(result.size(), ids.size());

        for (size_t i = 0; i < ids.size(); ++i) {
            const std::size_t id = ids[i];
            BOOST_CHECK_EQUAL(values[id], result[i]);
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(selectionArraySimple, T, dataset_test_types) {
    selectionArraySimpleTest<T>();
}

template <typename T>
void columnSelectionTest() {
    std::ostringstream filename;
    filename << "h5_rw_select_column_test_" << typeid(T).name() << "_test.h5";

    const size_t x_size = 10;
    const size_t y_size = 7;

    const std::string DATASET_NAME("dset");

    T values[x_size][y_size];

    ContentGenerate<T> generator;
    generate2D(values, x_size, y_size, generator);

    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

    // Create the data space for the dataset.
    std::vector<size_t> dims;
    dims.push_back(x_size);
    dims.push_back(y_size);

    DataSpace dataspace(dims);
    // Create a dataset with arbitrary type
    DataSet dataset = file.createDataSet<T>(DATASET_NAME, dataspace);

    dataset.write(values);

    file.flush();

    std::vector<size_t> columns;
    columns.push_back(1);
    columns.push_back(3);
    columns.push_back(5);

    Selection slice = dataset.select(columns);
    T result[x_size][3];
    slice.read(result);

    BOOST_CHECK_EQUAL(slice.getSpace().getDimensions()[0], x_size);
    BOOST_CHECK_EQUAL(slice.getMemSpace().getDimensions()[0], x_size);

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < x_size; ++j)
            BOOST_CHECK_EQUAL(result[j][i], values[j][columns[i]]);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(columnSelection, T, numerical_test_types) {
    columnSelectionTest<T>();
}

template <typename T>
void attribute_scalar_rw() {
    std::ostringstream filename;
    filename << "h5_rw_attribute_scalar_rw" << typeid(T).name() << "_test.h5";

    File h5file(filename.str(),
                File::ReadWrite | File::Create | File::Truncate);

    ContentGenerate<T> generator;

    const T attribute_value(generator());

    Group g = h5file.createGroup("metadata");

    bool family_exist = g.hasAttribute("family");
    BOOST_CHECK_EQUAL(family_exist, false);

    // write a scalar attribute
    {
        T out(attribute_value);
        Attribute att = g.createAttribute<T>("family", DataSpace::From(out));
        att.write(out);
    }

    h5file.flush();

    // test if attribute exist
    family_exist = g.hasAttribute("family");
    BOOST_CHECK_EQUAL(family_exist, true);

    // read back a scalar attribute
    {
        T res;
        Attribute att = g.getAttribute("family");
        att.read(res);
        BOOST_CHECK_EQUAL(res, attribute_value);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(attribute_scalar_rw_all, T, dataset_test_types) {
    attribute_scalar_rw<T>();
}

// regression test https://github.com/BlueBrain/HighFive/issues/98
BOOST_AUTO_TEST_CASE(HighFiveOutofDimension) {
    std::string filename("h5_rw_reg_zero_dim_test.h5");

    const std::string DATASET_NAME("dset");

    {
        // Create a new file using the default property lists.
        File file(filename, File::ReadWrite | File::Create | File::Truncate);

        DataSpace d_null(DataSpace::DataspaceType::datascape_null);

        DataSet d1 = file.createDataSet<double>(DATASET_NAME, d_null);

        file.flush();

        DataSpace recovered_d1 = d1.getSpace();

        auto ndim = recovered_d1.getNumberDimensions();
        BOOST_CHECK_EQUAL(ndim, 0);

        auto dims = recovered_d1.getDimensions();
        BOOST_CHECK_EQUAL(dims.size(), 0);
    }
}

template <typename T>
void readWriteShuffleDeflateTest() {
    std::ostringstream filename;
    filename << "h5_rw_deflate_" << typeid(T).name() << "_test.h5";
    const std::string DATASET_NAME("dset");
    const size_t x_size = 128;
    const size_t y_size = 32;
    const size_t x_chunk = 16;
    const size_t y_chunk = 16;

    const int deflate_level = 9;

    T array[x_size][y_size];

    // write a compressed file
    {
        File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

        // Create the data space for the dataset.
        std::vector<size_t> dims;
        dims.push_back(x_size);
        dims.push_back(y_size);

        DataSpace dataspace(dims);

        // Use chunking
        DataSetCreateProps props;
        props.add(Chunking(std::vector<hsize_t>{x_chunk, y_chunk}));

        // Enable shuffle
        props.add(Shuffle());

        // Enable deflate
        props.add(Deflate(deflate_level));

        // Create a dataset with arbitrary type
        DataSet dataset = file.createDataSet<T>(DATASET_NAME, dataspace, props);

        ContentGenerate<T> generator;
        generate2D(array, x_size, y_size, generator);

        dataset.write(array);

        file.flush();
    }

    // read it back
    {
        File file_read(filename.str(), File::ReadOnly);
        DataSet dataset_read = file_read.getDataSet("/" + DATASET_NAME);

        T result[x_size][y_size];

        dataset_read.read(result);

        for (size_t i = 0; i < x_size; ++i) {
            for (size_t j = 0; i < y_size; ++i) {
                BOOST_CHECK_EQUAL(result[i][j], array[i][j]);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ReadWriteShuffleDeflate, T, numerical_test_types) {
    readWriteShuffleDeflateTest<T>();
}
