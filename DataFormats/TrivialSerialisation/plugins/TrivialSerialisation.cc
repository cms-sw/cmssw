#include <string>
#include <vector>

// Including MemoryCopyTraits.h is not necessary, but it is included explicitly
// because it contains the specialisations used below.
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"

// Arithmetic types:
DEFINE_TRIVIAL_SERIALISER_PLUGIN(bool);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(char);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(signed char);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(unsigned char);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(short);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(unsigned short);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(int);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(unsigned int);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(long);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(unsigned long);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(long long);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(unsigned long long);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(float);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(double);

// std::vector of arithmetic types, except for std::vector<bool>:
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::vector<char>);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::vector<signed char>);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::vector<unsigned char>);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::vector<short>);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::vector<unsigned short>);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::vector<int>);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::vector<unsigned int>);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::vector<long>);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::vector<unsigned long>);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::vector<long long>);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::vector<unsigned long long>);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::vector<float>);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::vector<double>);

// std::string:
DEFINE_TRIVIAL_SERIALISER_PLUGIN(std::string);
