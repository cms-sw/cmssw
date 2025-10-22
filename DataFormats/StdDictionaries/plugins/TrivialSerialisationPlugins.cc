#include "TrivialSerialisation/Common/interface/SerialiserFactory.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN(int);

DEFINE_TRIVIAL_SERIALISER_PLUGIN(unsigned short);

using basic_string = std::basic_string<char, std::char_traits<char>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(basic_string);
