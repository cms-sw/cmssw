#ifndef GENERS_CPP11_ARRAY_HH_
#define GENERS_CPP11_ARRAY_HH_

#include "Alignment/Geners/interface/CPP11_config.hh"

#ifdef CPP11_STD_AVAILABLE
#include <array>
#define CPP11_array std::array
#else
#include <tr1/array>
#define CPP11_array std::tr1::array
#endif

#endif // GENERS_CPP11_ARRAY_HH_

