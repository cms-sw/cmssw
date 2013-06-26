#ifndef GENERS_CPP11_TYPE_TRAITS_HH_
#define GENERS_CPP11_TYPE_TRAITS_HH_

#include "Alignment/Geners/interface/CPP11_config.hh"

#ifdef CPP11_STD_AVAILABLE
#include <type_traits>
#define CPP11_is_pod std::is_pod
#define CPP11_is_pointer std::is_pointer
#else
#include <tr1/type_traits>
#define CPP11_is_pod std::tr1::is_pod
#define CPP11_is_pointer std::tr1::is_pointer
#endif

#endif // GENERS_CPP11_TYPE_TRAITS_HH_

