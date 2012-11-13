#ifndef GENERS_CPP11_SHARED_PTR_HH_
#define GENERS_CPP11_SHARED_PTR_HH_

#include "Alignment/Geners/interface/CPP11_config.hh"

#ifdef CPP11_STD_AVAILABLE
#include <memory>
#define CPP11_shared_ptr std::shared_ptr
#else
#include <tr1/memory>
#define CPP11_shared_ptr std::tr1::shared_ptr
#endif

#endif // GENERS_CPP11_SHARED_PTR_HH_

