#ifndef GENERS_CPP11_AUTO_PTR_HH_
#define GENERS_CPP11_AUTO_PTR_HH_

#include "Alignment/Geners/interface/CPP11_config.hh"

#include <memory>

#ifdef CPP11_STD_AVAILABLE
#define CPP11_auto_ptr std::unique_ptr
#else
#define CPP11_auto_ptr std::auto_ptr
#endif

#endif // GENERS_CPP11_AUTO_PTR_HH_

