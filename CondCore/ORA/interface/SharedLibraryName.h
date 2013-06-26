#ifndef INCLUDE_ORA_SHAREDLIBRARYNAME_H
#define INCLUDE_ORA_SHAREDLIBRARYNAME_H
#include <string>
#include <boost/filesystem/path.hpp>

namespace ora {
  /** Utility functor.
      Return a shared library name that follows the system conventions
      for naming shared library.  @a name is the basic name of the
      shared library, without the name prefix ("lib" on unix) or the
      extension (".so", ".sl", ".dylib" or ".dll").  @a name must not
    have any directory components. */
  struct SharedLibraryName{
    const boost::filesystem::path operator () (const std::string &name){
#ifdef _WIN32
      return boost::filesystem::path(name + ".dll");
#elif defined __DARWIN
      return boost::filesystem::path("lib" + name + ".dylib");
#elif defined __hpux
      return boost::filesystem::path("lib" + name + ".sl");
#else
      return boost::filesystem::path("lib" + name + ".so");
#endif
    }
  };
}
#endif
