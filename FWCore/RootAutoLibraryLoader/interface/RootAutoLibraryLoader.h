#ifndef LibraryLoader_RootAutoLibraryLoader_h
#define LibraryLoader_RootAutoLibraryLoader_h
/**\class RootAutoLibraryLoader
 *
 * ROOT helper class which can automatically load the
 * proper shared library when ROOT needs a new class dictionary
 *
 * \author Chris Jones, Cornell
 *
 */
#include "TClassGenerator.h"
#include "RVersion.h"

#include <typeinfo>

class DummyClassToStopCompilerWarning;

namespace edm {
class RootAutoLibraryLoader : public TClassGenerator {
  friend class DummyClassToStopCompilerWarning;
public:
  /// return class type
  virtual TClass *GetClass(char const* classname, Bool_t load);
  /// return class type
  virtual TClass *GetClass(type_info const& typeinfo, Bool_t load);
  /// interface for TClass generators
  //ClassDef(RootAutoLibraryLoader,1);
  /// enable automatic library loading
  static void enable();

  /// load all known libraries holding dictionaries
  static void loadAll();

private:
  char const* classNameAttemptingToLoad_; //!
  RootAutoLibraryLoader();
  RootAutoLibraryLoader(RootAutoLibraryLoader const&); // stop default
  RootAutoLibraryLoader const& operator=(RootAutoLibraryLoader const&); // stop default
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,6)
  bool isInitializingCintex_;
#endif
};

}
#endif
