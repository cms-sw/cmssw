#ifndef LibraryLoader_RootAutoLibraryLoader_h
#define LibraryLoader_RootAutoLibraryLoader_h
/**\class RootAutoLibraryLoader
 *
 * ROOT helper class which can automatically load the 
 * proper shared library when ROOT needs a new class dictionary
 *
 * \author Chris Jones, Cornell
 *
 * $Id: RootAutoLibraryLoader.h,v 1.2 2010/12/15 21:22:25 chrjones Exp $
 *
 */
#include "TClassGenerator.h"
#include "RVersion.h"

class DummyClassToStopCompilerWarning;

namespace edm {
class RootAutoLibraryLoader : public TClassGenerator {
  friend class DummyClassToStopCompilerWarning;
public:
  /// return class type
  virtual TClass *GetClass(const char* classname, Bool_t load);
  /// return class type
  virtual TClass *GetClass(const type_info& typeinfo, Bool_t load);
  /// interface for TClass generators
  //ClassDef(RootAutoLibraryLoader,1);
  /// enable automatic library loading  
  static void enable();
  
  /// load all known libraries holding dictionaries
  static void loadAll();

private:
  const char* classNameAttemptingToLoad_; //!
  RootAutoLibraryLoader();
  RootAutoLibraryLoader(const RootAutoLibraryLoader&); // stop default
  const RootAutoLibraryLoader& operator=(const RootAutoLibraryLoader&); // stop default
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,6)
  bool isInitializingCintex_;
#endif
};

}
#endif
