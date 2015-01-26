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

#include <string>
#include <typeinfo>

class DummyClassToStopCompilerWarning;

namespace edm {

class RootAutoLibraryLoader : public TClassGenerator {
  friend class DummyClassToStopCompilerWarning;
private: // Private Data Members
  std::string classNameAttemptingToLoad_;
private: // Private Function Members
  RootAutoLibraryLoader();
  RootAutoLibraryLoader(RootAutoLibraryLoader const&); // NOT IMPLEMENTED
  RootAutoLibraryLoader const& operator=(RootAutoLibraryLoader const&); // NOT IMPLEMENTED
public: // Public Static Function Members
  /// enable automatic library loading
  static void enable();
  /// load all known libraries holding dictionaries
  static void loadAll();
public: // Public Function Members
  /// return class type
  virtual TClass* GetClass(const char* classname, Bool_t load);
  /// return class type
  virtual TClass* GetClass(const type_info& typeinfo, Bool_t load);
};

} // namespace edm

#endif // LibraryLoader_RootAutoLibraryLoader_h
