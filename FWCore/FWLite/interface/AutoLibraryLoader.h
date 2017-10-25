#ifndef FWCore_FWLite_AutoLibraryLoader_h
#define FWCore_FWLite_AutoLibraryLoader_h
/**\class AutoLibraryLoader
 *
 * ROOT helper class which can automatically load the 
 * proper shared library when ROOT needs a new class dictionary
 *
 * \author Chris Jones, Cornell
 *
 *
 */
class DummyClassToStopCompilerWarning;

class AutoLibraryLoader {
  friend class DummyClassToStopCompilerWarning;
public:
  /// enable automatic library loading  
  static void enable();
  
  /// load all known libraries holding dictionaries
  static void loadAll();

private:
  static bool enabled_;
  AutoLibraryLoader();
  AutoLibraryLoader(const AutoLibraryLoader&) = delete; // stop default
  const AutoLibraryLoader& operator=(const AutoLibraryLoader&) = delete; // stop default
};


#endif
