#ifndef FWCore_FWLite_FWLiteEnabler_h
#define FWCore_FWLite_FWLiteEnabler_h
/**\class FWLiteEnabler
 *
 * helper class to enable fwlite.
 * Using a free function enable() directly does not work in macros.
 *
 */
class DummyClassToStopCompilerWarning;

class FWLiteEnabler {
  friend class DummyClassToStopCompilerWarning;
public:
  FWLiteEnabler(const FWLiteEnabler&) = delete; // stop default
  FWLiteEnabler const& operator=(FWLiteEnabler const&) = delete; // stop default
  /// enable automatic library loading  
  static void enable();

private:
  static bool enabled_;
  FWLiteEnabler();
};


#endif
