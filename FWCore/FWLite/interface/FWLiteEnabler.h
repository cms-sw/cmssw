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
  /// enable automatic library loading  
  static void enable();

private:
  FWLiteEnabler(const FWLiteEnabler&); // stop default
  FWLiteEnabler const& operator=(FWLiteEnabler const&); // stop default
  static bool enabled_;
  FWLiteEnabler();
};


#endif
