#ifndef Fireworks_Core_DataPath_h
#define Fireworks_Core_DataPath_h

class TString;

namespace fireworks {
  void setPath(TString& v);
  void getDecomposedVersion(const TString& s, int* out);
  int* supportedDataFormatsVersion();
  bool acceptDataFormatsVersion(TString& n);
}  // namespace fireworks

#endif
