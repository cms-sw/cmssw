#ifndef TFileAdaptor_H
#define TFileAdaptor_H
#include <vector>
#include <string>
#include <iosfwd>

namespace edm {
  class ParameterSet;
  class ActivityRegistry;

}

class TPluginManager;

struct TFileAdaptorParams {
  bool doStats;
  bool doBuffering;
  bool doCashing;
  std::string mode;
  int  cacheSize;
  int  cachePageSize;
  std::vector<std::string> m_native;

  void init() const;
  bool native(const char * prot) const;
private:
  void pinit();
  
};

class TFileAdaptor {
private:
  TFileAdaptorParams const m_params;
private:
  static void addFileType (TPluginManager *mgr, const char *type);

  static void addSystemType (TPluginManager *mgr, const char *type);

  bool native(const char * prot) const;

public:

  TFileAdaptor (const TFileAdaptorParams& iparams);
  ~TFileAdaptor ();

  void stats(std::ostream& co) const;


};


#endif // TFileAdaptor_H
