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

/* 
 * configuration parameters of TFileAdaptor
 */
struct TFileAdaptorParams {
  bool doStats;
  bool doBuffering;
  bool doCaching;
  std::string mode;
  int  cacheSize;
  int  cachePageSize;
  std::vector<std::string> m_native;

  void init() const;
  bool native(const char * prot) const;
private:
  void pinit();
  
};


/*  
 * driver for configuring root plugin manager to use TStorageFactoryFile 
 */
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

  // Write current Storage statistics on a ostream
  void stats(std::ostream& co) const;


};


#endif // TFileAdaptor_H
