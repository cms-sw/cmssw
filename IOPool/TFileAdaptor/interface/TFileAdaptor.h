#ifndef TFILE_ADAPTOR_TFILE_ADAPTOR_H
# define TFILE_ADAPTOR_TFILE_ADAPTOR_H
# include <vector>
# include <string>
# include <iosfwd>

namespace edm
{
  class ParameterSet;
  class ActivityRegistry;
}

class TPluginManager;

// Configuration parameters for TFileAdaptor
struct TFileAdaptorParams
{
  bool doStats;
  bool doBuffering;
  bool doCaching;
  std::string mode;
  int  cacheSize;
  int  cachePageSize;
  std::vector<std::string> m_native;

  void init (void) const;
  bool native (const char *proto) const;
private:
  void pinit (void);
};

// Driver for configuring ROOT plug-in manager to use TStorageFactoryFile.
class TFileAdaptor
{
  const TFileAdaptorParams m_params;
  static void addType (TPluginManager *mgr, const char *type);
  bool native (const char *proto) const;

public:
  TFileAdaptor (const TFileAdaptorParams &iparams);
  ~TFileAdaptor ();

  // Write current Storage statistics on a ostream
  void stats (std::ostream &o) const;
  void statsXML (std::ostream &o) const;
};

#endif // TFILE_ADAPTOR_TFILE_ADAPTOR_H
