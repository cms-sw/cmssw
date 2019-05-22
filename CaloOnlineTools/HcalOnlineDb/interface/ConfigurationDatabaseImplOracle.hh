#ifndef _ConfigurationDatabaseImplOracle_hh_included
#define _ConfigurationDatabaseImplOracle_hh_included 1

#include "CaloOnlineTools/HcalOnlineDb/interface/PluginManager.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImpl.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseStandardXMLParser.hh"

#ifdef HAVE_XDAQ
#include "xgi/Method.h"
#include "xdata/xdata.h"
#else
#include <string>
#endif

//OCCI include
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "ConfigurationDatabaseStandardXMLParser.hh"

class ConfigurationDatabaseImplOracle : public hcal::ConfigurationDatabaseImpl {
public:
  ConfigurationDatabaseImplOracle();
  ~ConfigurationDatabaseImplOracle() override;
  bool canHandleMethod(const std::string& method) const override;
  void connect(const std::string& accessor) noexcept(false) override;
  void disconnect() override;

  void getLUTs(
      const std::string& tag,
      int crate,
      int slot,
      std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::LUT>& LUTs) noexcept(false) override;

  void getLUTChecksums(const std::string& tag,
                       std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::MD5Fingerprint>&
                           checksums) noexcept(false) override;

  void getPatterns(const std::string& tag,
                   int crate,
                   int slot,
                   std::map<hcal::ConfigurationDatabase::PatternId, hcal::ConfigurationDatabase::HTRPattern>&
                       patterns) noexcept(false) override;

  void getRBXdata(const std::string& tag,
                  const std::string& rbx,
                  hcal::ConfigurationDatabase::RBXdatumType dtype,
                  std::map<hcal::ConfigurationDatabase::RBXdatumId, hcal::ConfigurationDatabase::RBXdatum>&
                      RBXdata) noexcept(false) override;

  void getZSThresholds(const std::string& tag,
                       int crate,
                       int slot,
                       std::map<hcal::ConfigurationDatabase::ZSChannelId, int>& thresholds) noexcept(false) override;

  void getHLXMasks(const std::string& tag,
                   int crate,
                   int slot,
                   std::map<hcal::ConfigurationDatabase::FPGAId, hcal::ConfigurationDatabase::HLXMasks>&
                       masks) noexcept(false) override;

  // added by Gena Kukartsev
  oracle::occi::Connection* getConnection(void) override;
  oracle::occi::Environment* getEnvironment(void) override;

private:
  //OCCI Env, Conn
  oracle::occi::Environment* env_;
  oracle::occi::Connection* conn_;

  //oracle::occi::Connection* getConnection() throw (xgi::exception::Exception);

  ConfigurationDatabaseStandardXMLParser m_parser;

#ifdef HAVE_XDAQ
  xdata::String username_;
  xdata::String password_;
  xdata::String database_;
#else
  std::string username_;
  std::string password_;
  std::string database_;
#endif

  //Used by getZSThresholds
  std::string lhwm_version;

  //Utility methods
  std::string clobToString(const oracle::occi::Clob&);

#ifdef HAVE_XDAQ
  std::string getParameter(cgicc::Cgicc& cgi, const std::string& name);
#endif

  void getLUTs_real(
      const std::string& tag,
      int crate,
      std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::LUT>& LUTs) noexcept(false);
  void getPatterns_real(const std::string& tag,
                        int crate,
                        std::map<hcal::ConfigurationDatabase::PatternId, hcal::ConfigurationDatabase::HTRPattern>&
                            patterns) noexcept(false);
  void getHLXMasks_real(
      const std::string& tag,
      int crate,
      std::map<hcal::ConfigurationDatabase::FPGAId, hcal::ConfigurationDatabase::HLXMasks>& masks) noexcept(false);

  struct LUTCache {
    void clear() {
      luts.clear();
      crate = -1;
      tag.clear();
    }
    std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::LUT> luts;
    int crate;
    std::string tag;
  } m_lutCache;

  struct PatternCache {
    void clear() {
      patterns.clear();
      crate = -1;
      tag.clear();
    }
    std::map<hcal::ConfigurationDatabase::PatternId, hcal::ConfigurationDatabase::HTRPattern> patterns;
    int crate;
    std::string tag;
  } m_patternCache;

  struct HLXMaskCache {
    void clear() {
      masks.clear();
      crate = -1;
      tag.clear();
    }
    std::map<hcal::ConfigurationDatabase::FPGAId, hcal::ConfigurationDatabase::HLXMasks> masks;
    int crate;
    std::string tag;
  } m_hlxMaskCache;
};

#endif
