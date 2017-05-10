#ifndef hcal_ConfigurationDatabaseImpl_hh_included
#define hcal_ConfigurationDatabaseImpl_hh_included 1

#include <string>
#include <vector>
#include <map>
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseException.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/PluginManager.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabase.hh"

#ifdef HAVE_XDAQ
#include "log4cplus/logger.h"
#else
#include "CaloOnlineTools/HcalOnlineDb/interface/xdaq_compat.h"  // Includes typedef for log4cplus::Logger
#endif

//OCCI include
#include "OnlineDB/Oracle/interface/Oracle.h"

namespace hcal {

  /** \brief Implementation of an accessor to the configuration database.

   Accessors look like: 
     method://[user@]host[:port]/[database]?KEY=VALUE,...
     \ingroup hcalBase
  */
  class ConfigurationDatabaseImpl : public hcal::Pluggable {
  public:
    ConfigurationDatabaseImpl();
    /** \brief Set logger */
    void setLogger(log4cplus::Logger logger) { m_logger=logger; }
    /** \brief Static method to parse an accessor into various fields */
    static void parseAccessor(const std::string& accessor, std::string& method, std::string& host, std::string& port, std::string& user, std::string& db, std::map<std::string,std::string>& params);
    /** \brief Used by the Application to determine which implementation to use for a given accessor */
    virtual bool canHandleMethod(const std::string& method) const = 0;   
    /** \brief Connect to the database using the given accessor */
    virtual void connect(const std::string& accessor) throw (hcal::exception::ConfigurationDatabaseException) = 0;
    /** \brief Disconnect from the database */
    virtual void disconnect() = 0;

    /* Various requests (default for all is to throw an exception indicating that no implementation is available. */
    virtual std::vector<std::string> getValidTags() throw (hcal::exception::ConfigurationDatabaseException);
    virtual ConfigurationDatabase::ApplicationConfig getApplicationConfig(const std::string& tag, const std::string& classname, int instance) throw (hcal::exception::ConfigurationDatabaseException);
    virtual std::string getConfigurationDocument(const std::string& tag) throw (hcal::exception::ConfigurationDatabaseException);
    /** \brief Retrieve the checksum for a given firmware version */
    virtual unsigned int getFirmwareChecksum(const std::string& board, unsigned int version) throw (hcal::exception::ConfigurationDatabaseException);
    /** \brief Retrieve the MCS file lines for a given firmware version */
    virtual void getFirmwareMCS(const std::string& board, unsigned int version, std::vector<std::string>& mcsLines) throw (hcal::exception::ConfigurationDatabaseException);
    virtual void getLUTs(const std::string& tag, int crate, int slot, std::map<ConfigurationDatabase::LUTId, ConfigurationDatabase::LUT >& LUTs) throw (hcal::exception::ConfigurationDatabaseException);
    virtual void getLUTChecksums(const std::string& tag, std::map<ConfigurationDatabase::LUTId, ConfigurationDatabase::MD5Fingerprint>& checksums) throw (hcal::exception::ConfigurationDatabaseException);
    virtual void getPatterns(const std::string& tag, int crate, int slot, std::map<ConfigurationDatabase::PatternId, ConfigurationDatabase::HTRPattern>& patterns) throw (hcal::exception::ConfigurationDatabaseException);
    virtual void getZSThresholds(const std::string& tag, int crate, int slot, std::map<ConfigurationDatabase::ZSChannelId, int>& thresholds) throw (hcal::exception::ConfigurationDatabaseException);
    virtual void getHLXMasks(const std::string& tag, int crate, int slot, std::map<ConfigurationDatabase::FPGAId, ConfigurationDatabase::HLXMasks>& masks) throw (hcal::exception::ConfigurationDatabaseException);
    virtual void getRBXdata(const std::string& tag, const std::string& rbx, ConfigurationDatabase::RBXdatumType dtype, std::map<ConfigurationDatabase::RBXdatumId, ConfigurationDatabase::RBXdatum>& RBXdata) throw (hcal::exception::ConfigurationDatabaseException);
    virtual void getRBXpatterns(const std::string& tag, const std::string& rbx,	std::map<ConfigurationDatabase::RBXdatumId, ConfigurationDatabase::RBXpattern>& patterns) throw (hcal::exception::ConfigurationDatabaseException);

    // added by Gena Kukartsev
    virtual oracle::occi::Connection * getConnection( void );
    virtual oracle::occi::Environment * getEnvironment( void );

  protected:
    log4cplus::Logger m_logger;
  };

}

#endif // hcal_ConfigurationDatabaseImpl_hh_included
