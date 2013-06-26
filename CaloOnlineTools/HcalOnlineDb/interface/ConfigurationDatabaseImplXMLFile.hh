#ifndef ConfigurationDatabaseImplXerces_hh_included
#define ConfigurationDatabaseImplXerces_hh_included 1

#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImpl.hh"
#include <map>
#include <string>
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseStandardXMLParser.hh"

//OCCI include
#include "OnlineDB/Oracle/interface/Oracle.h"

class ConfigurationDatabaseImplXMLFile : public hcal::ConfigurationDatabaseImpl {
public:
  ConfigurationDatabaseImplXMLFile();
  virtual ~ConfigurationDatabaseImplXMLFile();
  virtual bool canHandleMethod(const std::string& method) const;
  virtual void connect(const std::string& accessor) throw (hcal::exception::ConfigurationDatabaseException);
  virtual void disconnect();

  virtual unsigned int getFirmwareChecksum(const std::string& board, unsigned int version) throw (hcal::exception::ConfigurationDatabaseException);
  virtual void getZSThresholds(const std::string& tag, int crate, int slot, std::map<hcal::ConfigurationDatabase::ZSChannelId, int>& thresholds) throw (hcal::exception::ConfigurationDatabaseException);
  virtual void getFirmwareMCS(const std::string& board, unsigned int version, std::vector<std::string>& mcsLines) throw (hcal::exception::ConfigurationDatabaseException);
  // maximally simple implementation
  virtual void getLUTs(const std::string& tag, int crate, int slot, std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::LUT >& LUTs) throw (hcal::exception::ConfigurationDatabaseException);
  virtual void getLUTChecksums(const std::string& tag, std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::MD5Fingerprint>& checksums) throw (hcal::exception::ConfigurationDatabaseException);
  virtual void getPatterns(const std::string& tag, int crate, int slot, std::map<hcal::ConfigurationDatabase::PatternId, hcal::ConfigurationDatabase::HTRPattern >& patterns) throw (hcal::exception::ConfigurationDatabaseException);

  // added by Gena Kukartsev
  //virtual oracle::occi::Connection * getConnection( void );
  //virtual oracle::occi::Environment * getEnvironment( void );

private:
  std::map<std::string, std::string> extractParams(int beg, int end);
  std::map<std::string, std::string> parseWhere(const std::string& where);
  std::string createKey(const std::map<std::string,std::string>& params);
  std::map<std::string, std::pair<int,int> > m_lookup;
  std::string m_buffer;
  ConfigurationDatabaseStandardXMLParser m_parser;
};

#endif // ConfigurationDatabaseImplXMLFile_hh_included
