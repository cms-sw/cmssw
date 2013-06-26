#ifndef _ConfigurationDatabaseImplOracle_hh_included
#define _ConfigurationDatabaseImplOracle_hh_included 1

#include "CaloOnlineTools/HcalOnlineDb/interface/PluginManager.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImpl.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseStandardXMLParser.hh"

#include "xgi/Method.h"
#include "xdata/xdata.h"

//OCCI include
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "ConfigurationDatabaseStandardXMLParser.hh"

class ConfigurationDatabaseImplOracle: public hcal::ConfigurationDatabaseImpl {

	public:

		ConfigurationDatabaseImplOracle();
		virtual ~ConfigurationDatabaseImplOracle();
		virtual bool canHandleMethod(const std::string& method) const;
		virtual void connect(const std::string& accessor) throw (hcal::exception::ConfigurationDatabaseException);
		virtual void disconnect();
	
		virtual void getLUTs(const std::string& tag, int crate, int slot,  
			std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::LUT >& LUTs) 
							throw (hcal::exception::ConfigurationDatabaseException);

		virtual void getLUTChecksums(const std::string& tag, 
			std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::MD5Fingerprint>& checksums) 
							throw (hcal::exception::ConfigurationDatabaseException);

		virtual void getPatterns(const std::string& tag, int crate, int slot, 
			std::map<hcal::ConfigurationDatabase::PatternId, hcal::ConfigurationDatabase::HTRPattern >& patterns) 
							throw (hcal::exception::ConfigurationDatabaseException);

		virtual void getRBXdata(const std::string& tag, const std::string& rbx,
                                        hcal::ConfigurationDatabase::RBXdatumType dtype,
                                        std::map<hcal::ConfigurationDatabase::RBXdatumId, hcal::ConfigurationDatabase::RBXdatum>& RBXdata)
                                                        throw (hcal::exception::ConfigurationDatabaseException);

	        virtual void getZSThresholds(const std::string& tag, int crate, int slot,
                                        std::map<hcal::ConfigurationDatabase::ZSChannelId, int>& thresholds)
                                        throw (hcal::exception::ConfigurationDatabaseException);

	        virtual void getHLXMasks(const std::string& tag, int crate, int slot,
                                        std::map<hcal::ConfigurationDatabase::FPGAId,
                                                        hcal::ConfigurationDatabase::HLXMasks>& masks)
                                        throw (hcal::exception::ConfigurationDatabaseException);

  // added by Gena Kukartsev
  virtual oracle::occi::Connection * getConnection( void );
  virtual oracle::occi::Environment * getEnvironment( void );

	private:
		//OCCI Env, Conn     
		oracle::occi::Environment *env_;
        	oracle::occi::Connection *conn_;

  //oracle::occi::Connection* getConnection() throw (xgi::exception::Exception);

		ConfigurationDatabaseStandardXMLParser m_parser;
		
		xdata::String username_;
		xdata::String password_;
		xdata::String database_;

		//Used by getZSThresholds
		std::string lhwm_version;

		//Utility methods
		std::string clobToString(const oracle::occi::Clob&);
		std::string getParameter(cgicc::Cgicc &cgi,const std::string &name);

  		void getLUTs_real(const std::string& tag, int crate, std::map<hcal::ConfigurationDatabase::LUTId, 
				hcal::ConfigurationDatabase::LUT >& LUTs) throw (hcal::exception::ConfigurationDatabaseException);
  		void getPatterns_real(const std::string& tag, int crate, std::map<hcal::ConfigurationDatabase::PatternId, 
				hcal::ConfigurationDatabase::HTRPattern >& patterns) throw (hcal::exception::ConfigurationDatabaseException);
                void getHLXMasks_real(const std::string& tag, int crate,
                                std::map<hcal::ConfigurationDatabase::FPGAId, hcal::ConfigurationDatabase::HLXMasks>& masks)
                                                throw (hcal::exception::ConfigurationDatabaseException);

		struct LUTCache {
		    void clear() {
		      luts.clear();
		      crate=-1;
		      tag.clear();
		    }
		    std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::LUT > luts;
		    int crate;
		    std::string tag;
		} m_lutCache;

		struct PatternCache {
		    void clear() {
		      patterns.clear();
		      crate=-1;
		      tag.clear();
		    }
		    std::map<hcal::ConfigurationDatabase::PatternId, hcal::ConfigurationDatabase::HTRPattern > patterns;
		    int crate;
		    std::string tag;
		} m_patternCache;

                struct HLXMaskCache {
                  void clear() {
                    masks.clear();
                    crate=-1;
                    tag.clear();
                  }
                  std::map<hcal::ConfigurationDatabase::FPGAId, hcal::ConfigurationDatabase::HLXMasks> masks;
                  int crate;
                  std::string tag;
                } m_hlxMaskCache;


};

#endif
