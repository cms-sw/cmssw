#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImpl.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabase.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationItemNotFoundException.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/PluginManager.hh"
#include <ctype.h>

#ifdef HAVE_XDAQ
#include <toolbox/string.h>
#else
#include "CaloOnlineTools/HcalOnlineDb/interface/xdaq_compat.h"  // Includes typedef for log4cplus::Logger
#endif

namespace hcal {

  ConfigurationDatabase::ConfigurationDatabase(log4cplus::Logger logger) : m_logger(logger) {
    m_implementation=0;
  }

  void ConfigurationDatabase::open(const std::string& accessor) throw (hcal::exception::ConfigurationDatabaseException) {
    if (m_implementationOptions.empty()) {
      std::vector<hcal::AbstractPluginFactory*> facts;
      hcal::PluginManager::getFactories("hcal::ConfigurationDatabaseImpl",facts);
      for (std::vector<hcal::AbstractPluginFactory*>::iterator j=facts.begin(); j!=facts.end(); j++)
        m_implementationOptions.push_back(dynamic_cast<hcal::ConfigurationDatabaseImpl*>((*j)->newInstance()));
    }

    std::map<std::string,std::string> params;
    std::string user, host, method, db, port,password;
    ConfigurationDatabaseImpl::parseAccessor(accessor,method,host,port,user,db,params);

    if (m_implementation==0 || !m_implementation->canHandleMethod(method)) {
      m_implementation=0;
      std::vector<ConfigurationDatabaseImpl*>::iterator j;
      for (j=m_implementationOptions.begin(); j!=m_implementationOptions.end(); j++)
        if ((*j)->canHandleMethod(method)) {
          m_implementation=*j;
          break;
        }
    }

    if (m_implementation==0)
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,toolbox::toString("Unable to open database using '%s'",accessor.c_str()));
    m_implementation->setLogger(m_logger);
    m_implementation->connect(accessor);

  }

  void ConfigurationDatabase::close() {
    if (m_implementation!=0) m_implementation->disconnect();
  }

  unsigned int ConfigurationDatabase::getFirmwareChecksum(const std::string& board, unsigned int version) throw (hcal::exception::ConfigurationDatabaseException) {
    if (m_implementation==0) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Database connection not open");
    }

    return m_implementation->getFirmwareChecksum(board,version);
  }

  ConfigurationDatabase::ApplicationConfig ConfigurationDatabase::getApplicationConfig(const std::string& tag, const std::string& classname, int instance) throw (hcal::exception::ConfigurationDatabaseException) {
    if (m_implementation==0) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Database connection not open");
    }
    return m_implementation->getApplicationConfig(tag,classname,instance);

  }


  std::string ConfigurationDatabase::getConfigurationDocument(const std::string& tag) throw (hcal::exception::ConfigurationDatabaseException) {
    if (m_implementation==0) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Database connection not open");
    }
    return m_implementation->getConfigurationDocument(tag);
  }

  void ConfigurationDatabase::getFirmwareMCS(const std::string& board, unsigned int version, std::vector<std::string>& mcsLines) throw (hcal::exception::ConfigurationDatabaseException) {
    if (m_implementation==0) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Database connection not open");
    }

    m_implementation->getFirmwareMCS(board, version, mcsLines);

  }

  void ConfigurationDatabase::getLUTs(const std::string& tag, int crate, int slot, std::map<LUTId, LUT >& LUTs) throw (hcal::exception::ConfigurationDatabaseException) {

    if (m_implementation==0) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Database connection not open");
    }

    std::map<unsigned int, std::string> results;

    m_implementation->getLUTs(tag, crate, slot, LUTs);

    if (LUTs.size()==0) {
      XCEPT_RAISE(hcal::exception::ConfigurationItemNotFoundException,toolbox::toString("Not enough found (%d)",LUTs.size()));
    }
  }

  void ConfigurationDatabase::getLUTChecksums(const std::string& tag, std::map<LUTId, MD5Fingerprint>& checksums) throw (hcal::exception::ConfigurationDatabaseException) {
    checksums.clear();

    if (m_implementation==0) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Database connection not open");
    }

    m_implementation->getLUTChecksums(tag, checksums);
  }

  void ConfigurationDatabase::getPatterns(const std::string& tag, int crate, int slot, std::map<PatternId, HTRPattern>& patterns) throw (hcal::exception::ConfigurationDatabaseException) {

    if (m_implementation==0) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Database connection not open");
    }

    m_implementation->getPatterns(tag,crate,slot,patterns);

    if (patterns.size()==0) {
      XCEPT_RAISE(hcal::exception::ConfigurationItemNotFoundException,toolbox::toString("Not found '$s',%d,%d",tag.c_str(),crate,slot));
    }
  }


  void ConfigurationDatabase::getRBXdata(const std::string& tag,
					 const std::string& rbx,
					 RBXdatumType dtype,
					 std::map<RBXdatumId, RBXdatum>& RBXdata)
    throw (hcal::exception::ConfigurationDatabaseException) {

    if (m_implementation==0) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Database connection not open");
    }

    m_implementation->getRBXdata(tag,rbx,dtype,RBXdata);
  }

  void ConfigurationDatabase::getRBXpatterns(const std::string& tag,
					     const std::string& rbx,
					     std::map<RBXdatumId, RBXpattern>& patterns)
    throw (hcal::exception::ConfigurationDatabaseException) {

    if (m_implementation==0) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Database connection not open");
    }

    m_implementation->getRBXpatterns(tag,rbx,patterns);
  }

  void ConfigurationDatabase::getZSThresholds(const std::string& tag, int crate, int slot, std::map<ZSChannelId, int>& thresholds)
    throw (hcal::exception::ConfigurationDatabaseException) {

    if (m_implementation==0) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Database connection not open");
    }

    m_implementation->getZSThresholds(tag,crate,slot,thresholds);
  }

  void ConfigurationDatabase::getHLXMasks(const std::string& tag, int crate, int slot, std::map<FPGAId, HLXMasks>& m)
    throw (hcal::exception::ConfigurationDatabaseException) {

    if (m_implementation==0) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Database connection not open");
    }

    m_implementation->getHLXMasks(tag,crate,slot,m);
  }



  bool ConfigurationDatabase::FPGAId::operator<(const FPGAId& a) const {
    if (crate<a.crate) return true; if (crate>a.crate) return false;
    if (slot<a.slot) return true; if (slot>a.slot) return false;
    if (fpga<a.fpga) return true; if (fpga>a.fpga) return false;
    return false; // equal is not less
  }
  bool ConfigurationDatabase::LUTId::operator<(const LUTId& a) const {
    if (crate<a.crate) return true; if (crate>a.crate) return false;
    if (slot<a.slot) return true; if (slot>a.slot) return false;
    if (fpga<a.fpga) return true; if (fpga>a.fpga) return false;
    if (fiber_slb<a.fiber_slb) return true; if (fiber_slb>a.fiber_slb) return false;
    if (channel<a.channel) return true; if (channel>a.channel) return false;
    if (lut_type<a.lut_type) return true; if (lut_type>a.lut_type) return false;
    return false; // equal is not less
  }
  bool ConfigurationDatabase::PatternId::operator<(const PatternId& a) const {
    if (crate<a.crate) return true; if (crate>a.crate) return false;
    if (slot<a.slot) return true; if (slot>a.slot) return false;
    if (fpga<a.fpga) return true; if (fpga>a.fpga) return false;
    if (fiber<a.fiber) return true; if (fiber>a.fiber) return false;
    return false; // equal is not less
  }
  bool ConfigurationDatabase::ZSChannelId::operator<(const ZSChannelId& a) const {
    if (crate<a.crate) return true; if (crate>a.crate) return false;
    if (slot<a.slot) return true; if (slot>a.slot) return false;
    if (fpga<a.fpga) return true; if (fpga>a.fpga) return false;
    if (fiber<a.fiber) return true; if (fiber>a.fiber) return false;
    if (channel<a.channel) return true; if (channel>a.channel) return false;
    return false; // equal is not less
  }
  bool ConfigurationDatabase::RBXdatumId::operator<(const RBXdatumId& a) const {
    if (rm<a.rm) return true; if (rm>a.rm) return false;
    if (card<a.card) return true; if (card>a.card) return false;
    if (qie_or_gol<a.qie_or_gol) return true; if (qie_or_gol>a.qie_or_gol) return false;
    if (dtype<a.dtype) return true; if (dtype>a.dtype) return false;
    if (ltype<a.ltype) return true; if (ltype>a.ltype) return false;
    return false; // equal is not less
  }

}
