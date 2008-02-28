// Last commit: $Id: CommissioningHistosUsingDb.h,v 1.7 2008/02/14 13:53:04 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H
#define DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>

class SiStripConfigDb;
class SiStripFedCabling;
class MonitorUserInterface;

class CommissioningHistosUsingDb : public virtual CommissioningHistograms {
  
 public:

  // ---------- con(de)structors ----------

  // DEPRECATE
  class DbParams;

  // DEPRECATE
  CommissioningHistosUsingDb( const DbParams& );

  // DEPRECATE
  CommissioningHistosUsingDb( SiStripConfigDb* const,
			      sistrip::RunType = sistrip::UNDEFINED_RUN_TYPE );

  CommissioningHistosUsingDb( SiStripConfigDb* const,
			      MonitorUserInterface* const,			      
			      sistrip::RunType = sistrip::UNDEFINED_RUN_TYPE );
  
  virtual ~CommissioningHistosUsingDb();

  // ---------- db connection params ----------

  // DEPRECATE
  class DbParams {
  public:
    bool usingDb_;
    std::string confdb_;
    std::string partition_;
    uint32_t major_;
    uint32_t minor_;
    DbParams();
    ~DbParams() {;}
  };

  // ---------- public interface ----------

  void uploadAnalyses();
  
  virtual void uploadConfigurations() {;}
  
  inline bool doUploadAnal() const;
  
  inline bool doUploadConf() const;

  inline void doUploadAnal( bool );
  
  inline void doUploadConf( bool );
  
  virtual void addDcuDetIds();

  // ---------- protected methods ----------

 protected:
  
  virtual void create( SiStripConfigDb::AnalysisDescriptions& );

  virtual void create( SiStripConfigDb::AnalysisDescriptions&, Analysis ) {;}
  
  inline SiStripConfigDb* const db() const; 

  inline SiStripFedCabling* const cabling() const;
  
  class DetInfo { 
  public:
    DetInfo() : 
      dcuId_(sistrip::invalid32_), 
      detId_(sistrip::invalid32_), 
      pairs_(sistrip::invalid_) {;}
    uint32_t dcuId_;
    uint32_t detId_;
    uint16_t pairs_;
  };
  
  typedef std::map<uint32_t,DetInfo> DetInfoMap;
  
  void detInfo( DetInfoMap& );
  
  // ---------- private member data ----------
  
 private: 

  CommissioningHistosUsingDb(); // private constructor

  sistrip::RunType runType_;
  
  SiStripConfigDb* db_;
  
  SiStripFedCabling* cabling_;
  
  bool uploadAnal_;
  
  bool uploadConf_;

};

void CommissioningHistosUsingDb::doUploadConf( bool upload ) { uploadConf_ = upload; }
void CommissioningHistosUsingDb::doUploadAnal( bool upload ) { uploadAnal_ = upload; }
SiStripConfigDb* const CommissioningHistosUsingDb::db() const { return db_; } 
SiStripFedCabling* const CommissioningHistosUsingDb::cabling() const { return cabling_; }
bool CommissioningHistosUsingDb::doUploadAnal() const { return uploadAnal_; }
bool CommissioningHistosUsingDb::doUploadConf() const { return uploadConf_; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H
