// Last commit: $Id: CommissioningHistosUsingDb.h,v 1.9 2008/03/06 13:30:50 delaer Exp $

#ifndef DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H
#define DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>

class SiStripConfigDb;
class SiStripFedCabling;
class DQMOldReceiver;

class CommissioningHistosUsingDb : public virtual CommissioningHistograms {
  
 public:

  // ---------- con(de)structors ----------

  // DEPRECATE
  CommissioningHistosUsingDb( SiStripConfigDb* const,
			      sistrip::RunType = sistrip::UNDEFINED_RUN_TYPE );

  CommissioningHistosUsingDb( SiStripConfigDb* const,
			      DQMOldReceiver* const,			      
			      sistrip::RunType = sistrip::UNDEFINED_RUN_TYPE );
  
  virtual ~CommissioningHistosUsingDb();

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
  
  virtual void create( SiStripConfigDb::AnalysisDescriptionsV& );

  virtual void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ) {;}
  
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
