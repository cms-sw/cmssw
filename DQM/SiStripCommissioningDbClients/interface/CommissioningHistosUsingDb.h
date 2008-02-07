// Last commit: $Id: CommissioningHistosUsingDb.h,v 1.5 2007/12/12 15:06:15 bainbrid Exp $

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
  
  inline void doUploadAnal( bool );
  
  inline void doUploadConf( bool );
  
  virtual void addDcuDetIds();

  // ---------- protected methods ----------

 protected:
  
  virtual void create( SiStripConfigDb::AnalysisDescriptions& );

  virtual void create( SiStripConfigDb::AnalysisDescriptions&, Analysis ) {;}
  
  inline SiStripConfigDb* const db() const; 

  inline SiStripFedCabling* const cabling() const;
  
  inline bool doUploadAnal() const;
  
  inline bool doUploadConf() const;

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
bool CommissioningHistosUsingDb::doUploadAnal() const { return uploadConf_; }
bool CommissioningHistosUsingDb::doUploadConf() const { return uploadAnal_; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H
