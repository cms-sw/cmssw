
#ifndef DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H
#define DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "boost/range/iterator_range.hpp"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class SiStripConfigDb;
class SiStripFedCabling;

class CommissioningHistosUsingDb : public virtual CommissioningHistograms {
  
 // ---------- public interface ----------

 public:

  CommissioningHistosUsingDb( SiStripConfigDb* const,
                              sistrip::RunType = sistrip::UNDEFINED_RUN_TYPE );

  ~CommissioningHistosUsingDb() override;

  void configure( const edm::ParameterSet&, const edm::EventSetup& ) override;

  void uploadToConfigDb();
  
  bool doUploadAnal() const;
  
  bool doUploadConf() const;

  void doUploadAnal( bool );
  
  void doUploadConf( bool );
  
 // ---------- protected methods ----------

 protected:
  
  void buildDetInfo();
  
  virtual void addDcuDetIds();
  
  virtual void uploadConfigurations() {;}
  
  void uploadAnalyses();
  
  virtual void createAnalyses( SiStripConfigDb::AnalysisDescriptionsV& );
  
  virtual void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ) {;}
  
  SiStripConfigDb* const db() const; 

  SiStripFedCabling* const cabling() const;
  
  class DetInfo { 
  public:
    uint32_t dcuId_;
    uint32_t detId_;
    uint16_t pairs_;
    DetInfo() : 
      dcuId_(sistrip::invalid32_), 
      detId_(sistrip::invalid32_), 
      pairs_(sistrip::invalid_) {;}
  };
  
  std::pair<std::string,DetInfo> detInfo( const SiStripFecKey& );
  
  bool deviceIsPresent( const SiStripFecKey& );
  
 // ---------- private member data ----------
  
 private: 
  
  CommissioningHistosUsingDb(); 

  sistrip::RunType runType_;
  
  SiStripConfigDb* db_;
  
  SiStripFedCabling* cabling_;

  typedef std::map<uint32_t,DetInfo> DetInfos;
  
  std::map<std::string,DetInfos> detInfo_;
  
  bool uploadAnal_;
  
  bool uploadConf_;
  
};

inline void CommissioningHistosUsingDb::doUploadConf( bool upload ) { uploadConf_ = upload; }
inline void CommissioningHistosUsingDb::doUploadAnal( bool upload ) { uploadAnal_ = upload; }

inline bool CommissioningHistosUsingDb::doUploadAnal() const { return uploadAnal_; }
inline bool CommissioningHistosUsingDb::doUploadConf() const { return uploadConf_; }

inline SiStripConfigDb* const CommissioningHistosUsingDb::db() const { return db_; } 
inline SiStripFedCabling* const CommissioningHistosUsingDb::cabling() const { return cabling_; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H
