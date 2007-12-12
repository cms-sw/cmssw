// Last commit: $Id: CommissioningHistosUsingDb.h,v 1.4 2007/05/24 15:59:44 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H
#define DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H

#include <boost/cstdint.hpp>
#include <string>
#include <map>

class SiStripConfigDb;
class SiStripFedCabling;
class CommissioningAnalysis;

class CommissioningHistosUsingDb {
  
 public:

  /** */ 
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
  
  /** */ 
  CommissioningHistosUsingDb( const DbParams& );

  /** */ 
  CommissioningHistosUsingDb( SiStripConfigDb* const );
  
  /** */ 
  virtual ~CommissioningHistosUsingDb();

  inline void testOnly( bool );
  
 protected:

  /** */
  void addDcuDetId( CommissioningAnalysis* );

  /** */
  SiStripConfigDb* db_;

  /** */
  SiStripFedCabling* cabling_;

  bool test_;
  
 private: 

  CommissioningHistosUsingDb() {;}

};

void CommissioningHistosUsingDb::testOnly( bool test ) { test_ = test; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H
