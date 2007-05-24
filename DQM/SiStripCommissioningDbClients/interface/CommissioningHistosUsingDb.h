// Last commit: $Id: CommissioningHistosUsingDb.h,v 1.3 2007/04/04 07:21:08 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H
#define DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H

#include <boost/cstdint.hpp>
#include <string>

class SiStripConfigDb;
class SiStripFedCabling;

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
  SiStripConfigDb* db_;

  /** */
  SiStripFedCabling* cabling_;

  bool test_;
  
 private: 

  CommissioningHistosUsingDb() {;}

};

void CommissioningHistosUsingDb::testOnly( bool test ) { test_ = test; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H
