// Last commit: $Id: CommissioningHistosUsingDb.h,v 1.2 2007/03/21 16:55:06 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H
#define DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H

#include <boost/cstdint.hpp>
#include <string>

class SiStripConfigDb;

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

  bool test_;
  
 private: 

  CommissioningHistosUsingDb() {;}

};

void CommissioningHistosUsingDb::testOnly( bool test ) { test_ = test; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H
