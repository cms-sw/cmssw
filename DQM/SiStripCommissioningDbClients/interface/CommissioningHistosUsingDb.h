// Last commit: $Id: $

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
  virtual ~CommissioningHistosUsingDb();
  
 protected:

  /** */
  SiStripConfigDb* db_;

 private: 

  CommissioningHistosUsingDb() {;}

};

#endif // DQM_SiStripCommissioningClients_CommissioningHistosUsingDb_H
