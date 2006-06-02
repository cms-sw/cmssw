// Last commit: $Id$
// Latest tag:  $Name$
// Location:    $Source$

#ifndef OnlineDB_SiStripESSources_SiStripFedCablingBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripFedCablingBuilderFromDb_H

#include "CalibTracker/SiStripConnectivity/interface/SiStripFedCablingESSource.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>

class SiStripConfigDb;

class SiStripFedCablingBuilderFromDb : public SiStripFedCablingESSource {
  
 public:

  SiStripFedCablingBuilderFromDb( const edm::ParameterSet& );
  ~SiStripFedCablingBuilderFromDb(); 
  
  /** Builds FED cabling using information from TK config DB. */
  virtual SiStripFedCabling* makeFedCabling();
  
 protected:
  
  /** */
  virtual void writeFedCablingToCondDb() {;}
  
  /** */
  inline SiStripConfigDb* const database();
  
 private:

  /** */
  SiStripConfigDb* db_;

  std::vector<std::string> partitions_;
  
};

SiStripConfigDb* const SiStripFedCablingBuilderFromDb::database() { return db_; }

#endif // OnlineDB_SiStripESSources_SiStripFedCablingBuilderFromDb_H
