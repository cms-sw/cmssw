// Last commit: $Id: SiStripFedCablingBuilderFromDb.h,v 1.4 2006/06/09 13:15:43 bainbrid Exp $
// Latest tag:  $Name: V00-01-01 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h,v $

#ifndef OnlineDB_SiStripESSources_SiStripFedCablingBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripFedCablingBuilderFromDb_H

#include "CalibTracker/SiStripConnectivity/interface/SiStripFedCablingESSource.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>

class SiStripFedCabling;
class SiStripFecCabling;
class DcuDetIdMap;

class SiStripFedCablingBuilderFromDb : public SiStripFedCablingESSource {
  
 public:

  SiStripFedCablingBuilderFromDb( const edm::ParameterSet& );
  ~SiStripFedCablingBuilderFromDb(); 
  
  /** Builds FED cabling using information from TK config DB. */
  virtual SiStripFedCabling* makeFedCabling();
  
  // ----------------------------------------------------------------------
  
  /** Generic method which builds FED cabling. Call ones of the three
      methods below depending on what descriptions are available
      within the database or which xml files are available. */
  static void buildFedCabling( SiStripConfigDb* const,
			       SiStripFedCabling&,
			       SiStripConfigDb::DcuDetIdMap& );

  /** Builds the SiStripFedCabling conditions object using information
      found within the "module.xml" and "dcuinfo.xml" files. "Dummy"
      values are provided when necessary. */ 
  static void buildFedCablingFromFedConnections( SiStripConfigDb* const,
						 SiStripFedCabling&,
						 SiStripConfigDb::DcuDetIdMap& );
  
  /** Builds the SiStripFedCabling conditions object using information
      found within the "fec.xml" and "dcuinfo.xml" files. "Dummy"
      values are provided when necessary. */
  static void buildFedCablingFromDevices( SiStripConfigDb* const,
					  SiStripFedCabling&,
					  SiStripConfigDb::DcuDetIdMap& );
  
  /** Builds the SiStripFedCabling conditions object using information
      found within the "dcuinfo.xml" file (ie, based on DetIds). "Dummy"
      values are provided when necessary. */
  static void buildFedCablingFromDetIds( SiStripConfigDb* const,
					 SiStripFedCabling&,
					 SiStripConfigDb::DcuDetIdMap& );

  // ----------------------------------------------------------------------
  
  /** Utility method that takes a FEC cabling object as input and
      returns (as an arg) the corresponding FED cabling object. */
  static void getFedCabling( const SiStripFecCabling&, 
			     SiStripFedCabling& );
  
  /** Utility method that takes a FED cabling object as input and
      returns (as an arg) the corresponding FEC cabling object. */
  static void getFecCabling( const SiStripFedCabling&, 
			     SiStripFecCabling& );
  
 protected:
  
  /** Virtual method that is called by makeFedCabling() to allow FED
      cabling to be written to the conds DB (local or otherwise). */
  virtual void writeFedCablingToCondDb() {;}
  
  /** Returns pointer to database interface for any derived classes
      that implement writeFedCablingToCondDb(). */
  inline SiStripConfigDb* const database();
  
 private:
  
  /** Access to the configuration DB interface class. */
  SiStripConfigDb* db_;
  
  /** Vector of strings holding partition names. */
  std::vector<std::string> partitions_;

  /** Defines the MessageLogger category for this class. */
  static const std::string logCategory_;
  
};

SiStripConfigDb* const SiStripFedCablingBuilderFromDb::database() { return db_; }

#endif // OnlineDB_SiStripESSources_SiStripFedCablingBuilderFromDb_H

