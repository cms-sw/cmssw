// Last commit: $Id$
// Latest tag:  $Name$
// Location:    $Source$

#ifndef OnlineDB_SiStripESSources_SiStripFedCablingBuilder_H
#define OnlineDB_SiStripESSources_SiStripFedCablingBuilder_H

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "boost/cstdint.hpp"

class SiStripFedCabling;
class SiStripFecCabling;
class DcuIdDetIdMap;

/**	
   \class SiStripFedCablingBuilder 
   \brief Utility class that builds various POOL-ORA objects that are
   used to populate the ORCON/ORCON conditions databases. The objects
   are built using the cached descriptions within the SiStripConfigDb
   interface class.
   \author R.Bainbridge
*/
class SiStripFedCablingBuilder {
  
 public:
  
  SiStripFedCablingBuilder();
  ~SiStripFedCablingBuilder(); 
  
  /** */
  static void createFedCabling( SiStripConfigDb* const,
				SiStripFedCabling&,
				SiStripConfigDb::DcuIdDetIdMap& );
  
  /** Builds the SiStripFedCabling conditions object using information
      found within the "module.xml" and "dcuinfo.xml" files. "Dummy"
      values are provided when necessary. */ 
  static void createFedCablingFromFedConnections( SiStripConfigDb* const,
						  SiStripFedCabling&,
						  SiStripConfigDb::DcuIdDetIdMap& );
  
  /** Builds the SiStripFedCabling conditions object using information
      found within the "fec.xml" and "dcuinfo.xml" files. "Dummy"
      values are provided when necessary. */
  static void createFedCablingFromDevices( SiStripConfigDb* const,
					   SiStripFedCabling&,
					   SiStripConfigDb::DcuIdDetIdMap& );
  
  /** Builds the SiStripFedCabling conditions object using information
      found within the "dcuinfo.xml" file (ie, based on DetIds). "Dummy"
      values are provided when necessary. */
  static void createFedCablingFromDetIds( SiStripConfigDb* const,
					  SiStripFedCabling&,
					  SiStripConfigDb::DcuIdDetIdMap& );
  
  /** Utility method that takes a FEC cabling object as input and
      returns (as an arg) the corresponding FED cabling object. */
  static void getFedCabling( const SiStripFecCabling&, 
			     SiStripFedCabling& );
  
  /** Utility method that takes a FED cabling object as input and
      returns (as an arg) the corresponding FEC cabling object. */
  static void getFecCabling( const SiStripFedCabling&, 
			     SiStripFecCabling& );
  
};

#endif // OnlineDB_SiStripESSources_SiStripFedCablingBuilder_H
