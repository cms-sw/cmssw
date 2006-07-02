#ifndef RPCReadOutMapping_H
#define RPCReadOutMapping_H
/** \class RPCReadOutMapping
 *
 *  Description:
 *       Class to map read-out channels to physical RPC strips
 *
 *  $Date: 2006/06/30 06:21:06 $
 *  $Revision: 1.3 $
 *  \author Marcello Maggi -- INFN Bari
 *
 */

#include <map>
#include <vector>
#include <utility>
#include <string>
#include <boost/cstdint.hpp>

#include "CondFormats/RPCObjects/interface/DccSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberRawDataSpec.h"
class LinkBoardSpec;


class RPCReadOutMapping {
public:

  RPCReadOutMapping(const std::string & version = ""); 

  /// FED identified by ID
  const DccSpec * dcc( int dccId) const;

  /// Range of FED IDs in map (min and max id) 
  std::pair<int,int> dccNumberRange() const;

  /// all FEDs in map
  std::vector<const DccSpec*> dccList() const;

  /// conversion between electronic and detector indexing
  std::pair< const LinkBoardSpec*, const ChamberLocationSpec* > location (const ChamberRawDataSpec & ele) const;  

  /// attach FED to map
  void add(const DccSpec & dcc);

  /// version as string
  const std::string & version() const { return theVersion; }

private:
   typedef std::map<int, DccSpec>::const_iterator IMAP;
   std::map<int, DccSpec> theFeds;
   std::string theVersion;

  
};

#endif // RPCReadOutMapping_H

