#ifndef RPCReadOutMapping_H
#define RPCReadOutMapping_H
/** \class RPCReadOutMapping
 *
 *  Description:
 *       Class to map read-out channels to physical RPC strips
 */

#include <map>
#include <vector>
#include <utility>
#include <string>
#include <boost/cstdint.hpp>

#include "CondFormats/RPCObjects/interface/DccSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberRawDataSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardChannelCoding.h"
class LinkBoardSpec;


class RPCReadOutMapping {
public:

  /// first member is DetUnit ID, second strip in DetUnit frame 
  typedef std::pair<uint32_t,int> StripInDetUnit;

  RPCReadOutMapping(const std::string & version = ""); 

  /// FED identified by ID
  const DccSpec * dcc( int dccId) const;

  /// Range of FED IDs in map (min and max id) 
  std::pair<int,int> dccNumberRange() const;

  /// all FEDs in map
  std::vector<const DccSpec*> dccList() const;

  /// conversion between electronic and detector indexing
  const LinkBoardSpec* location (const ChamberRawDataSpec & ele) const;  

  /// attach FED to map
  void add(const DccSpec & dcc);

  /// version as string
  const std::string & version() const { return theVersion; }

  /// get linkboards for given chamber name
  std::vector<const LinkBoardSpec*> getLBforChamber(const std::string & name) const;

  /// get RAW data specification for a given CMS strip in given chamber
  std::pair<ChamberRawDataSpec, int> getRAWSpecForCMSChamberSrip(uint32_t  detId, int strip,  int dccInputChannel) const;

  /// get strip info for given LB channel in given LB location.
  StripInDetUnit strip(const ChamberRawDataSpec & linkboard, int chanelLB) const;

  /// convert strip location as in raw data to detUnit frame
  StripInDetUnit detUnitFrame(const LinkBoardSpec* location, 
      int febInLB, int stripPinInFeb) const;

  std::pair< ChamberRawDataSpec, LinkBoardChannelCoding> rawDataFrame (uint32_t rawDetId, int stripInDU) const;
  

private:
   typedef std::map<int, DccSpec>::const_iterator IMAP;
   std::map<int, DccSpec> theFeds;
   std::string theVersion;
  
};

#endif // RPCReadOutMapping_H

