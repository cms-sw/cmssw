#ifndef RPCReadOutMapping_H
#define RPCReadOutMapping_H
/** \class RPCReadOutMapping
 *
 *  Description:
 *       Class to map read-out channels to physical RPC strips
 */

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <vector>
#include <utility>
#include <string>

#include "CondFormats/RPCObjects/interface/DccSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"
#include "CondFormats/RPCObjects/interface/LinkBoardPackedStrip.h"
#include <cstdint>
class LinkBoardSpec;

class RPCReadOutMapping {
public:
  /// first member is DetUnit ID, second strip in DetUnit frame
  typedef std::pair<uint32_t, int> StripInDetUnit;

  RPCReadOutMapping(const std::string& version = "");
  virtual ~RPCReadOutMapping() {}

  /// FED identified by ID
  const DccSpec* dcc(int dccId) const;

  /// Range of FED IDs in map (min and max id)
  std::pair<int, int> dccNumberRange() const;

  /// all FEDs in map
  std::vector<const DccSpec*> dccList() const;

  /// attach FED to map
  void add(const DccSpec& dcc);

  /// version as string
  const std::string& version() const { return theVersion; }

  /// conversion between electronic and detector indexing
  virtual const LinkBoardSpec* location(const LinkBoardElectronicIndex& ele) const;

  /// convert strip location as in raw data (LB and LBchannel) to detUnit frame
  virtual StripInDetUnit detUnitFrame(const LinkBoardSpec& location, const LinkBoardPackedStrip& packedStrip) const;

  /// connection "paths" that lead from one digi to one strip
  std::vector<std::pair<LinkBoardElectronicIndex, LinkBoardPackedStrip> > rawDataFrame(
      const StripInDetUnit& duFrame) const;

  // TEMPORARY
  std::vector<const LinkBoardSpec*> getLBforChamber(const std::string& name) const;
  std::pair<LinkBoardElectronicIndex, int> getRAWSpecForCMSChamberSrip(uint32_t detId,
                                                                       int strip,
                                                                       int dccInputChannel) const;

private:
  typedef std::map<int, DccSpec>::const_iterator IMAP;
  std::map<int, DccSpec> theFeds;
  std::string theVersion;

  COND_SERIALIZABLE;
};

#endif  // RPCReadOutMapping_H
