#ifndef RPCReadOutMappingWithFastSearch_H
#define RPCReadOutMappingWithFastSearch_H

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include <string>
#include <map>

class RPCReadOutMappingWithFastSearch : public RPCReadOutMapping {
public:
  RPCReadOutMappingWithFastSearch();
  ~RPCReadOutMappingWithFastSearch() override{} 

  /// takes ownership of map
  void init(const RPCReadOutMapping * arm);

  const LinkBoardSpec* location (const LinkBoardElectronicIndex & ele) const override;

  RPCReadOutMapping::StripInDetUnit detUnitFrame(
      const LinkBoardSpec& location, const LinkBoardPackedStrip & lbstrip) const override;

private:
  std::string theVersion;
  const RPCReadOutMapping * theMapping;

  struct lessMap {
     bool operator()(const LinkBoardElectronicIndex & lb1, const LinkBoardElectronicIndex & lb2) const;
  };

  typedef std::map<LinkBoardElectronicIndex, const LinkBoardSpec*, lessMap> LBMap;
  LBMap theLBMap;
};
#endif
