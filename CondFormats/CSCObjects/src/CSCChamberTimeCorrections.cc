#include "CondFormats/CSCObjects/interface/CSCChamberTimeCorrections.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

CSCChamberTimeCorrections::CSCChamberTimeCorrections(){}
CSCChamberTimeCorrections::~CSCChamberTimeCorrections(){}

const CSCChamberTimeCorrections::ChamberTimeCorrections & CSCChamberTimeCorrections::item(const CSCDetId & cscId) const
 {
  CSCIndexer indexer;
  return chamberCorrections[ indexer.chamberIndex(cscId)-1 ]; // no worries about range!
 }
