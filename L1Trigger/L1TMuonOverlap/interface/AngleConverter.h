#ifndef ANGLECONVERTER_H
#define ANGLECONVERTER_H

#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <memory>

class CSCLayer;

class L1MuDTChambPhDigi;
class L1MuDTChambThContainer;
class CSCCorrelatedLCTDigi;
class RPCDigi;
class CSCDetId;
class RPCDetId;

class AngleConverter {
public:
  AngleConverter(edm::ConsumesCollector &, bool getDuringEvent = true);
  ~AngleConverter();

  ///Update the Geometry with current Event Setup
  void checkAndUpdateGeometry(const edm::EventSetup &, unsigned int);

  /// get phi of DT,CSC and RPC azimutal angle digi in processor scale, used by OMTF algorithm.
  /// in case of wrong phi returns OMTFConfiguration::instance()->nPhiBins
  int getProcessorPhi(unsigned int iProcessor, l1t::tftype part, const L1MuDTChambPhDigi &digi) const;
  int getProcessorPhi(unsigned int iProcessor,
                      l1t::tftype part,
                      const CSCDetId &csc,
                      const CSCCorrelatedLCTDigi &digi) const;
  int getProcessorPhi(unsigned int iProcessor, l1t::tftype part, const RPCDetId &rollId, const unsigned int &digi) const;
  int getProcessorPhi(unsigned int iProcessor,
                      l1t::tftype part,
                      const RPCDetId &rollId,
                      const unsigned int &digi1,
                      const unsigned int &digi2) const;

  ///Convert local eta coordinate to global digital microGMT scale.
  int getGlobalEta(unsigned int rawid, const L1MuDTChambPhDigi &aDigi, const L1MuDTChambThContainer *dtThDigis);

  ///Convert local eta coordinate to global digital microGMT scale.
  int getGlobalEta(unsigned int rawid, const CSCCorrelatedLCTDigi &aDigi);

  ///Convert local eta coordinate to global digital microGMT scale.
  int getGlobalEta(unsigned int rawid, const unsigned int &aDigi);

private:
  ///Check orientation of strips in given CSC chamber
  bool isCSCCounterClockwise(const std::unique_ptr<const CSCLayer> &layer) const;

  ///Find BTI group
  const int findBTIgroup(const L1MuDTChambPhDigi &aDigi, const L1MuDTChambThContainer *dtThDigis);

  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeometryToken_;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeometryToken_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeometryToken_;

  // pointers to the current geometry records
  unsigned long long _geom_cache_id;
  RPCGeometry const *_georpc;
  CSCGeometry const *_geocsc;
  DTGeometry const *_geodt;

  ///Number of phi bins along 2Pi.
  unsigned int nPhiBins;
};

#endif
