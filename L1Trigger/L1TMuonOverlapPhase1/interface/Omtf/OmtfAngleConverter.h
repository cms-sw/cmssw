/*
 * OmtfAngleConverter.h
 *
 *  Created on: Jan 14, 2019
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_OMTFANGLECONVERTER_H_
#define L1T_OmtfP1_OMTFANGLECONVERTER_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/ProcConfigurationBase.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include <memory>

class RPCGeometry;
class CSCGeometry;
class CSCLayer;
class DTGeometry;

class L1MuDTChambPhDigi;
class L1MuDTChambThDigi;
class L1MuDTChambThContainer;
class CSCCorrelatedLCTDigi;
class RPCDigi;

class DTChamberId;
class CSCDetId;
class RPCDetId;

struct MuonGeometryTokens {
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeometryEsToken;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeometryEsToken;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeometryEsToken;
};

class OmtfAngleConverter {
public:
  OmtfAngleConverter() {}

  virtual ~OmtfAngleConverter();

  ///Update the Geometry with current Event Setup
  virtual void checkAndUpdateGeometry(const edm::EventSetup&,
                                      const ProcConfigurationBase* config,
                                      const MuonGeometryTokens& muonGeometryTokens);

  /// get phi of DT,CSC and RPC azimutal angle digi in processor scale, used by OMTF algorithm.
  /// in case of wrong phi returns OMTFConfiguration::instance()->nPhiBins
  /// phiZero - desired phi where the scale should start, should be in the desired scale, use getProcessorPhiZero to obtain it
  virtual int getProcessorPhi(int phiZero, l1t::tftype part, int dtScNum, int dtPhi) const;

  virtual int getProcessorPhi(
      int phiZero, l1t::tftype part, const CSCDetId& csc, const CSCCorrelatedLCTDigi& digi, unsigned int iInput) const;

  virtual int getProcessorPhi(int phiZero,
                              l1t::tftype part,
                              const RPCDetId& rollId,
                              const unsigned int& digi1,
                              const unsigned int& digi2) const;

  ///Convert local eta coordinate to global digital microGMT scale.
  ///theta is  returned only if in the dtThDigis is only one hit, otherwise eta = 95 or middle of the chamber
  virtual int getGlobalEta(const DTChamberId dTChamberId, const L1MuDTChambThContainer* dtThDigis, int bxNum) const;

  ///Convert local eta coordinate to global digital microGMT scale.
  virtual int getGlobalEta(unsigned int rawid, const CSCCorrelatedLCTDigi& aDigi, float& r) const;

  ///Convert local eta coordinate to global digital microGMT scale.
  virtual int getGlobalEtaRpc(unsigned int rawid, const unsigned int& aDigi, float& r) const;

protected:
  ///Check orientation of strips in given CSC chamber
  virtual bool isCSCCounterClockwise(const CSCLayer* layer) const;

  ///Find BTI group
  virtual const int findBTIgroup(const L1MuDTChambPhDigi& aDigi, const L1MuDTChambThContainer* dtThDigis);

  // pointers to the current geometry records
  unsigned long long _geom_cache_id = 0;
  edm::ESHandle<RPCGeometry> _georpc;
  edm::ESHandle<CSCGeometry> _geocsc;
  edm::ESHandle<DTGeometry> _geodt;

  edm::ESWatcher<MuonGeometryRecord> muonGeometryRecordWatcher;

  const ProcConfigurationBase* config = nullptr;
  ///Number of phi bins along 2Pi.
  unsigned int nPhiBins = 0;
};

#endif /* L1T_OmtfP1_OMTFANGLECONVERTER_H_ */
