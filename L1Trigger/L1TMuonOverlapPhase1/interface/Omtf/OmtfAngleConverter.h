/*
 * OmtfAngleConverter.h
 *
 *  Created on: Jan 14, 2019
 *      Author: kbunkow
 */

#ifndef OMTF_OMTFANGLECONVERTER_H_
#define OMTF_OMTFANGLECONVERTER_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/AngleConverterBase.h"

class OmtfAngleConverter : public AngleConverterBase {
public:
  OmtfAngleConverter() : AngleConverterBase(){};

  ~OmtfAngleConverter() override;

  ///Convert local eta coordinate to global digital microGMT scale.
  ///theta is  returned only if in the dtThDigis is only one hit, otherwise eta = 95 or middle of the chamber
  virtual int getGlobalEta(const DTChamberId dTChamberId, const L1MuDTChambThContainer *dtThDigis, int bxNum) const;

  ///Convert local eta coordinate to global digital microGMT scale.
  virtual int getGlobalEta(unsigned int rawid, const CSCCorrelatedLCTDigi &aDigi) const;

  ///Convert local eta coordinate to global digital microGMT scale.
  virtual int getGlobalEtaRpc(unsigned int rawid, const unsigned int &aDigi) const;
};

#endif /* INTERFACE_OMTF_OMTFANGLECONVERTER_H_ */
