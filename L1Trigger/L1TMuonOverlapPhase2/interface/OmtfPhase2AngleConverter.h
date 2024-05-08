#ifndef OmtfPhase2AngleConverter_h
#define OmtfPhase2AngleConverter_h

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OmtfAngleConverter.h"

class OmtfPhase2AngleConverter : public OmtfAngleConverter {
public:
  OmtfPhase2AngleConverter(){};
  ~OmtfPhase2AngleConverter() override = default;

  // Convert DT phi to OMTF coordinate system.
  int getProcessorPhi(int phiZero, l1t::tftype part, int dtScNum, int dtPhi) const override;
};

#endif
