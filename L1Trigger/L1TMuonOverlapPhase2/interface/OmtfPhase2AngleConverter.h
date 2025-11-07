#ifndef OmtfPhase2AngleConverter_h
#define OmtfPhase2AngleConverter_h

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OmtfAngleConverter.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTThContainer.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

class OmtfPhase2AngleConverter : public OmtfAngleConverter {
public:
  OmtfPhase2AngleConverter() : OmtfAngleConverter() {}
  ~OmtfPhase2AngleConverter() override = default;

  // Convert DT phi to OMTF coordinate system.
  int getProcessorPhi(int phiZero, l1t::tftype part, int dtScNum, int dtPhi) const override;

  //using different name of the method to avoid hiding OmtfAngleConverter methods getGlobalEta
  int getGlobalEtaPhase2(DTChamberId dTChamberId, const L1Phase2MuDTThContainer *dtThDigis, int bxNum) const;
};

#endif
