#include "L1Trigger/L1TMuonOverlapPhase2/interface/OmtfPhase2AngleConverter.h"

int OmtfPhase2AngleConverter::getProcessorPhi(int phiZero, l1t::tftype part, int dtScNum, int dtPhi) const {
  constexpr int dtPhiBins = 65536;          //65536. for [-0.5,0.5] radians
  double hsPhiPitch = 2 * M_PI / nPhiBins;  // width of phi Pitch, related to halfStrip at CSC station 2

  int sector = dtScNum + 1;  //NOTE: there is a inconsistency in DT sector numb. Thus +1 needed to get detector numb.

  double scale = 0.5 / dtPhiBins / hsPhiPitch;  //was 0.8
  int scale_coeff = lround(scale * (1 << 15));

  int ichamber = sector - 1;
  if (ichamber > 6)
    ichamber = ichamber - 12;

  int offsetGlobal = (int)nPhiBins * ichamber / 12;

  int phiConverted = ((dtPhi * scale_coeff) >> 15) + offsetGlobal - phiZero;

  return config->foldPhi(phiConverted);
}

/* TODO implement the etat for the phase2 stubs
int getGlobalEta(const DTChamberId dTChamberId, const L1Phase2MuDTThContainer *dtThDigis, int bxNum) const {

  //const DTChamberId dTChamberId(aDigi.whNum(),aDigi.stNum(),aDigi.scNum()+1);
  DTTrigGeom trig_geom(_geodt->chamber(dTChamberId), false);


}
*/
