#ifndef DataFormats_L1Scouting_L1ScoutingMuon_h
#define DataFormats_L1Scouting_L1ScoutingMuon_h

#include <cmath>

class ScMuon {
  public:
  int pt, eta, phi, qual, chrg, chrgv;
  int iso, index, etae, phie, ptUncon;
  //float fpt, feta, fphi, fetae, fphie, fptUncon;

  inline float getPt(){return 0.05*(pt-1);}
  inline float getEta(){return 0.0870/8*eta;}
  inline float getPhi(){return 2.*M_PI/576.*phi;}
  inline float getPtUncon(){return 0.05*(ptUncon-1);}
  inline float getEtaExt(){return 0.0870/8*etae;}
  inline float getPhiExt(){return 2.*M_PI/576.*phie;}
};

#endif // DataFormats_L1Scouting_L1ScoutingMuon_h