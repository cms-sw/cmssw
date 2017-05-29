#include "FastSimulation/PPSFastObjects/interface/PPSBaseTrack.h"
PPSBaseTrack::PPSBaseTrack() {};
PPSBaseTrack::PPSBaseTrack(const TLorentzVector& p,double _t, double _xi):t(_t),xi(_xi),
              pT(0.),momentum(0.),eta(0.),phi(0.),theta(0.),E(0.),Px(0.),Py(0.),Pz(0.){
   if (p.P()==0) return;
   momentum =p.P();
   phi      =p.Phi();
   theta    =p.Theta();
   pT       =p.Pt();
   eta      =p.Eta();
   E        =p.E();
   Px       =p.Px();
   Py       =p.Py();
   Pz       =p.Pz();
};
