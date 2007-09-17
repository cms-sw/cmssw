#ifndef HLTriggerOffline_Egamma_h
#define HLTriggerOffline_Egamma_h

#include <vector>

/* ElecHLTCutVars is used to store isolation variables for electrons and other 
 * variables that could be used to make cuts on electrons in the HLT */
struct ElecHLTCutVarsPreTrack {
  bool l1Match;
  double Et; // transverse energy
  double IHcal; // electron Hcal isolation (CMS Note 2006/078)
  int pixMatch;
  double eta;
  double phi;
  double mcEt;
  double mcEta;
  double mcPhi;
};

struct ElecHLTCutVars {
  bool l1Match;
  double Et; // transverse energy
  double IHcal; // electron Hcal isolation (CMS Note 2006/078)
  int pixMatch;
  double Eoverp;
  double Itrack; // electron track isolation (CMS Note 2006/078)
  double eta;
  double phi;
  double mcEt;
  double mcEta;
  double mcPhi;
};

/* PhotHLTCutVars provides a similar function to that of ElecHLTCutVars but for
 * photons.  The definition of the variables is not necessarilly the same */
struct PhotHLTCutVars {
  bool l1Match;
  double Et; // transverse energy
  double IEcal; // photon Ecal isolation (CMS Note 2006/078)
  double IHcal; // photon Hcal isolation (CMS Note 2006/078)
  double Itrack; // photon track isolation (CMS Note 2006/078)
  double eta;
  double phi;
  double mcEt;
  double mcEta;
  double mcPhi;
};

struct HLTTiming {
  double l1Match;
  double Et;
  double ElecIHcal;
  double pixMatch;
  double Eoverp;
  double ElecItrack;
  double IEcal;
  double PhotIHcal;
  double PhotItrack;
};

/* CaloVars stores basic information about MC particles in the event */
struct CaloVars {
  double Et; // MC transverse energy
  double eta;
  double phi;
};

/* Collections of the above data types */
typedef std::vector<ElecHLTCutVarsPreTrack> ElecHLTCutVarsPreTrackCollection;
typedef std::vector<ElecHLTCutVars> ElecHLTCutVarsCollection;
typedef std::vector<PhotHLTCutVars> PhotHLTCutVarsCollection;
typedef std::vector<CaloVars> CaloVarsCollection;
typedef std::vector<HLTTiming> HLTTimingCollection;
#endif
