#ifndef HLTriggerOffline_Egamma_h
#define HLTriggerOffline_Egamma_h

#include <vector>

/* ElecHLTCutVars is used to store isolation variables for electrons and other 
 * variables that could be used to make cuts on electrons in the HLT */
struct ElecHLTCutVarsPreTrack {
  bool l1Match;
  float Et; // transverse energy
  float IHcal; // electron Hcal isolation (CMS Note 2006/078)
  int pixMatch;
  float eta;
  float phi;
  float mcEt;
  float mcEta;
  float mcPhi;
};

struct ElecHLTCutVars {
  bool l1Match;
  float Et; // transverse energy
  float IHcal; // electron Hcal isolation (CMS Note 2006/078)
  int pixMatch;
  float Eoverp;
  float Itrack; // electron track isolation (CMS Note 2006/078)
  float eta;
  float phi;
  float mcEt;
  float mcEta;
  float mcPhi;
};

/* PhotHLTCutVars provides a similar function to that of ElecHLTCutVars but for
 * photons.  The definition of the variables is not necessarilly the same */
struct PhotHLTCutVars {
  bool l1Match;
  float Et; // transverse energy
  float IEcal; // photon Ecal isolation (CMS Note 2006/078)
  float IHcal; // photon Hcal isolation (CMS Note 2006/078)
  int Itrack; // photon track isolation (CMS Note 2006/078)
  float eta;
  float phi;
  float mcEt;
  float mcEta;
  float mcPhi;
};

struct HLTTiming {
  float l1Match;
  float Et;
  float ElecIHcal;
  float pixMatch;
  float Eoverp;
  float ElecItrack;
  float IEcal;
  float PhotIHcal;
  float PhotItrack;
};

/* CaloVars stores basic information about MC particles in the event */
struct CaloVars {
  float Et; // MC transverse energy
  float eta;
  float phi;
};

/* Collections of the above data types */
typedef std::vector<ElecHLTCutVarsPreTrack> ElecHLTCutVarsPreTrackCollection;
typedef std::vector<ElecHLTCutVars> ElecHLTCutVarsCollection;
typedef std::vector<PhotHLTCutVars> PhotHLTCutVarsCollection;
typedef std::vector<CaloVars> CaloVarsCollection;
typedef std::vector<HLTTiming> HLTTimingCollection;
#endif
