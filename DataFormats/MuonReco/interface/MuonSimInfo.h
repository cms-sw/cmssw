#ifndef MuonReco_MuonSimInfo_h
#define MuonReco_MuonSimInfo_h

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

namespace reco {
/*


 CLASSIFICATION: For each RECO Muon, match to SIM particle, and then:
  - If the SIM is not a Muon, label as Punchthrough (1) except if it is an electron or positron (11)
  - If the SIM is a Muon, then look at it's provenance.
     A) the SIM muon is also a GEN muon, whose parent is NOT A HADRON AND NOT A TAU
        -> classify as "primary" (4).
     B) the SIM muon is also a GEN muon, whose parent is HEAVY FLAVOURED HADRON OR A TAU
        -> classify as "heavy flavour" (3)
     C) classify as "light flavour/decay" (2)

  In any case, if the TP is not preferentially matched back to the same RECO muon,
  label as Ghost (flip the classification)


 FLAVOUR:
  - for non-muons: 0
  - for primary muons: 13
  - for non primary muons: flavour of the mother: std::abs(pdgId) of heaviest quark, or 15 for tau

*/

  enum MuonSimType { 
    Unknown                     = 999, 
    NotMatched                  = 0,
    MatchedPunchthrough         = 1,
    MatchedElectron             = 11,
    MatchedPrimaryMuon          = 4,
    MatchedMuonFromHeavyFlavour = 3,
    MatchedMuonFromLightFlavour = 2,
    GhostPunchthrough           = -1,
    GhostElectron               = -11,
    GhostPrimaryMuon            = -4,
    GhostMuonFromHeavyFlavour   = -3,
    GhostMuonFromLightFlavour   = -2
  };

  enum ExtendedMuonSimType { 
    ExtUnknown                        = 999, 
    ExtNotMatched                     = 0,
    ExtMatchedPunchthrough            = 1,
    ExtMatchedElectron                = 11,
    MatchedMuonFromGaugeOrHiggsBoson  = 10,
    MatchedMuonFromTau                = 9,
    MatchedMuonFromB                  = 8,
    MatchedMuonFromBtoC               = 7,
    MatchedMuonFromC                  = 6,
    MatchedMuonFromOtherLight         = 5,
    MatchedMuonFromPiKppMuX           = 4,
    MatchedMuonFromPiKNotppMuX        = 3,
    MatchedMuonFromNonPrimaryParticle = 2,
    ExtGhostPunchthrough              = -1,
    ExtGhostElectron                  = -11,
    GhostMuonFromGaugeOrHiggsBoson    = -10,
    GhostMuonFromTau                  = -9,
    GhostMuonFromB                    = -8,
    GhostMuonFromBtoC                 = -7,
    GhostMuonFromC                    = -6,
    GhostMuonFromOtherLight           = -5,
    GhostMuonFromPiKppMuX             = -4,
    GhostMuonFromPiKNotppMuX          = -3,
    GhostMuonFromNonPrimaryParticle   = -2
    
  };


  class MuonSimInfo {
  public:
    MuonSimInfo();
    typedef math::XYZPointD Point; ///< point in the space
    typedef math::XYZTLorentzVectorD LorentzVector; ///< Lorentz vector
    MuonSimType primaryClass; 
    ExtendedMuonSimType extendedClass;
    int flavour;
    int pdgId; // pdg ID of matching tracking particle
    int g4processType; // Geant process producing the particle
    int motherPdgId;
    int motherFlavour;
    int motherStatus;  // Status of the first gen particle 
    int grandMotherPdgId;
    int grandMotherFlavour;
    int heaviestMotherFlavour; 
    int tpId;
    int tpEvent;
    int tpBX;    // bunch crossing
    int charge;
    LorentzVector p4;
    Point vertex;
    Point motherVertex;
    float tpAssoQuality;
  };
}


#endif
