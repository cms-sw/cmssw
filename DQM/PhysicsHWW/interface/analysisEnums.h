#ifndef WW_enums_h
#define WW_enums_h

#include "Math/LorentzVector.h"
#include "Rtypes.h"
#include <vector>
#include <vector>

typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> > LorentzVector;
typedef UInt_t wwcuts_t; // 32 bits only!
typedef std::pair<LorentzVector,unsigned int> JetPair;
typedef std::pair<bool, unsigned int> LeptonPair; // first bool(true-muon, false-electron) , second index

//
// Jets
//

enum WWJetType { CaloJet, jptJet, pfJet, TrkJet, GenJet };
enum jetregion { HCAL, HF, ALLJET};

//
// Leptons
//

enum EleFOTypes { EleFOV1, EleFOV2, EleFOV3, EleFOV4};
enum MuFOTypes { MuFOV1, MuFOV2 };

//
// Cuts
//

enum hyp_selection {  
  PASSED_BaseLine                = 1UL<<0,  
  PASSED_Charge                  = 1UL<<1,
  PASSED_ZVETO                   = 1UL<<2,
  PASSED_ZControlSampleVeryTight = 1UL<<3,   // within Z mass window +/- 5GeV
  PASSED_ZControlSampleTight     = 1UL<<4,   // within Z mass window +/- 10GeV
  PASSED_ZControlSampleLoose     = 1UL<<5,   // within Z mass window +/- 20GeV
  PASSED_MET                     = 1UL<<6,
  PASSED_LT_FINAL                = 1UL<<7,
  PASSED_LT_FO_MU1               = 1UL<<8,
  PASSED_LT_FO_MU2               = 1UL<<9,
  PASSED_LT_FO_ELEV1             = 1UL<<10,
  PASSED_LT_FO_ELEV2             = 1UL<<11,
  PASSED_LT_FO_ELEV3             = 1UL<<12,
  PASSED_LT_FO_ELEV4             = 1UL<<13,
  PASSED_LL_FINAL                = 1UL<<14,
  PASSED_LL_FO_MU1               = 1UL<<15,
  PASSED_LL_FO_MU2               = 1UL<<16,
  PASSED_LL_FO_ELEV1             = 1UL<<17,
  PASSED_LL_FO_ELEV2             = 1UL<<18,
  PASSED_LL_FO_ELEV3             = 1UL<<19,
  PASSED_LL_FO_ELEV4             = 1UL<<20,
  PASSED_JETVETO                 = 1UL<<21,  
  PASSED_TopControlSample        = 1UL<<22,  // 2 or more jets
  PASSED_1BJET                   = 1UL<<23,
  PASSED_SOFTMUVETO_NotInJets    = 1UL<<24,
  PASSED_SOFTMUVETO              = 1UL<<25,
  PASSED_EXTRALEPTONVETO         = 1UL<<26,  
  PASSED_TOPVETO_NotInJets       = 1UL<<27,  // exclude jets over threshold from top tagging
  PASSED_TOPVETO                 = 1UL<<28,  
  PASSED_Skim1                   = 1UL<<29,  // one fakable object + one final; full met
  PASSED_Trigger                 = 1UL<<30,  
  PASSED_Skim3                   = 1UL<<31   // one fakable object + one final
};

#endif
