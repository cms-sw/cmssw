//
// $Id: JetCorrFactors.h,v 1.1 2008/03/07 18:52:54 lowette Exp $
//

#ifndef DataFormats_PatCandidates_JetCorrFactors_h
#define DataFormats_PatCandidates_JetCorrFactors_h

/**
  \class    pat::JetCorrFactors JetCorrFactors.h "DataFormats/PatCandidates/interface/JetCorrFactors.h"
  \brief    Class for storage of jet correction factors

   JetCorrFactors implements a class, basically a struct, that contains
   all possible useful jet correction factors. These are then poduced at PAT
   Layer-0 as assoiacted objects, and collpased back in the pat::Jet class at
   PAT Layer-1.

  \author   Giovanni Petrucciani
  \version  $Id: JetCorrFactors.h,v 1.1 2008/03/07 18:52:54 lowette Exp $
*/

#include <vector>
#include <string>
#include <math.h>

namespace pat {


  class JetCorrFactors {
      public:
          /// define a simple struct for flavour dependent corrections
          struct FlavourCorrections { 
              FlavourCorrections() :  
                  uds(-1), g(-1), c(-1), b(-1) {}
              FlavourCorrections(float corr_uds, float corr_g, float corr_c, float corr_b) :  
                  uds(corr_uds), g(corr_g), c(corr_c), b(corr_b) {}
              float uds, g, c, b; 
          };

          /** Define a single enum to point a step in the sequence of corrections
           *  that is to say "up to L4",
           *  or "up to L6, and the choice of flavour for L5 was 'uds'"
           *  or "up to L7, and the choice of flavour for L5 and L7 was 'b' " */
          enum CorrStep { Raw = 0x0,   L1    = 0x10, L2 = 0x20,  L3 = 0x30, L4 = 0x40,
                          L5g = 0x50,  L5uds = 0x51, L5c = 0x54, L5b = 0x55,
                          L6g = 0x60,  L6uds = 0x61, L6c = 0x64, L6b = 0x65,
                          L7g = 0x70,  L7uds = 0x71, L7c = 0x74, L7b = 0x75 };

   	  /// Default Constructor
          JetCorrFactors();
          JetCorrFactors(float l1, float l2, float l3, float l4, FlavourCorrections l5, float l6, FlavourCorrections l7);

          /// Default Scale Factor: Raw & L1 & L2 & L3 (ignore -1's)
          float scaleDefault() const { return fabs(correction(L3)); };
	 
          /// Returns True if a specific correction is available, false otherwise
          bool  hasCorrection(CorrStep step) const;
          /// Returns True if a specific correction is available, false otherwise
          bool  hasCorrection(const size_t step, const size_t flavour) const {return hasCorrection((CorrStep)(step<<4|flavour));};
            
          /// Returns the correction for a jet up to a given step, starting from another step.
          /// It will return -1.0 if either the start or the end step are not available
          float correction(CorrStep step, CorrStep begin=Raw) const ;
          /// Convert a string into a CorrStep
          static CorrStep const corrStep(const std::string &step, const std::string &flavour="");  
          /// Convert a CorrStep into a string
          std::string corrStep(CorrStep step) const;
          /// Return Flavour 
          std::string flavour (CorrStep step) const; 

      private:
          // one vector to hold flavour independent corrections
          // for L1,L2,L3,L4,L6 (in this order).
          // if some corrections are not available, it can be shorter.
          std::vector<float> flavourIndepCorrections_;

          // one vector to hold flavour dependent corrections (L5, L7 in this order);
          // if some are not available, it can be shorter (or even empty)
          std::vector<FlavourCorrections>  flavourDepCorrections_;

          /// Convert a CorrStep into a number 0-7
          static inline size_t istep(const CorrStep &cstep ) { return (static_cast<size_t>(cstep)) >> 4; }
          /// Convert a CorrStep into a flavour code 0-5
          static inline size_t iflav(const CorrStep &cstep ) { return (static_cast<size_t>(cstep)) & 0xF; }
          /// Get a flavour correction out of a CorrStep
          static float getFlavCorr(const FlavourCorrections &, const size_t & flavcode ) ;
          /// Get a correction factor for just one step, relative to the previous one.
        

  };


}

#endif
