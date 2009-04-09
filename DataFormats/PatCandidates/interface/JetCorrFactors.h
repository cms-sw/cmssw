//
// $Id: JetCorrFactors.h,v 1.5 2009/03/26 20:04:10 rwolf Exp $
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
  \version  $Id: JetCorrFactors.h,v 1.5 2009/03/26 20:04:10 rwolf Exp $
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
	    /// check if all are different from -1 (default), or 0
	    bool operator!=(const float f) const {return (f!=uds && f!=g && f!=c && f!=b);}   
          };

	  /// define a single enum to point to a step in the sequence of corrections
          /// that is to say:
	  ///  * "up to L4",
          ///  * "up to L6, and the choice of flavour for L5 was 'uds'"
          ///  * "up to L7, and the choice of flavour for L5 and L7 was 'b'"
          enum CorrStep { Raw = 0x0,   L1    = 0x10, L2  = 0x20,  L3 = 0x30, L4 = 0x40,
                          L5g = 0x50,  L5uds = 0x51, L5c = 0x54, L5b = 0x55,
                          L6g = 0x60,  L6uds = 0x61, L6c = 0x64, L6b = 0x65,
                          L7g = 0x70,  L7uds = 0x71, L7c = 0x74, L7b = 0x75 
	                };

   	  /// default Constructor
          JetCorrFactors();
	  /// constructor by value
          JetCorrFactors(std::string &label, float l1, float l2, float l3, float l4, FlavourCorrections l5, FlavourCorrections l6, FlavourCorrections l7);

          /// default scale factor: Raw & L1 & L2 & L3
          float scaleDefault() const { return correction(L3); };
          /// returns the correction for a jet up to a given step, starting from another step.
          float correction(CorrStep step, CorrStep begin=Raw) const ;           
          /// convert a string into a CorrStep
          static CorrStep const corrStep(const std::string &step, const std::string &flavour="");  
          /// convert a CorrStep into a string
          std::string corrStep(CorrStep step) const;
          /// return flavour string
          std::string flavour (CorrStep step) const; 
	  /// return label, i.e. the identifying name of this set of correction factors
	  std::string getLabel() const { return label_; };
	  /// clear label to save storage, if only one set of correction factors is used
	  void clearLabel() { label_.clear(); };
	  /// print function for debugging
	  void print() const;

      private:
          /// vector to hold flavour independent corrections
	  /// for L1,L2,L3,L4,L6 (in this order); if some 
	  /// corrections are not available, it can be shorter.
          std::vector<float> flavourIndepCorrections_;

          /// vector to hold flavour dependent corrections (L5, 
	  /// L7 in this order); if some are not available, it 
	  /// can be shorter (or even empty)
          std::vector<FlavourCorrections>  flavourDepCorrections_;

	  /// label for this set of jet correction factors;
	  /// different sets are distinguished by this string;
	  /// if only one set is attached to each jet, than 
	  /// this string is empty to save storage
	  std::string label_;

          /// convert a CorrStep into a number 0-7
          static inline size_t istep(const CorrStep &cstep ) { return (static_cast<size_t>(cstep)) >> 4; }
          /// convert a CorrStep into a flavour code 0-5
          static inline size_t iflav(const CorrStep &cstep ) { return (static_cast<size_t>(cstep)) & 0xF; }
          /// get a flavour correction out of a CorrStep
          static float getFlavorCorrection(const FlavourCorrections &, const size_t& flav) ;
  };
}

#endif
