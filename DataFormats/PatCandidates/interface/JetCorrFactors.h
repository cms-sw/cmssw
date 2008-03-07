//
// $Id$
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

  \author   Steven Lowette
  \version  $Id$
*/


namespace pat {


  class JetCorrFactors {

    public:

      JetCorrFactors();
      JetCorrFactors(float scaleDefault, float scaleUds, float scaleGlu, float scaleC, float scaleB);
      virtual ~JetCorrFactors() {}

      float scaleDefault() const;
      float scaleUds() const;
      float scaleGlu() const;
      float scaleC() const;
      float scaleB() const;

    private:

      float scaleDefault_;
      float scaleUds_;
      float scaleGlu_;
      float scaleC_;
      float scaleB_;

  };


}

#endif
