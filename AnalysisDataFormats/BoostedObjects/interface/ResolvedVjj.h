#ifndef ANALYSISDATAFORMATS_BOOSTEDOBJECTS_RESOLVEDVJJ_HH
#define ANALYSISDATAFORMATS_BOOSTEDOBJECTS_RESOLVEDVJJ_HH

#include <utility>

#include "AnalysisDataFormats/BoostedObjects/interface/Candidate.h"

/*\
 * Simple class for bosons W, H, Z decaying to to quarks
 * To be used with B2G EDMNtuples
 * https://github.com/cmsb2g/B2GAnaFW
 */

namespace vlq { 
  class ResolvedVjj ; 
  typedef std::vector<ResolvedVjj> ResolvedVjjCollection ; 
  class ResolvedVjj : public Candidate {
    public:
      ResolvedVjj () : drjj_(-1), mass_(-1), massdrop_(-1), scaledmassdrop_(-1), jetindex1_(-1), jetindex2_(-1) {}
      ~ResolvedVjj () {} 

      float getScaledMassDrop () const { return scaledmassdrop_ ; }
      std::pair<int,int>  getJetIndices () const { return std::pair<int, int>(jetindex1_, jetindex2_) ; }

      void setDrjj ( const float& drjj ) { drjj_ = drjj ; }
      void setMass ( const float& mass ) { mass_ = mass ; }
      void setMassDrop ( const float& massdrop ) { massdrop_ = massdrop ; }
      void setScaledMassDrop ( const float& scaledmassdrop ) { scaledmassdrop_ = scaledmassdrop ; } 
      void setJetIndices ( const std::pair<int,int>& jetindices ) { jetindex1_ = jetindices.first; jetindex2_ = jetindices.second ; } 

    protected:

    private:

      float drjj_ ; 
      float mass_ ; 
      float massdrop_ ;
      float scaledmassdrop_ ;
      int jetindex1_ ; 
      int jetindex2_ ;

  }; 
}

#endif
