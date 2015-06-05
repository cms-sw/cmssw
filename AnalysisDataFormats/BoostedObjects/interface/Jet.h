#ifndef ANALYSISDATAFORMATS_BOOSTEDOBJECTS_JET_HH
#define ANALYSISDATAFORMATS_BOOSTEDOBJECTS_JET_HH

#include "AnalysisDataFormats/BoostedObjects/interface/ResolvedVjj.h"

/*\ 
 * Simple Jet class to be used with B2G EDMNtuples
 * https://github.com/cmsb2g/B2GAnaFW
 */

namespace vlq{
  class Jet ;
  typedef std::vector<Jet> JetCollection ; 
  class Jet : public Candidate {
    friend class vlq::ResolvedVjj ; 
    public:
      Jet () : index_(-1), partonFlavour_(-1), hadronFlavour_(-1), csvDiscrimiator_(-1000), 
      isbtagged_(0), istoptagged_(0), ishtagged_(0), iswtagged_(0) {}
      ~Jet () {}

      int   getIndex () const { return index_ ; } 
      double getCSV () const { return csvDiscrimiator_ ; }
      bool  getIsbtagged () const { return isbtagged_ ; } 
      bool  getIstoptagged () const { return istoptagged_ ; }
      bool  getIshtagged () const { return ishtagged_ ; }
      bool  getIswtagged () const { return iswtagged_ ; } 

      void setIndex( const int& index) {index_ = index ; } 
      void setCSV( const double& csv) {csvDiscrimiator_ = csv ; } 
      void setIsbtagged( const bool& isbtagged) {isbtagged_ = isbtagged ; } 
      void setIstoptagged( const bool& istoptagged) {istoptagged_ = istoptagged ; }
      void setIshtagged( const bool& ishtagged) {ishtagged_ = ishtagged ; }
      void setIswtagged( const bool& iswtagged) {iswtagged_ = iswtagged ; } 

    protected:

      int index_ ; 
      int partonFlavour_ ;
      int hadronFlavour_ ; 
      float csvDiscrimiator_ ; 
      bool isbtagged_ ; 
      bool istoptagged_ ;
      bool ishtagged_ ;
      bool iswtagged_ ; 
  }; 
}

#endif 
