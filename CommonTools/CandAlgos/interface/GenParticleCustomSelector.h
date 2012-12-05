#ifndef RecoSelectors_GenParticleCustomSelector_h
#define RecoSelectors_GenParticleCustomSelector_h
/* \class GenParticleCustomSelector
 *
 * \author Giuseppe Cerati, UCSD
 *
 */

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class GenParticleCustomSelector {

public:
  GenParticleCustomSelector(){}
  GenParticleCustomSelector ( double ptMin,double minRapidity,double maxRapidity,
			double tip,double lip, bool chargedOnly, int status,
			std::vector<int> pdgId = std::vector<int>()) :
    ptMin_( ptMin ), minRapidity_( minRapidity ), maxRapidity_( maxRapidity ),
    tip_( tip ), lip_( lip ), chargedOnly_(chargedOnly), status_(status), pdgId_( pdgId ) { }
  
  /// Operator() performs the selection: e.g. if (tPSelector(tp)) {...}
  bool operator()( const reco::GenParticle & tp ) const { 

    if (chargedOnly_ && tp.charge()==0) return false;//select only if charge!=0
    bool testId = false;
    unsigned int idSize = pdgId_.size();
    if (idSize==0) testId = true;
    else for (unsigned int it=0;it!=idSize;++it){
      if (tp.pdgId()==pdgId_[it]) testId = true;
    }

    return (
	    tp.pt() >= ptMin_ && 
	    tp.eta() >= minRapidity_ && tp.eta() <= maxRapidity_ && 
	    sqrt(tp.vertex().perp2()) <= tip_ &&
	    fabs(tp.vertex().z()) <= lip_ &&
	    tp.status() == status_ &&
	    testId 
	    );
  }
  
private:
  double ptMin_;
  double minRapidity_;
  double maxRapidity_;
  double tip_;
  double lip_;
  bool chargedOnly_;
  int status_;
  std::vector<int> pdgId_;

};

#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<GenParticleCustomSelector> {
      static GenParticleCustomSelector make( const edm::ParameterSet & cfg ) {
	return GenParticleCustomSelector(    
 	  cfg.getParameter<double>( "ptMin" ),
	  cfg.getParameter<double>( "minRapidity" ),
	  cfg.getParameter<double>( "maxRapidity" ),
	  cfg.getParameter<double>( "tip" ),
	  cfg.getParameter<double>( "lip" ),
	  cfg.getParameter<bool>( "chargedOnly" ),
	  cfg.getParameter<int>( "status" ),
	  cfg.getParameter<std::vector<int> >( "pdgId" )); 
      }
    };
    
  }
}

#endif
