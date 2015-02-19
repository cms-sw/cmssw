#ifndef CSCSegment_CSCCondSegFit_h
#define CSCSegment_CSCCondSegFit_h

// CSCCondSegFit.h -- segment fit factored out of CSCSegAlgoST - Tim Cox
// Last mod: 29.01.2015

/* This class extends basic CSCSegFit with the complexities built into
 * the original entangled CSCSegAlgoST extension of the CSCSegAlgoSK fit.
 * i.e. the uncertainties on the rechit positions can be adjusted 
 * according to external conditions in various complex ways, according to 
 * the requirements of the CSC segment-building algorithm CSCSegAlgoST.
 */

#include <RecoLocalMuon/CSCSegment/src/CSCSegFit.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

class CSCCondSegFit : public CSCSegFit {

public:

  CSCCondSegFit( const edm::ParameterSet& ps, const CSCChamber* csc, const CSCSetOfHits& hits) : 
    CSCSegFit( csc, hits ),
    worstHit_( 0 ),
    chi2Norm_ ( ps.getParameter<double>("NormChi2Cut2D") ),
    condSeed1_ ( ps.getParameter<double>("SeedSmall") ),
    condSeed2_ ( ps.getParameter<double>("SeedBig") ),
    covToAnyNumber_ ( ps.getParameter<bool>("ForceCovariance") ),
    covToAnyNumberAll_ ( ps.getParameter<bool>("ForceCovarianceAll") ),
    covAnyNumber_ ( ps.getParameter<double>("Covariance") ) {}

  ~CSCCondSegFit() {}

  // The fit - override base class version with this version
  // which passes in bool flags for up to two extra conditioning passes
  void fit( bool condpass1 = false, bool condpass2 = false ); // fill uslope_, vslope_, intercept_

  int worstHit( void ) { return worstHit_; }
  
  private:  
  
  // Rest can all be private since we don't plan on more derived classes

  // PRIVATE MEMBER FUNCTIONS
  
  // Calculate chi2 - override base class version with this version
  // which passes in bool flags for up to two extra conditioning passes
  void setChi2( bool condpass1, bool condpass2 ); // fill chi2_ & ndof_
  void correctTheCovMatrix(CSCSegFit::SMatrixSym2& IC);
  void correctTheCovX(void);
  
  
  // EXTRA MEMBER VARIABLES 

  int worstHit_;  //@@ FKA maxContrIndex
 
  // Parameters related to adjustment for numerical robustness
  std::vector<double> lex_;  //@@ FKA e_Cxx; LOCAL ERROR x COMPONENT FOR EACH HIT
  double chi2Norm_;          //@@ FKA chi2Norm_2D_

  // PSet values that might reasonably be accessed directly 
  // since used ONLY in correctTheCovMatrix:
  //@@ the comments on following parameters don't help me understand them
  double condSeed1_, condSeed2_; /// The correction parameters
  bool covToAnyNumber_;          /// Allow to use any number for covariance (by hand)
  bool covToAnyNumberAll_;       /// Allow to use any number for covariance for all RecHits
  double covAnyNumber_;          /// The number to force the Covariance

};
  
#endif
  
