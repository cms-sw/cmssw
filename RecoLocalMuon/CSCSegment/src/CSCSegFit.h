#ifndef CSCSegment_CSCSegFit_h
#define CSCSegment_CSCSegFit_h

// CSCSegFit.h  - Segment fitting factored out of CSC segment builders - Tim Cox
// Last mod: 03.02.2015

/* This as an object which is initialized by a set of rechits (2 to 6) in a 
 * specific CSC and has the functionality to make a least squares fit to a 
 * straight line in 2-dim for those rechits.
 * The covariance matrix and chi2 of the fit are calculated.
 * The original code made use of CLHEP matrices but this version uses 
 * ROOT SMatrices because they are  multithreading compatible.
 * Because of this, the no. of rechits that can be handled is limited to
 * a maximum of 6, one per layer of a CSC. This means maximum dimensions
 * can be specified at compile time and hence satisfies SMatrix constraints.
 * For 2 hits of course there is no fit - just draw a straight line between them.
 * Details of the algorithm are in the .cc file
 *
 */
   
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>

#include <vector>

class CSCSegFit {

public:

// TYPES

   typedef std::vector<const CSCRecHit2D*> CSCSetOfHits;
   
  // 12 x12 Symmetric
  typedef ROOT::Math::SMatrix<double,12,12,ROOT::Math::MatRepSym<double,12> > SMatrixSym12;

  // 12 x 4
  typedef ROOT::Math::SMatrix<double,12,4 > SMatrix12by4;

  // 4 x 4 General + Symmetric
  typedef ROOT::Math::SMatrix<double, 4 > SMatrix4;
  typedef ROOT::Math::SMatrix<double,4,4,ROOT::Math::MatRepSym<double,4> > SMatrixSym4;

  // 2 x 2 Symmetric
  typedef ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > SMatrixSym2;

  // 4-dim vector
  typedef ROOT::Math::SVector<double,4> SVector4;


  // PUBLIC FUNCTIONS

  //@@ WANT OBJECT TO CACHE THE SET OF HITS SO CANNOT PASS BY REF
  CSCSegFit( const CSCChamber* csc, CSCSetOfHits hits) : 
  chamber_( csc ), hits_( hits ), scaleXError_( 1.0 ), fitdone_( false ) {}

  virtual ~CSCSegFit() {}

  // Least-squares fit
  void fit( void ); // fill uslope_, vslope_, intercept_  @@ FKA fitSlopes()
  // Calculate covariance matrix of fitted parameters
  AlgebraicSymMatrix covarianceMatrix(void);

  // Change scale factor of rechit x error 
  // - expert use only!
  void setScaleXError ( double factor ) { scaleXError_ = factor; }

  // Fit values
  float xfit( float z ) const;
  float yfit( float z ) const;

  // Deviations from fit for given input (local w.r.t. chamber)
  float xdev( float x, float z ) const;
  float ydev ( float y, float z ) const;
  float Rdev( float x, float y, float z ) const;

  // Other public functions are accessors
  CSCSetOfHits hits(void) const { return hits_; }
  double scaleXError(void) const { return scaleXError_; }
  size_t nhits(void) const { return hits_.size(); }
  double chi2(void) const { return chi2_; }
  int ndof(void) const { return ndof_; }
  LocalPoint intercept() const { return intercept_;}
  LocalVector localdir() const { return localdir_;}
  const CSCChamber* chamber() const { return chamber_; }
  bool fitdone() const { return fitdone_; }
  
  private:  
  
  // PRIVATE FUNCTIONS

  void fit2(void); // fit for 2 hits
  void fitlsq(void); // least-squares fit for 3-6 hits  
  void setChi2(void); // fill chi2_ & ndof_ @@ FKA fillChiSquared()


 protected:

  // PROTECTED FUNCTIONS - derived class needs access

 // Set segment direction 'out' from IP
  void setOutFromIP(void); // fill localdir_  @@ FKA fillLocalDirection()

  SMatrix12by4 derivativeMatrix(void);
  SMatrixSym12 weightMatrix(void);
  AlgebraicSymMatrix flipErrors(const SMatrixSym4&);
  
  // PROTECTED MEMBER VARIABLES - derived class needs access

  const CSCChamber* chamber_;  
  CSCSetOfHits hits_;     //@@ FKA protoSegment
  float       uslope_;    //@@ FKA protoSlope_u
  float       vslope_;    //@@ FKA protoSlope_v
  LocalPoint  intercept_; //@@ FKA protoIntercept		
  LocalVector localdir_;  //@@ FKA protoDirection
  double      chi2_;      //@@ FKA protoChi2
  int         ndof_;      //@@ FKA protoNDF, which was double!!
  double      scaleXError_;
  bool        fitdone_;  
};
  
#endif
  
