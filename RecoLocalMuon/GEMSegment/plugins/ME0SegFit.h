#ifndef GEMSegment_ME0SegFit_h
#define GEMSegment_ME0SegFit_h

// ME0SegFit.h - Segment fitting factored oout of ME0 segment builder based on
// CSCSegFit.h  - Segment fitting factored out of CSC segment builders - Tim Cox
// Last mod: 03.02.2015


/* This as an object which is initialized by a set of rechits (2 to 6) in a 
 * specific ME0 chamber and has the functionality to make a least squares fit 
 * to a straight line in 2-dim for those rechits.
 * The covariance matrix and chi2 of the fit are calculated.
 * The original code made use of CLHEP matrices but this version uses 
 * ROOT SMatrices because they are  multithreading compatible.
 * Because of this, the no. of rechits that can be handled is limited to
 * a maximum of 6, one per layer of a ME0 chamber. This means maximum dimensions
 * can be specified at compile time and hence satisfies SMatrix constraints.
 * This means that if at a later stage we would change the geometry to have
 * for instance 10 detection layers, we will have to modify this code too.
 * For 2 hits of course there is no fit - just draw a straight line between them.
 * Details of the algorithm are in the .cc file
 *
 */

#include <DataFormats/MuonDetId/interface/ME0DetId.h>   
#include <DataFormats/GEMRecHit/interface/ME0RecHit.h>
#include <Geometry/GEMGeometry/interface/ME0EtaPartition.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>

#include <vector>

class ME0SegFit {

public:

// TYPES

   typedef std::vector<const ME0RecHit*> ME0SetOfHits;
   
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
  ME0SegFit( std::map<uint32_t, const ME0EtaPartition*> me0etapartmap, ME0SetOfHits hits) : 
  me0etapartmap_( me0etapartmap ), hits_( hits ), scaleXError_( 1.0 ), refid_(me0etapartmap_.begin()->first ), fitdone_( false ) 
    {
      // --- LogDebug info about reading of ME0 Eta Partition map ------------------------------------------
      edm::LogVerbatim("ME0SegFit") << "[ME0SegFit::ctor] cached the me0etapartmap";

      // --- LogDebug for ME0 Eta Partition map ------------------------------------------------------------
      std::stringstream gemetapartmapss; gemetapartmapss<<"[ME0SegFit::ctor] :: me0etapartmap :: elements ["<<std::endl;
      for(std::map<uint32_t, const ME0EtaPartition*>::const_iterator mapIt = me0etapartmap_.begin(); mapIt != me0etapartmap_.end(); ++mapIt)
	{
	  gemetapartmapss<<"[ME0 DetId "<<mapIt->first<<" ="<<ME0DetId(mapIt->first)<<", ME0 EtaPart "<<mapIt->second<<"],"<<std::endl;
	}
      gemetapartmapss<<"]"<<std::endl;
      std::string gemetapartmapstr = gemetapartmapss.str();
      edm::LogVerbatim("ME0SegFit") << gemetapartmapstr;
      // --- End LogDebug -----------------------------------------------------------------------------------
    }

  virtual ~ME0SegFit() {}

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
  ME0SetOfHits hits(void) const { return hits_; }
  double scaleXError(void) const { return scaleXError_; }
  size_t nhits(void) const { return hits_.size(); }
  double chi2(void) const { return chi2_; }
  int ndof(void) const { return ndof_; }
  LocalPoint intercept() const { return intercept_;}
  LocalVector localdir() const { return localdir_;}
  const ME0EtaPartition* me0etapartition(uint32_t id) const { return me0etapartmap_.find(id)->second; }
  const ME0EtaPartition* refme0etapart() const { return me0etapartmap_.find(refid_)->second; }
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

  // const ME0Chamber* chamber_;  
  std::map<uint32_t, const ME0EtaPartition*> me0etapartmap_;

  ME0SetOfHits hits_;     //@@ FKA protoSegment
  float       uslope_;    //@@ FKA protoSlope_u
  float       vslope_;    //@@ FKA protoSlope_v
  LocalPoint  intercept_; //@@ FKA protoIntercept		
  LocalVector localdir_;  //@@ FKA protoDirection
  double      chi2_;      //@@ FKA protoChi2
  int         ndof_;      //@@ FKA protoNDF, which was double!!
  double      scaleXError_;
  uint32_t    refid_;
  bool        fitdone_;  
};
  
#endif
  
