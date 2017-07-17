#ifndef GEMCSCSegment_GEMCSCSegFit_h
#define GEMCSCSegment_GEMCSCSegFit_h

// GEMCSCSegFit.h extension to fit both CSCHits and GEMHits - Piet Verwilligen
// Created: 21.04.2015

/* Work with general Tracking Rechits, such that GEM and CSC rechits
 * can be treated at the same level. In case needed the Tracking Rechit can be
 * cast to the CSCRecHit2D or the GEMRecHit for access to more specific info.
 */
  
// CSCSegFit.h - Segment fitting factored out of CSC segment builders - Tim Cox
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
   
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/GEMGeometry/interface/GEMChamber.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>

#include <vector>

class GEMCSCSegFit {

public:

// TYPES

  // 16 x 16 Symmetric
  typedef ROOT::Math::SMatrix<double,16,16,ROOT::Math::MatRepSym<double,16> > SMatrixSym16;

  // 16 x 4
  typedef ROOT::Math::SMatrix<double,16,4 > SMatrix16by4;

  // 4 x 4 General + Symmetric
  typedef ROOT::Math::SMatrix<double, 4 > SMatrix4;
  typedef ROOT::Math::SMatrix<double,4,4,ROOT::Math::MatRepSym<double,4> > SMatrixSym4;

  // 2 x 2 Symmetric
  typedef ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > SMatrixSym2;

  // 4-dim vector
  typedef ROOT::Math::SVector<double,4> SVector4;


  // PUBLIC FUNCTIONS

  //@@ WANT OBJECT TO CACHE THE SET OF HITS SO CANNOT PASS BY REF
 GEMCSCSegFit(std::map<uint32_t, const CSCLayer*> csclayermap, std::map<uint32_t, const GEMEtaPartition*> gemrollmap, const std::vector<const TrackingRecHit*> hits) : 
  csclayermap_( csclayermap ), gemetapartmap_( gemrollmap ), hits_( hits ), scaleXError_( 1.0 ), refid_( csclayermap_.begin()->first ), fitdone_( false ) 
    {
      // --- LogDebug info about reading of CSC Layer map and GEM Eta Partition map -----------------------
      edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::ctor] cached the csclayermap and the gemrollmap";

      // --- LogDebug for CSC Layer map -------------------------------------------------------------------
      std::stringstream csclayermapss; csclayermapss<<"[GEMCSCSegFit::ctor] :: csclayermap :: elements ["<<std::endl;
      for(std::map<uint32_t, const CSCLayer*>::const_iterator mapIt = csclayermap_.begin(); mapIt != csclayermap_.end(); ++mapIt)
	{
	  csclayermapss<<"[CSC DetId "<<mapIt->first<<" ="<<CSCDetId(mapIt->first)<<", CSC Layer "<<mapIt->second<<" ="<<(mapIt->second)->id()<<"],"<<std::endl;
	}
      csclayermapss<<"]"<<std::endl;
      std::string csclayermapstr = csclayermapss.str();
      edm::LogVerbatim("GEMCSCSegFit") << csclayermapstr;
      // --- End LogDebug -----------------------------------------------------------------------------------

      // --- LogDebug for GEM Eta Partition map ------------------------------------------------------------
      std::stringstream gemetapartmapss; gemetapartmapss<<"[GEMCSCSegFit::ctor] :: gemetapartmap :: elements ["<<std::endl;
      for(std::map<uint32_t, const GEMEtaPartition*>::const_iterator mapIt = gemetapartmap_.begin(); mapIt != gemetapartmap_.end(); ++mapIt)
	{
	  gemetapartmapss<<"[GEM DetId "<<mapIt->first<<" ="<<GEMDetId(mapIt->first)<<", GEM EtaPart "<<mapIt->second<<"],"<<std::endl;
	}
      gemetapartmapss<<"]"<<std::endl;
      std::string gemetapartmapstr = gemetapartmapss.str();
      edm::LogVerbatim("GEMCSCSegFit") << gemetapartmapstr;
      // --- End LogDebug -----------------------------------------------------------------------------------
    } 

  virtual ~GEMCSCSegFit() {}

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
  std::vector<const TrackingRecHit*> hits(void) const { return hits_; }
  double scaleXError(void) const { return scaleXError_; }
  size_t nhits(void) const { return hits_.size(); }
  double chi2(void) const { return chi2_; }
  int ndof(void) const { return ndof_; }
  LocalPoint intercept() const { return intercept_;}
  LocalVector localdir() const { return localdir_;}

  const CSCChamber*      cscchamber     (uint32_t id) const {
    // Test whether id is found
    if(csclayermap_.find(id)==csclayermap_.end()) 
      { // id is not found
	throw cms::Exception("InvalidDetId") << "[GEMCSCSegFit] Failed to find CSCChamber in CSCLayerMap"<< std::endl; 
      } // chamber is not found and exception is thrown
    else 
      {  // id is found
	return (csclayermap_.find(id)->second)->chamber(); 
      } // chamber found and returned
  }
  const CSCLayer*        csclayer       (uint32_t id) const {
    if(csclayermap_.find(id)==csclayermap_.end()) { 
      throw cms::Exception("InvalidDetId") << "[GEMCSCSegFit] Failed to find CSCLayer in CSCLayerMap" << std::endl;
    }
    else { return csclayermap_.find(id)->second; }
  }
  const GEMEtaPartition* gemetapartition(uint32_t id) const { 
    if(gemetapartmap_.find(id)==gemetapartmap_.end()) { 
      throw cms::Exception("InvalidDetId") << "[GEMCSCSegFit] Failed to find GEMEtaPartition in GEMEtaPartMap" << std::endl;
    }
    else { return gemetapartmap_.find(id)->second; }
  }
  const CSCChamber*      refcscchamber  ()            const { 
    if(csclayermap_.find(refid_)==csclayermap_.end()) { 
      throw cms::Exception("InvalidDetId") << "[GEMCSCSegFit] Failed to find Reference CSCChamber in CSCLayerMap" << std::endl;
    }
    else { return (csclayermap_.find(refid_)->second)->chamber(); }
  }
  bool fitdone() const { return fitdone_; }
  
  private:  
  
  // PRIVATE FUNCTIONS

  void fit2(void);    // fit for 2 hits
  void fitlsq(void);  // least-squares fit for 3-6 hits  
  void setChi2(void); // fill chi2_ & ndof_ @@ FKA fillChiSquared()


 protected:

  // PROTECTED FUNCTIONS - derived class needs access

 // Set segment direction 'out' from IP
  void setOutFromIP(void); // fill localdir_  @@ FKA fillLocalDirection()

  SMatrix16by4 derivativeMatrix(void);
  SMatrixSym16 weightMatrix(void);
  AlgebraicSymMatrix flipErrors(const SMatrixSym4&);
  
  // PROTECTED MEMBER VARIABLES - derived class needs access

  std::map<uint32_t, const CSCLayer*>        csclayermap_;
  std::map<uint32_t, const GEMEtaPartition*> gemetapartmap_;
  const CSCChamber*                          refcscchamber_;


  std::vector<const TrackingRecHit*> hits_;      //@@ FKA protoSegment
  float                              uslope_;    //@@ FKA protoSlope_u
  float                              vslope_;    //@@ FKA protoSlope_v
  LocalPoint                         intercept_; //@@ FKA protoIntercept		
  LocalVector                        localdir_;  //@@ FKA protoDirection
  double                             chi2_;      //@@ FKA protoChi2
  int                                ndof_;      //@@ FKA protoNDF, which was double!!
  double                             scaleXError_;
  uint32_t                           refid_;
  bool                               fitdone_;  
  // order needs to be the same here when filled by the constructor

};
  
#endif
  
