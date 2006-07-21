#ifndef Demo_PFClusterAlgo_PFRecHitFraction_h
#define Demo_PFClusterAlgo_PFRecHitFraction_h

#include "Math/GenVector/PositionVector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include <iostream>
#include <vector>

#include "DataFormats/PFReco/interface/PFRecHit.h"


/* class PFRecHit { */

/*  public: */
/*   typedef ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Double32_t> > REPPoint; */

/*   PFRecHit() :  */
/*     detId_(0),  */
/*     layer_(0),  */
/*     energy_(0),  */
/*     isSeed_(-1) {} */

/*   PFRecHit(unsigned detId, int layer,  */
/* 	   double energy,  */
/* 	   double posx, double posy, double posz,  */
/* 	   double axisx, double axisy, double axisz ); */

/*   PFRecHit(const PFRecHit& other); */

/*   virtual ~PFRecHit() {} */

/*   void      SetNeighbours( const std::vector<PFRecHit*>& neighbours ); */

/*   void      SetNWCorner( double posx, double posy, double posz ); */
/*   void      SetSWCorner( double posx, double posy, double posz ); */
/*   void      SetSECorner( double posx, double posy, double posz ); */
/*   void      SetNECorner( double posx, double posy, double posz ); */
  
/*   void      YouAreSeed(int seedstate=1) {isSeed_ = seedstate;}  */

/*   unsigned  GetDetId() const { return detId_; } */
/*   int       GetLayer() const { return layer_; } */
/*   double    GetEnergy() const { return energy_; } */

/*   const math::XYZPoint& GetPositionXYZ() const  */
/*     {return posxyz_; } */

/*   const REPPoint& GetPositionREP() const  */
/*     {return posrep_; } */

/*   const math::XYZVector& GetAxisXYZ() const  */
/*     {return axisxyz_;} */

/*   const std::vector< math::XYZPoint >& GetCornersXYZ() const  */
/*     {return cornersxyz_;} */

/*   const std::vector< REPPoint  >& GetCornersREP() const  */
/*     {return cornersrep_;} */

/*   const std::vector< PFRecHit* >& GetNeighbours() const  */
/*     {return neighbours_;}   */

/*   const std::vector< PFRecHit* >& GetNeighbours4() const  */
/*     {return neighbours4_;}   */

/*   const std::vector< PFRecHit* >& GetNeighbours8() const  */
/*     {return neighbours8_;}   */

/*   const int IsSeed() const {return isSeed_;}  */


/*   friend    std::ostream& operator<<(std::ostream& out, const PFRecHit& hit); */

/*  private: */

/*   /// cell detid */
/*   unsigned      detId_;              */

/*   /// cell layer, see PFClusterLayer.h */
/*   int           layer_;           */

/*   /// rechit energy */
/*   double        energy_; */

/*   /// is this a seed ? (-1:unknown, 0:no, 1 yes) */
/*   int           isSeed_; */

/*   /// rechit position: cartesian */
/*   math::XYZPoint      posxyz_; */

/*   /// rechit position: rho, eta, phi */
/*   REPPoint            posrep_; */

/*   /// rechit cell axis */
/*   math::XYZVector     axisxyz_;   */

/*   /// 4 corners rep (for display)  */
/*   std::vector< REPPoint >  cornersrep_; */

/*   /// 4 corners xyz (for display)  */
/*   std::vector< math::XYZPoint >  cornersxyz_; */

/*   /// pointers to neighbours (if null: no neighbour here) */
/*   std::vector<PFRecHit*>   neighbours_; */
  
/*   /// pointers to existing neighbours (1 common side) */
/*   std::vector<PFRecHit*>   neighbours4_; */

/*   /// pointers to existing neighbours (1 common side or diagonal) */
/*   std::vector<PFRecHit*>   neighbours8_; */
 
/*   /// number of neighbours */
/*   static const unsigned    nNeighbours_; */
 
/*   /// number of corners */
/*   static const unsigned    nCorners_; */
 
/*   void      SetCorner( unsigned i, double posx, double posy, double posz ); */
/* }; */



namespace reco {
  
  class PFRecHitFraction {
  public:
    
    PFRecHitFraction() : recHit_(0), fraction_(-1), distance_(0) {}
    
    PFRecHitFraction(const reco::PFRecHit* rechit, double fraction, double dist) 
      : recHit_(rechit), fraction_(fraction), distance_(dist) {}
    
    PFRecHitFraction(const reco::PFRecHit* rechit, double fraction) 
      : recHit_(rechit), fraction_(fraction), distance_(0) {}
    
    PFRecHitFraction(const PFRecHitFraction& other) 
      : recHit_(other.recHit_), fraction_(other.fraction_), distance_(other.distance_) {}
    
    const reco::PFRecHit* GetRecHit() const {return recHit_;} 
    
    /// sets distance to cluster
    void   SetDistToCluster(double dist) { distance_ = dist;}
    
    /// \return energy fraction
    double GetFraction() const {return fraction_;}
    
    /// \return recHit_->GetEnergy() * fraction_
    double GetEnergy() const 
      { return recHit_->GetEnergy() * fraction_;}
    
    /// \return distance to cluster
    double GetDistToCluster() const {return distance_;}
    
    friend    std::ostream& operator<<(std::ostream& out,
				       const PFRecHitFraction& hit);
    
  private:
    
    /// corresponding rechit (not owner)
    const reco::PFRecHit* recHit_;
    
    /// fraction of the rechit energy owned by the cluster
    double    fraction_;
    
    /// distance to the cluster
    double    distance_;
    
  };
}



#endif
