#ifndef TRACKINFO_TPTORECOTRACK_H
#define TRACKINFO_TPTORECOTRACK_H

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "TMath.h"
//#include <vector>


class TPtoRecoTrack 
{
    public:
  
    TPtoRecoTrack();
    ~TPtoRecoTrack();
  
    void SetTrackingParticle(TrackingParticleRef tp)  {trackingParticle_ = tp;}
  
    void SetRecoTrack_AlgoA(reco::TrackBaseRef track)     {recoTrack_AlgoA_ = track;}
    void SetRecoTrack_AlgoB(reco::TrackBaseRef track)     {recoTrack_AlgoB_ = track;}
    
    void SetShared_AlgoA(const float &mA) {sharedA_= mA;}
    void SetShared_AlgoB(const float &mB) {sharedB_= mB;}
  
    void SetRecoVertex_AlgoA(reco::VertexRef vertex)  {recoVertex_AlgoA_ = vertex;}
    void SetRecoVertex_AlgoB(reco::VertexRef vertex)  {recoVertex_AlgoB_ = vertex;}

    void SetBeamSpot(math::XYZPoint bs)           {beamSpot_ = bs;}
  
    // Interogation Functions
    reco::Track       RTA()         const {return recoTrack_AlgoA_.isNonnull()  ? *recoTrack_AlgoA_  : reco::Track();}
    reco::Track       RTB()         const {return recoTrack_AlgoB_.isNonnull()  ? *recoTrack_AlgoB_  : reco::Track();}
    TrackingParticle  TP()          const {return trackingParticle_.isNonnull() ? *trackingParticle_ : TrackingParticle();}
    reco::Vertex      RVA()         const {return recoVertex_AlgoA_.isNonnull() ? *recoVertex_AlgoA_ : reco::Vertex();} 
    reco::Vertex      RVB()         const {return recoVertex_AlgoB_.isNonnull() ? *recoVertex_AlgoB_ : reco::Vertex();}
    math::XYZPoint    BeamSpot()    const {return beamSpot_;}
  
    bool          matched()         const {return matchedA() && matchedB();}
    bool          matchedA()        const {return trackingParticle_.isNonnull() && recoTrack_AlgoA_.isNonnull();} 
    bool          matchedB()        const {return trackingParticle_.isNonnull() && recoTrack_AlgoB_.isNonnull();}
    bool          matchedAnotB()    const {return matchedA() && !matchedB();}
    bool          matchedBnotA()    const {return matchedB() && !matchedA();}
    bool          hasRVA()          const {return recoVertex_AlgoA_.isNonnull() && fabs(recoVertex_AlgoA_->position().Mag2())>0.0;}   // position is ROOT::MATH::Cartesian3D<double> 
    bool          hasRVB()          const {return recoVertex_AlgoB_.isNonnull() && fabs(recoVertex_AlgoB_->position().Mag2())>0.0;}   // position is ROOT::MATH::Cartesian3D<double>
    bool          hasRV()           const {return hasRVA() && hasRVB();}
    bool          hasPCA()          const {return s_pca().mag()<9999.0;}
    bool          allmatchedA()     const {return matchedA() && hasRVA();} 
    bool          allmatchedB()     const {return matchedB() && hasRVA();} 
    bool          allmatched()      const {return matched() && hasRV();} 
    float         GetSharedA()      const {return sharedA_;}
    float         GetSharedB()      const {return sharedB_;}
  
    // These members for reco d0 and dz are the TIP and LIP w.r.t the reconstructed BeamSpot (as opposed to (0,0,0) ).
    double        rA_dxy()          const {return RTA().dxy( BeamSpot() );}
    double        rB_dxy()          const {return RTB().dxy( BeamSpot() );}
    double        rA_dsz()          const {return RTA().dsz( BeamSpot() );}
    double        rB_dsz()          const {return RTB().dsz( BeamSpot() );}
    double        rA_d0()           const {return -1.0 * rA_dxy();}
    double        rB_d0()           const {return -1.0 * rB_dxy();}
    double        rA_dz()           const {return RTA().dz( BeamSpot() );}
    double        rB_dz()           const {return RTB().dz( BeamSpot() );}
    
    // These members for reco d0 and dz are the TIP and LIP w.r.t the reconstructed vertex (as opposed to (0,0,0) ).
    double        rA_d02()          const {return -1.0 * RTA().dxy( RVA().position() );}
    double        rA_dz2()          const {return RTA().dz( RVA().position() );}
    double        rB_d02()          const {return -1.0 * RTB().dxy( RVB().position() );}
    double        rB_dz2()          const {return RTB().dz( RVB().position());}
    
    // These members for sim d0 and dz are not included in the TrackingParticle class and must be included seperately.
    void SetTrackingParticleMomentumPCA(const GlobalVector &p)    {simMomPCA_ = p;} 
    void SetTrackingParticlePCA(const GlobalPoint &v)             {simPCA_ = v;} 
  
    GlobalVector    s_p()           const {return simMomPCA_;}
    GlobalPoint     s_pca()         const {return simPCA_;}
    GlobalPoint     s_v()           const {return GlobalPoint(s_pca().x()-BeamSpot().x(), s_pca().y()-BeamSpot().y(), s_pca().z()-BeamSpot().z() );}
  
    double          s_qoverp()      const {return TP().charge() / s_p().mag();}
    double          s_theta()       const {return s_p().theta();}
    double          s_lambda()      const {return M_PI/2-s_p().theta();}
    double          s_phi()         const {return s_p().phi();}
    double          s_eta()         const {return -1.0*log( tan (0.5*s_p().theta()) );}
  
    double          s_dxy()         const {return ( - s_v().x() * s_p().y() + s_v().y() * s_p().x() ) / s_p().perp();}
    double          s_dsz()         const {return s_v().z()*s_p().perp()/ s_p().mag() - ((s_v().x()*s_p().x() + s_v().y()*s_p().y()) / s_p().perp()) * s_p().z()/s_p().mag();}
    double          s_d0()          const {return -1.0*s_dxy();}
    double          s_dz()          const {return s_v().z() - ( s_v().x() * s_p().x() + s_v().y() * s_p().y() )/s_p().perp() * s_p().z()/s_p().perp();}  
  
    // Short cut methods to get TP truth info
    TrackingParticle      TPMother(unsigned short i) const;
    TrackingParticle      TPMother()            const {return numTPMothers()==1 ? TPMother(0) : TrackingParticle();}
    int                   numTPSourceTracks()   const {return TP().parentVertex()->nSourceTracks();}
    int                   numTPMothers() const;
    bool                  hasTPMother()         const {return numTPMothers()>0;}

    protected:
  
    reco::TrackBaseRef recoTrack_AlgoA_;
    reco::VertexRef recoVertex_AlgoA_;
  
    reco::TrackBaseRef recoTrack_AlgoB_;
    reco::VertexRef recoVertex_AlgoB_;
  
    TrackingParticleRef trackingParticle_;
  
    GlobalVector      simMomPCA_;       // Momentum at point of closest approach to the beamspot of the trackingParticle.
    GlobalPoint       simPCA_;          // Point of closest approach to the BeamSpot of the TrackingParticle.
    math::XYZPoint    beamSpot_;        // I use type XYZPoint to faciliate the use of the recoTrack memeber dxy(XYZPoint).
    float             sharedA_;          // Number of shared hits with track A
    float             sharedB_;          // Number of shared hits with track B

};


#endif // TRACKINFO_TPTORECOTRACK_H
        
