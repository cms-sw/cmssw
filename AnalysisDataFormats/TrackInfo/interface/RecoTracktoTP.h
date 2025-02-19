#ifndef TRACKINFO_RECOTRACKTOTP_H
#define TRACKINFO_RECOTRACKTOTP_H

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


class RecoTracktoTP 
{
    public:
  
    RecoTracktoTP();
    ~RecoTracktoTP();
  
    void SetTrackingParticle(TrackingParticleRef tp)    {trackingParticle_ = tp;}
    void SetRecoTrack(reco::TrackBaseRef track)         {recoTrack = track;}
    void SetRecoVertex(reco::VertexRef vertex)          {recoVertex = vertex;}
    void SetBeamSpot(math::XYZPoint bs)                 {beamSpot_ = bs;}
    void SetShared(const float &m)                      {shared_= m;}
  
    // Interogation Functions
    reco::Track       RT()          const {return recoTrack.isNonnull()  ? *recoTrack : reco::Track();}
    TrackingParticle  TP()          const {return trackingParticle_.isNonnull() ? *trackingParticle_ : TrackingParticle();}
    reco::Vertex      RV()          const {return recoVertex.isNonnull() ? *recoVertex : reco::Vertex();} 
    math::XYZPoint    BeamSpot()    const {return beamSpot_;}
  
    bool          matched()         const {return trackingParticle_.isNonnull() && recoTrack.isNonnull();} 
    bool          hasRV()           const {return recoVertex.isNonnull() && fabs(recoVertex->position().Mag2())>0.0;}   // position is ROOT::MATH::Cartesian3D<double> 
    bool          hasPCA()          const {return s_pca().mag()<9999.0;}
    bool          allmatched()      const {return matched() && hasRV();} 
  
    // These members for reco d0 and dz are the TIP and LIP w.r.t the reconstructed BeamSpot (as opposed to (0,0,0) ).
    double        r_dxy()           const {return RT().dxy( BeamSpot() );}
    double        r_dsz()           const {return RT().dsz( BeamSpot() );}
    double        r_d0()            const {return -1.0 * r_dxy();}
    double        r_dz()            const {return RT().dz( BeamSpot() );}
    
    // These members for reco d0 and dz are the TIP and LIP w.r.t the reconstructed vertex (as opposed to (0,0,0) ).
    double        r_d02()           const {return -1.0 * RT().dxy( RV().position() );}
    double        r_dz2()           const {return RT().dz( RV().position() );}
    
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
    float           GetShared()     const {return shared_;}
  
    // Short cut methods to get TP truth info
    TrackingParticle      TPMother(unsigned short i) const;
    TrackingParticle      TPMother()            const {return numTPMothers()==1 ? TPMother(0) : TrackingParticle();}
    int                   numTPSourceTracks()   const {return TP().parentVertex()->nSourceTracks();}
    int                   numTPMothers() const;
    bool                  hasTPMother()         const {return numTPMothers()>0;}

    protected:
  
    reco::TrackBaseRef recoTrack;
    reco::VertexRef recoVertex;
  
    TrackingParticleRef trackingParticle_;
  
    GlobalVector      simMomPCA_;       // Momentum at point of closest approach to the beamspot of the trackingParticle.
    GlobalPoint       simPCA_;          // Point of closest approach to the BeamSpot of the TrackingParticle.
    math::XYZPoint    beamSpot_;        // I use type XYZPoint to faciliate the use of the recoTrack memeber dxy(XYZPoint).
    float             shared_;          // Number of shared hits with TP

};


#endif // TRACKINFO_RECOTRACKTOTP_H

