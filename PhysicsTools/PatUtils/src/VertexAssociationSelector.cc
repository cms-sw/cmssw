#include "PhysicsTools/PatUtils/interface/VertexAssociationSelector.h"
//#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
//#include "DataFormats/TrackReco/interface/Track.h"
#include <cmath>

pat::VertexAssociationSelector::VertexAssociationSelector(const Config & conf) : 
    conf_(conf) 
{
}

pat::VertexAssociation
pat::VertexAssociationSelector::simpleAssociation(const reco::Candidate &c, const reco::VertexRef &vtx) const {    
    pat::VertexAssociation ret(vtx);
    ret.setDistances(c.vertex(), vtx->position(), vtx->error());
    return ret;
}


bool
pat::VertexAssociationSelector::operator()(const reco::Candidate &c, const reco::Vertex &vtx) const {    
    using std::abs;
    using std::sqrt;

    if ((conf_.dZ > 0)      && !( std::abs(c.vz() - vtx.z())                > conf_.dZ    )) return false;
    if ((conf_.sigmasZ > 0) && !( std::abs(c.vz() - vtx.z())                > conf_.sigmasZ * vtx.zError())) return false;
    if ((conf_.dR > 0)      && !( (c.vertex() - vtx.position()).Rho() > conf_.dR    )) return false;
    if ( conf_.sigmasR > 0) {
        // D = sqrt( DZ^2 + DY^2) => sigma^2(D) = d D/d X_i * cov(X_i, X_j) * d D/d X_j =>
        // d D / d X_i = DX_i / D
        // D > sigmaR * sigma(D)   if and only if D^4 > sigmaR^2 * DX_i DX_j cov(X_i,X_j)
        AlgebraicVector3 dist(c.vx() - vtx.x(), c.vy() - vtx.y(), 0);
        double D2 = dist[0]*dist[0] + dist[1]*dist[1];
        double DcovD = ROOT::Math::Similarity(dist, vtx.error());
        if ( D2*D2 > DcovD * (conf_.sigmasR*conf_.sigmasR) ) return false;
    }
    /*
    if (conf_.sigmas3d > 0) {
        // same as above, but 3D
         AlgebraicVector3 dist(c.vx() - vtx.x(), c.vy() - vtx.y(), c.vz() - vtx.z());
        double D2 = dist[0]*dist[0] + dist[1]*dist[1] + dist[2]*dist[2];
        double DcovD = ROOT::Math::Similarity(dist, vtx.error());
        if ( D2*D2 > DcovD * (conf_.sigmas3d*conf_.sigmas3d) ) return false;
    }
    */

    return true;
}

bool
pat::VertexAssociationSelector::operator()(const pat::VertexAssociation &vass) const {    
    if (vass.isNull()) return false;
    if ((conf_.dZ > 0)       && ( vass.dz().value()         > conf_.dZ      )) return false;
    if ((conf_.sigmasZ > 0)  && ( vass.dz().significance()  > conf_.sigmasZ )) return false;
    if ((conf_.dZ > 0)       && ( vass.dr().value()         > conf_.dR      )) return false;
    if ((conf_.sigmasZ > 0)  && ( vass.dr().significance()  > conf_.sigmasR )) return false;
    // if ((conf_.sigmas3d > 0) && ( vass.signif3d()           > conf_.sigmas3d)) return false;
    return true;
}
