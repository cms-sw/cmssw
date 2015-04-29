#ifndef TrackReco_TrackExtra_h
#define TrackReco_TrackExtra_h
/** \class reco::TrackExtra TrackExtra.h DataFormats/TrackReco/interface/TrackExtra.h
 *
 * Additional information about a reconstructed track. It is stored in RECO and supplements
 * the basic information stored in the Track class that is stored on AOD only.
 * If you wish to use information in the TrackExtra class, you should
 * access it via the reference supplied in the Track class.
 *
 * \author Luca Lista, INFN
 *
 *
 */
#include <Rtypes.h>
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/TrackReco/interface/TrackExtraBase.h"
#include "DataFormats/TrackReco/interface/TrackResiduals.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace reco
{
class TrackExtra : public TrackExtraBase
{
public:
    /// tracker parameter dimension
    enum { dimension = 5 };
    /// track error matrix size
    enum { covarianceSize = dimension * (dimension + 1) / 2 };
    /// point in the space
    typedef math::XYZPoint Point;
    /// spatial vector
    typedef math::XYZVector Vector;
    /// 5 parameter covariance matrix
    typedef math::Error<5>::type CovarianceMatrix;
    /// index type
    typedef unsigned int index;

    /// default constructor
    TrackExtra():
        outerMomentum_(),
        outerOk_(false),
        outerDetId_(0),
        innerPosition_(),
        innerMomentum_(),
        innerOk_(false),
        innerDetId_(0),
        seedDir_(anyDirection),
        seedRef_(),
        trackResiduals_() {
        index idx = 0;
        for (index i = 0; i < dimension; ++ i) {
            for (index j = 0; j <= i; ++ j) {
                outerCovariance_[idx] = 0;
                innerCovariance_[idx] = 0;
                ++idx;
            }
        }
    }

    /// constructor from outermost/innermost position and momentum and Seed information
    TrackExtra(const Point & outerPosition, const Vector & outerMomentum, bool ok ,
               const Point & innerPosition, const Vector & innerMomentum, bool iok,
               const CovarianceMatrix& outerState, unsigned int outerId,
               const CovarianceMatrix& innerState, unsigned int innerId,
               PropagationDirection seedDir,
               edm::RefToBase<TrajectorySeed> seedRef = edm::RefToBase<TrajectorySeed>());

    /// outermost hit position
    const Point &outerPosition() const {
        return outerPosition_;
    }
    /// momentum vector at outermost hit position
    const Vector &outerMomentum() const {
        return outerMomentum_;
    }
    /// returns true if the outermost hit is valid
    bool outerOk() const {
        return outerOk_;
    }
    /// innermost hit position
    const Point &innerPosition() const {
        return innerPosition_;
    }
    /// momentum vector at innermost hit position
    const Vector &innerMomentum() const {
        return innerMomentum_;
    }
    /// returns true if the innermost hit is valid
    bool innerOk() const {
        return innerOk_;
    }
    /// x coordinate of momentum vector at the outermost hit position
    double outerPx() const {
        return outerMomentum_.X();
    }
    /// y coordinate of momentum vector at the outermost hit position
    double outerPy() const {
        return outerMomentum_.Y();
    }
    /// z coordinate of momentum vector at the outermost hit position
    double outerPz() const {
        return outerMomentum_.Z();
    }
    /// x coordinate the outermost hit position
    double outerX() const {
        return outerPosition_.X();
    }
    /// y coordinate the outermost hit position
    double outerY() const {
        return outerPosition_.Y();
    }
    /// z coordinate the outermost hit position
    double outerZ() const {
        return outerPosition_.Z();
    }
    /// magnitude of momentum vector at the outermost hit position
    double outerP() const {
        return outerMomentum().R();
    }
    /// transverse momentum at the outermost hit position
    double outerPt() const {
        return outerMomentum().Rho();
    }
    /// azimuthal angle of the  momentum vector at the outermost hit position
    double outerPhi() const {
        return outerMomentum().Phi();
    }
    /// pseudorapidity the  momentum vector at the outermost hit position
    double outerEta() const {
        return outerMomentum().Eta();
    }
    /// polar angle of the  momentum vector at the outermost hit position
    double outerTheta() const {
        return outerMomentum().Theta();
    }
    /// polar radius of the outermost hit position
    double outerRadius() const {
        return outerPosition().Rho();
    }

    /// outermost trajectory state curvilinear errors
    CovarianceMatrix outerStateCovariance() const;
    /// innermost trajectory state curvilinear errors
    CovarianceMatrix innerStateCovariance() const;
    /// fill outermost trajectory state curvilinear errors
    CovarianceMatrix & fillOuter CMS_THREAD_SAFE (CovarianceMatrix &v) const;
    /// fill outermost trajectory state curvilinear errors
    CovarianceMatrix & fillInner CMS_THREAD_SAFE (CovarianceMatrix &v) const;
    /// DetId of the detector on which surface the outermost state is located
    unsigned int outerDetId() const {
        return outerDetId_;
    }
    /// DetId of the detector on which surface the innermost state is located
    unsigned int innerDetId() const {
        return innerDetId_;
    }
    // direction how the hits were sorted in the original seed
    PropagationDirection seedDirection() const {
        return seedDir_;
    }

    /**  return the edm::reference to the trajectory seed in the original
     *   seeds collection. If the collection has been dropped from the
     *   Event, the reference may be invalid. Its validity should be tested,
     *   before the reference is actually used.
     */
    edm::RefToBase<TrajectorySeed> seedRef() const {
        return seedRef_;
    }
    void setSeedRef(edm::RefToBase<TrajectorySeed> &r) {
        seedRef_ = r;
    }
    /// set the residuals
    void setResiduals(const TrackResiduals &r) {
        trackResiduals_ = r;
    }

    /// get the residuals
    const TrackResiduals &residuals() const {
        return trackResiduals_;
    }

private:

    /// outermost hit position
    Point outerPosition_;
    /// momentum vector at outermost hit position
    Vector outerMomentum_;
    /// outermost hit validity flag
    bool outerOk_;
    /// outermost trajectory state curvilinear errors
    float outerCovariance_[covarianceSize];
    unsigned int outerDetId_;


    /// innermost hit position
    Point innerPosition_;
    /// momentum vector at innermost hit position
    Vector innerMomentum_;
    /// innermost hit validity flag
    bool innerOk_;
    /// innermost trajectory state
    float innerCovariance_[covarianceSize];
    unsigned int innerDetId_;

    PropagationDirection seedDir_;
    edm::RefToBase<TrajectorySeed> seedRef_;

    /// unbiased track residuals
    TrackResiduals trackResiduals_;
};

}

#endif

