#ifndef TrackReco_TrackBase_h
#define TrackReco_TrackBase_h
/** \class reco::TrackBase TrackBase.h DataFormats/TrackReco/interface/TrackBase.h
 *
 * Common base class to all track types, including Muon fits.
 * Internally, the following information is stored: <BR>
 *   <DT> A reference position on the track: (vx,vy,vz) </DT>
 *   <DT> Momentum at this given reference point on track: (px,py,pz) </DT>
 *   <DT> 5D curvilinear covariance matrix from the track fit </DT>
 *   <DT> Charge </DT>
 *   <DT> Chi-square and number of degrees of freedom </DT>
 *   <DT> Summary information of the hit pattern </DT>
 *
 * For tracks reconstructed in the CMS Tracker, the reference position is the point of
 * closest approach to the centre of CMS. For muons, this is not necessarily true.
 *
 * Parameters associated to the 5D curvilinear covariance matrix: <BR>
 * <B> (qoverp, lambda, phi, dxy, dsz) </B><BR>
 * defined as:  <BR>
 *   <DT> qoverp = q / abs(p) = signed inverse of momentum [1/GeV] </DT>
 *   <DT> lambda = pi/2 - polar angle at the given point </DT>
 *   <DT> phi = azimuth angle at the given point </DT>
 *   <DT> dxy = -vx*sin(phi) + vy*cos(phi) [cm] </DT>
 *   <DT> dsz = vz*cos(lambda) - (vx*cos(phi)+vy*sin(phi))*sin(lambda) [cm] </DT>
 *
 * Geometrically, dxy is the signed distance in the XY plane between the
 * the straight line passing through (vx,vy) with azimuthal angle phi and
 * the point (0,0).<BR>
 * The dsz parameter is the signed distance in the SZ plane between the
 * the straight line passing through (vx,vy,vz) with angles (phi, lambda) and
 * the point (s=0,z=0). The S axis is defined by the projection of the
 * straight line onto the XY plane. The convention is to assign the S
 * coordinate for (vx,vy) as the value vx*cos(phi)+vy*sin(phi). This value is
 * zero when (vx,vy) is the point of minimum transverse distance to (0,0).
 *
 * Note that dxy and dsz provide sensible estimates of the distance from
 * the true particle trajectory to (0,0,0) ONLY in two cases:<BR>
 *   <DT> When (vx,vy,vz) already correspond to the point of minimum transverse
 *   distance to (0,0,0) or is close to it (so that the differences
 *   between considering the exact trajectory or a straight line in this range
 *   are negligible). This is usually true for Tracker tracks. </DT>
 *   <DT> When the track has infinite or extremely high momentum </DT>
 *
 * More details about this parametrization are provided in the following document: <BR>
 * <a href="http://cms.cern.ch/iCMS/jsp/openfile.jsp?type=NOTE&year=2006&files=NOTE2006_001.pdf">A. Strandlie, W. Wittek, "Propagation of Covariance Matrices...", CMS Note 2006/001</a> <BR>
 *
 * \author Thomas Speer, Luca Lista, Pascal Vanlaer, Juan Alcaraz
 *
 */

#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include <bitset>

namespace reco
{

class TrackBase
{

public:
    /// parameter dimension
    enum { dimension = 5 };

    /// error matrix size
    enum { covarianceSize = dimension * (dimension + 1) / 2 };

    /// parameter vector
    typedef math::Vector<dimension>::type ParameterVector;

    /// 5 parameter covariance matrix
    typedef math::Error<dimension>::type CovarianceMatrix;

    /// spatial vector
    typedef math::XYZVector Vector;

    /// point in the space
    typedef math::XYZPoint Point;

    /// enumerator provided indices to the five parameters
    enum {
        i_qoverp = 0,
        i_lambda,
        i_phi,
        i_dxy,
        i_dsz
    };

    /// index type
    typedef unsigned int index;


    /// track algorithm
    enum TrackAlgorithm {
        undefAlgorithm = 0, ctf = 1, rs = 2, cosmics = 3,
        initialStep = 4,
        lowPtTripletStep = 5,
        pixelPairStep = 6,
        detachedTripletStep = 7,
        mixedTripletStep = 8,
        pixelLessStep = 9,
        tobTecStep = 10,
        jetCoreRegionalStep = 11,
        conversionStep = 12,
        muonSeededStepInOut = 13,
        muonSeededStepOutIn = 14,
        outInEcalSeededConv = 15, inOutEcalSeededConv = 16,
        nuclInter = 17,
        standAloneMuon = 18, globalMuon = 19, cosmicStandAloneMuon = 20, cosmicGlobalMuon = 21,
        iter1LargeD0 = 22, iter2LargeD0 = 23, iter3LargeD0 = 24, iter4LargeD0 = 25, iter5LargeD0 = 26,
        bTagGhostTracks = 27,
        beamhalo = 28,
        gsf = 29,
	// HLT algo name
	hltPixel = 30,
	// steps used by PF
	hltIter0 = 31,
	hltIter1 = 32,
	hltIter2 = 33,
	hltIter3 = 34,
	hltIter4 = 35,
	// steps used by all other objects @HLT
	hltIterX = 36,
   // steps used by HI muon regional iterative tracking
   hiRegitMuInitialStep = 37,
   hiRegitMuLowPtTripletStep = 38,
   hiRegitMuPixelPairStep = 39,
   hiRegitMuDetachedTripletStep = 40,
   hiRegitMuMixedTripletStep = 41,
   hiRegitMuPixelLessStep = 42,
   hiRegitMuTobTecStep = 43,
   hiRegitMuMuonSeededStepInOut = 44,
   hiRegitMuMuonSeededStepOutIn = 45,
   algoSize = 46
    };

    /// algo mask
    typedef std::bitset<algoSize> AlgoMask;
 

    static const std::string algoNames[];

    /// track quality
    enum TrackQuality {
        undefQuality = -1,
        loose = 0,
        tight = 1,
        highPurity = 2,
        confirmed = 3,  // means found by more than one iteration
        goodIterative = 4,  // meaningless
        looseSetWithPV = 5,
        highPuritySetWithPV = 6,
        discarded = 7, // because a better track found. kept in the collection for reference....
        qualitySize = 8
    };

    static const std::string qualityNames[];

    /// default constructor
    TrackBase();

    /// constructor from fit parameters and error matrix
    TrackBase(double chi2, double ndof, const Point &vertex,
              const Vector &momentum, int charge, const CovarianceMatrix &cov,
              TrackAlgorithm = undefAlgorithm, TrackQuality quality = undefQuality,
              signed char nloops = 0);

    /// virtual destructor
    virtual ~TrackBase();

    /// chi-squared of the fit
    double chi2() const;

    /// number of degrees of freedom of the fit
    double ndof() const;

    /// chi-squared divided by n.d.o.f. (or chi-squared * 1e6 if n.d.o.f. is zero)
    double normalizedChi2() const;

    /// track electric charge
    int charge() const;

    /// q / p
    double qoverp() const;

    /// polar angle
    double theta() const;

    /// Lambda angle
    double lambda() const;

    /// dxy parameter. (This is the transverse impact parameter w.r.t. to (0,0,0) ONLY if refPoint is close to (0,0,0): see parametrization definition above for details). See also function dxy(myBeamSpot).
    double dxy() const;

    /// dxy parameter in perigee convention (d0 = -dxy)
    double d0() const;

    /// dsz parameter (THIS IS NOT the SZ impact parameter to (0,0,0) if refPoint is far from  (0,0,0): see parametrization definition above for details)
    double dsz() const;

    /// dz parameter (= dsz/cos(lambda)). This is the track z0 w.r.t (0,0,0) only if the refPoint is close to (0,0,0). See also function dz(myBeamSpot)
    double dz() const;

    /// momentum vector magnitude
    double p() const;

    /// track transverse momentum
    double pt() const;

    /// x coordinate of momentum vector
    double px() const;

    /// y coordinate of momentum vector
    double py() const;

    /// z coordinate of momentum vector
    double pz() const;

    /// azimuthal angle of momentum vector
    double phi() const;

    /// pseudorapidity of momentum vector
    double eta() const;

    /// x coordinate of the reference point on track
    double vx() const;

    /// y coordinate of the reference point on track
    double vy() const;

    /// z coordinate of the reference point on track
    double vz() const;

    /// track momentum vector
    const Vector &momentum() const;

    /// Reference point on the track
    const Point &referencePoint() const;

    /// reference point on the track. This method is DEPRECATED, please use referencePoint() instead
    const Point &vertex() const ;
    //__attribute__((deprecated("This method is DEPRECATED, please use referencePoint() instead.")));

    /// dxy parameter with respect to a user-given beamSpot  (WARNING: this quantity can only be interpreted as a minimum transverse distance if beamSpot, if the beam spot is reasonably close to the refPoint, since linear approximations are involved). This is a good approximation for Tracker tracks.
    double dxy(const Point &myBeamSpot) const;

    /// dxy parameter with respect to the beamSpot taking into account the beamspot slopes (WARNING: this quantity can only be interpreted as a minimum transverse distance if beamSpot, if the beam spot is reasonably close to the refPoint, since linear approximations are involved). This is a good approximation for Tracker tracks.
    double dxy(const BeamSpot &theBeamSpot) const;

    /// dsz parameter with respect to a user-given beamSpot (WARNING: this quantity can only be interpreted as the distance in the S-Z plane to the beamSpot, if the beam spot is reasonably close to the refPoint, since linear approximations are involved). This is a good approximation for Tracker tracks.
    double dsz(const Point &myBeamSpot) const;

    /// dz parameter with respect to a user-given beamSpot (WARNING: this quantity can only be interpreted as the track z0, if the beamSpot is reasonably close to the refPoint, since linear approximations are involved). This is a good approximation for Tracker tracks.
    double dz(const Point &myBeamSpot) const;

    /// Track parameters with one-to-one correspondence to the covariance matrix
    ParameterVector parameters() const;

    /// return track covariance matrix
    CovarianceMatrix covariance() const;

    /// i-th parameter ( i = 0, ... 4 )
    double parameter(int i) const;
    
    /// (i,j)-th element of covariance matrix (i, j = 0, ... 4)
    double covariance(int i, int j) const;

    /// error on specified element
    double error(int i) const;

    /// error on signed transverse curvature
    double qoverpError() const;

    /// error on Pt (set to 1000 TeV if charge==0 for safety)
    double ptError() const;

    /// error on theta
    double thetaError() const;

    /// error on lambda
    double lambdaError() const;

    /// error on eta
    double etaError() const;

    /// error on phi
    double phiError() const;

    /// error on dxy
    double dxyError() const;

    /// error on d0
    double d0Error() const;

    /// error on dsz
    double dszError() const;

    /// error on dz
    double dzError() const;

    /// fill SMatrix
    CovarianceMatrix &fill(CovarianceMatrix &v) const;

    /// covariance matrix index in array
    static index covIndex(index i, index j);

    /// Access the hit pattern, indicating in which Tracker layers the track has hits.
    const HitPattern &hitPattern() const;

    /// number of valid hits found
    unsigned short numberOfValidHits() const;

    /// number of cases where track crossed a layer without getting a hit.
    unsigned short numberOfLostHits() const;

    /// fraction of valid hits on the track
    double validFraction() const;

    /// append hit patterns from vector of hit references
    template<typename C>
    bool appendHits(const C &c, const TrackerTopology& ttopo);

    template<typename I>
    bool appendHits(const I &begin, const I &end, const TrackerTopology& ttopo);

    /// append a single hit to the HitPattern
    bool appendHitPattern(const TrackingRecHit &hit, const TrackerTopology& ttopo);
    bool appendHitPattern(const DetId &id, TrackingRecHit::Type hitType, const TrackerTopology& ttopo);

    /**
     * This is meant to be used only in cases where the an
     * already-packed hit information is re-interpreted in terms of
     * HitPattern (i.e. MiniAOD PackedCandidate, and the IO rule for
     * reading old versions of HitPattern)
     */
    bool appendTrackerHitPattern(uint16_t subdet, uint16_t layer, uint16_t stereo, TrackingRecHit::Type hitType);

    /**
     * This is meant to be used only in cases where the an
     * already-packed hit information is re-interpreted in terms of
     * HitPattern (i.e. the IO rule for reading old versions of
     * HitPattern)
     */
    bool appendMuonHitPattern(const DetId& id, TrackingRecHit::Type hitType);

    /// Sets HitPattern as empty
    void resetHitPattern();

    ///Track algorithm
    void setAlgorithm(const TrackAlgorithm a);
   
    void setOriginalAlgorithm(const TrackAlgorithm a);

    void setAlgoMask(AlgoMask a) { algoMask_ = a;}

    AlgoMask algoMask() const { return algoMask_;}
#if ( !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__) ) || defined(__ROOTCLING__)
    unsigned long long algoMaskUL() const { return algoMask().to_ullong();}
#endif
    bool isAlgoInMask(TrackAlgorithm a) const {return algoMask()[a];}


    TrackAlgorithm algo() const ;
    TrackAlgorithm originalAlgo() const ;


    std::string algoName() const;

    static std::string algoName(TrackAlgorithm);

    static TrackAlgorithm algoByName(const std::string &name);

    ///Track quality
    bool quality(const TrackQuality) const;

    void setQuality(const TrackQuality);

    static std::string qualityName(TrackQuality);

    static TrackQuality qualityByName(const std::string &name);

    int qualityMask() const;

    void setQualityMask(int qualMask);

    void setNLoops(signed char value);

    bool isLooper() const;

    signed char nLoops() const;

private:
    /// hit pattern
    HitPattern hitPattern_;

    /// perigee 5x5 covariance matrix
    float covariance_[covarianceSize];

    /// chi-squared
    float chi2_;

    /// innermost (reference) point on track
    Point vertex_;

    /// momentum vector at innermost point
    Vector momentum_;

    /// algo mask, bit set for the algo where it was reconstructed + each algo a track was found overlapping by the listmerger
    std::bitset<algoSize> algoMask_;

    /// number of degrees of freedom
    float ndof_;

    /// electric charge
    char charge_;

    /// track algorithm
    uint8_t algorithm_;

    /// track algorithm
    uint8_t originalAlgorithm_;


    /// track quality
    uint8_t quality_;

    /// number of loops made during the building of the trajectory of a looper particle
    signed char nLoops_; // I use signed char because I don't expect more than 128 loops and I could use a negative value for a special purpose.
};

//  Access the hit pattern, indicating in which Tracker layers the track has hits.
inline const HitPattern & TrackBase::hitPattern() const
{
    return hitPattern_;
}

inline bool TrackBase::appendHitPattern(const DetId &id, TrackingRecHit::Type hitType, const TrackerTopology& ttopo)
{
    return hitPattern_.appendHit(id, hitType, ttopo);
}

inline bool TrackBase::appendHitPattern(const TrackingRecHit &hit, const TrackerTopology& ttopo)
{
    return hitPattern_.appendHit(hit, ttopo);
}

inline bool TrackBase::appendTrackerHitPattern(uint16_t subdet, uint16_t layer, uint16_t stereo, TrackingRecHit::Type hitType) {
    return hitPattern_.appendTrackerHit(subdet, layer, stereo, hitType);
}

inline bool TrackBase::appendMuonHitPattern(const DetId& id, TrackingRecHit::Type hitType) {
    return hitPattern_.appendMuonHit(id, hitType);
}

inline void TrackBase::resetHitPattern()
{
    hitPattern_.clear();
}

template<typename I>
bool TrackBase::appendHits(const I &begin, const I &end, const TrackerTopology& ttopo)
{
    return hitPattern_.appendHits(begin, end, ttopo);
}

template<typename C>
bool TrackBase::appendHits(const C &c, const TrackerTopology& ttopo)
{
    return hitPattern_.appendHits(c.begin(), c.end(), ttopo);
}

inline TrackBase::index TrackBase::covIndex(index i, index j)
{
    int a = (i <= j ? i : j);
    int b = (i <= j ? j : i);
    return b * (b + 1) / 2 + a;
}

inline TrackBase::TrackAlgorithm TrackBase::algo() const
{
    return (TrackAlgorithm) (algorithm_);
}
inline TrackBase::TrackAlgorithm TrackBase::originalAlgo() const
{
    return (TrackAlgorithm) (originalAlgorithm_);
}



inline std::string TrackBase::algoName() const { return TrackBase::algoName(algo()); }

inline bool TrackBase::quality(const TrackBase::TrackQuality q) const
{
    switch (q) {
    case undefQuality:
        return quality_ == 0;
    case goodIterative:
        return (quality_ & (1 << TrackBase::highPurity)) >> TrackBase::highPurity;
    default:
        return (quality_ & (1 << q)) >> q;
    }
    return false;
}

inline void TrackBase::setQuality(const TrackBase::TrackQuality q)
{
    if (q == undefQuality) {
        quality_ = 0;
    } else {
        quality_ |= (1 << q);
    }
}

inline std::string TrackBase::qualityName(TrackQuality q)
{
    if (int(q) < int(qualitySize) && int(q) >= 0) {
        return qualityNames[int(q)];
    }
    return "undefQuality";
}

inline std::string TrackBase::algoName(TrackAlgorithm a)
{
    if (int(a) < int(algoSize) && int(a) > 0) {
        return algoNames[int(a)];
    }
    return "undefAlgorithm";
}

// chi-squared of the fit
inline double TrackBase::chi2() const
{
    return chi2_;
}

// number of degrees of freedom of the fit
inline double TrackBase::ndof() const
{
    return ndof_;
}

// chi-squared divided by n.d.o.f. (or chi-squared * 1e6 if n.d.o.f. is zero)
inline double TrackBase::normalizedChi2() const
{
    return ndof_ != 0 ? chi2_ / ndof_ : chi2_ * 1e6;
}

// track electric charge
inline int TrackBase::charge() const
{
    return charge_;
}

// q / p
inline double TrackBase::qoverp() const
{
    return charge() / p();
}

// polar angle
inline double TrackBase::theta() const
{
    return momentum_.theta();
}

// Lambda angle
inline double TrackBase::lambda() const
{
    return M_PI_2 - momentum_.theta();
}

// dxy parameter. (This is the transverse impact parameter w.r.t. to (0,0,0) ONLY if refPoint is close to (0,0,0): see parametrization definition above for details). See also function dxy(myBeamSpot) below.
inline double TrackBase::dxy() const
{
    return (-vx() * py() + vy() * px()) / pt();
}

// dxy parameter in perigee convention (d0 = -dxy)
inline double TrackBase::d0() const
{
    return -dxy();
}

// dsz parameter (THIS IS NOT the SZ impact parameter to (0,0,0) if refPoint is far from (0,0,0): see parametrization definition above for details)
inline double TrackBase::dsz() const
{
    return vz() * pt() / p() - (vx() * px() + vy() * py()) / pt() * pz() / p();
}

// dz parameter (= dsz/cos(lambda)). This is the track z0 w.r.t (0,0,0) only if the refPoint is close to (0,0,0). See also function dz(myBeamSpot) below.
inline double TrackBase::dz() const
{
    return vz() - (vx() * px() + vy() * py()) / pt() * (pz() / pt());
}

// momentum vector magnitude
inline double TrackBase::p() const
{
    return momentum_.R();
}

// track transverse momentum
inline double TrackBase::pt() const
{
    return sqrt(momentum_.Perp2());
}

// x coordinate of momentum vector
inline double TrackBase::px() const
{
    return momentum_.x();
}

// y coordinate of momentum vector
inline double TrackBase::py() const
{
    return momentum_.y();
}

// z coordinate of momentum vector
inline double TrackBase::pz() const
{
    return momentum_.z();
}

// azimuthal angle of momentum vector
inline double TrackBase::phi() const
{
    return momentum_.Phi();
}

// pseudorapidity of momentum vector
inline double TrackBase::eta() const
{
    return momentum_.Eta();
}

// x coordinate of the reference point on track
inline double TrackBase::vx() const
{
    return vertex_.x();
}

// y coordinate of the reference point on track
inline double TrackBase::vy() const
{
    return vertex_.y();
}

// z coordinate of the reference point on track
inline double TrackBase::vz() const
{
    return vertex_.z();
}

// track momentum vector
inline const TrackBase::Vector & TrackBase::momentum() const
{
    return momentum_;
}

// Reference point on the track
inline const TrackBase::Point & TrackBase::referencePoint() const
{
    return vertex_;
}

// reference point on the track. This method is DEPRECATED, please use referencePoint() instead
inline const TrackBase::Point & TrackBase::vertex() const
{
    return vertex_;
}

// dxy parameter with respect to a user-given beamSpot
// (WARNING: this quantity can only be interpreted as a minimum transverse distance if beamSpot, if the beam spot is reasonably close to the refPoint, since linear approximations are involved).
// This is a good approximation for Tracker tracks.
inline double TrackBase::dxy(const Point &myBeamSpot) const
{
    return (-(vx() - myBeamSpot.x()) * py() + (vy() - myBeamSpot.y()) * px()) / pt();
}

// dxy parameter with respect to the beamSpot taking into account the beamspot slopes
// (WARNING: this quantity can only be interpreted as a minimum transverse distance if beamSpot, if the beam spot is reasonably close to the refPoint, since linear approximations are involved).
// This is a good approximation for Tracker tracks.
inline double TrackBase::dxy(const BeamSpot &theBeamSpot) const
{
    return dxy(theBeamSpot.position(vz()));
}

// dsz parameter with respect to a user-given beamSpot
// (WARNING: this quantity can only be interpreted as the distance in the S-Z plane to the beamSpot, if the beam spot is reasonably close to the refPoint, since linear approximations are involved).
// This is a good approximation for Tracker tracks.
inline double TrackBase::dsz(const Point &myBeamSpot) const
{
    return (vz() - myBeamSpot.z()) * pt() / p() - ((vx() - myBeamSpot.x()) * px() + (vy() - myBeamSpot.y()) * py()) / pt() * pz() / p();
}

// dz parameter with respect to a user-given beamSpot
// (WARNING: this quantity can only be interpreted as the track z0, if the beamSpot is reasonably close to the refPoint, since linear approximations are involved).
// This is a good approximation for Tracker tracks.
inline double TrackBase::dz(const Point &myBeamSpot) const
{
    return (vz() - myBeamSpot.z()) - ((vx() - myBeamSpot.x()) * px() + (vy() - myBeamSpot.y()) * py()) / pt() * pz() / pt();
}

// Track parameters with one-to-one correspondence to the covariance matrix
inline TrackBase::ParameterVector TrackBase::parameters() const
{
    return TrackBase::ParameterVector(qoverp(), lambda(), phi(), dxy(), dsz());
}

// return track covariance matrix
inline TrackBase::CovarianceMatrix TrackBase::covariance() const
{
    CovarianceMatrix m;
    fill(m);
    return m;
}

// i-th parameter ( i = 0, ... 4 )
inline double TrackBase::parameter(int i) const
{
    return parameters()[i];
}

// (i,j)-th element of covariance matrix (i, j = 0, ... 4)
inline double TrackBase::covariance(int i, int j) const
{
    return covariance_[covIndex(i, j)];
}

// error on specified element
inline double TrackBase::error(int i) const
{
    return sqrt(covariance_[covIndex(i, i)]);
}

// error on signed transverse curvature
inline double TrackBase::qoverpError() const
{
    return error(i_qoverp);
}

// error on Pt (set to 1000 TeV if charge==0 for safety)
inline double TrackBase::ptError() const
{
    return (charge() != 0) ?  sqrt(
               pt() * pt() * p() * p() / charge() / charge() * covariance(i_qoverp, i_qoverp)
               + 2 * pt() * p() / charge() * pz() * covariance(i_qoverp, i_lambda)
               + pz() * pz() * covariance(i_lambda, i_lambda)) : 1.e6;
}

// error on theta
inline double TrackBase::thetaError() const
{
    return error(i_lambda);
}

// error on lambda
inline double TrackBase::lambdaError() const
{
    return error(i_lambda);
}

// error on eta
inline double TrackBase::etaError() const
{
    return error(i_lambda) * p() / pt();
}

// error on phi
inline double TrackBase::phiError() const
{
    return error(i_phi);
}

// error on dxy
inline double TrackBase::dxyError() const
{
    return error(i_dxy);
}

// error on d0
inline double TrackBase::d0Error() const
{
    return error(i_dxy);
}

// error on dsz
inline double TrackBase::dszError() const
{
    return error(i_dsz);
}

// error on dz
inline double TrackBase::dzError() const
{
    return error(i_dsz) * p() / pt();
}

// number of valid hits found
inline unsigned short TrackBase::numberOfValidHits() const
{
    return hitPattern_.numberOfValidHits();
}

// number of cases where track crossed a layer without getting a hit.
inline unsigned short TrackBase::numberOfLostHits() const
{
    return hitPattern_.numberOfLostHits(HitPattern::TRACK_HITS);
}

// fraction of valid hits on the track
inline double TrackBase::validFraction() const
{
    int valid = hitPattern_.numberOfValidTrackerHits();
    int lost  = hitPattern_.numberOfLostTrackerHits(HitPattern::TRACK_HITS);
    int lostIn = hitPattern_.numberOfLostTrackerHits(HitPattern::MISSING_INNER_HITS);
    int lostOut = hitPattern_.numberOfLostTrackerHits(HitPattern::MISSING_OUTER_HITS);

    if ((valid + lost + lostIn + lostOut) == 0) {
        return -1;
    }

    return valid / (double)(valid + lost + lostIn + lostOut);
}

//Track algorithm
inline void TrackBase::setAlgorithm(const TrackBase::TrackAlgorithm a)
{
    algorithm_  = a;
    algoMask_.reset();
    setOriginalAlgorithm(a);
}

inline void TrackBase::setOriginalAlgorithm(const TrackBase::TrackAlgorithm a)
{
   originalAlgorithm_  = a;
   algoMask_.set(a);
}



inline int TrackBase::qualityMask() const
{
    return quality_;
}

inline void TrackBase::setQualityMask(int qualMask)
{
    quality_ = qualMask;
}

inline void TrackBase::setNLoops(signed char value)
{
    nLoops_ = value;
}

inline bool TrackBase::isLooper() const
{
    return (nLoops_ > 0);
}

inline signed char TrackBase::nLoops() const
{
    return nLoops_;
}

} // namespace reco

#endif

