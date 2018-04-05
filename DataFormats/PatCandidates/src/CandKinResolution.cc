#include <DataFormats/PatCandidates/interface/CandKinResolution.h>
#include <DataFormats/PatCandidates/interface/ResolutionHelper.h>
#include <DataFormats/PatCandidates/interface/ParametrizationHelper.h>


pat::CandKinResolution::CandKinResolution() : 
    parametrization_(Invalid), 
    covariances_(),
    constraints_(),
    covmatrix_() 
{ 
}

pat::CandKinResolution::CandKinResolution(Parametrization parametrization, const std::vector<Scalar> &covariances, const std::vector<Scalar> &constraints) :
    parametrization_(parametrization),
    covariances_(covariances), 
    constraints_(constraints),
    covmatrix_()
{
    fillMatrix();
}

pat::CandKinResolution::CandKinResolution(Parametrization parametrization, const AlgebraicSymMatrix44 &covariance, const std::vector<Scalar> &constraints) :
    parametrization_(parametrization),
    covariances_(), 
    constraints_(constraints),
    covmatrix_(covariance)
{
    fillVector();
    if (sizeof(double) != sizeof(Scalar)) { // should become boost::mpl::if_c
        fillMatrix(); // forcing double => float => double conversion 
    }
}

pat::CandKinResolution::~CandKinResolution() {
}

double pat::CandKinResolution::resolEta(const pat::CandKinResolution::LorentzVector &p4)   const
{
    return pat::helper::ResolutionHelper::getResolEta(parametrization_, covmatrix_, p4);
}
double pat::CandKinResolution::resolTheta(const pat::CandKinResolution::LorentzVector &p4) const
{
    return pat::helper::ResolutionHelper::getResolTheta(parametrization_, covmatrix_, p4);
}
double pat::CandKinResolution::resolPhi(const pat::CandKinResolution::LorentzVector &p4)   const
{
    return pat::helper::ResolutionHelper::getResolPhi(parametrization_, covmatrix_, p4);
}
double pat::CandKinResolution::resolE(const pat::CandKinResolution::LorentzVector &p4)     const
{
    return pat::helper::ResolutionHelper::getResolE(parametrization_, covmatrix_, p4);
}
double pat::CandKinResolution::resolEt(const pat::CandKinResolution::LorentzVector &p4)    const
{
    return pat::helper::ResolutionHelper::getResolEt(parametrization_, covmatrix_, p4);
}
double pat::CandKinResolution::resolM(const pat::CandKinResolution::LorentzVector &p4)     const
{
    return pat::helper::ResolutionHelper::getResolM(parametrization_, covmatrix_, p4);
}
double pat::CandKinResolution::resolP(const pat::CandKinResolution::LorentzVector &p4)     const
{
    return pat::helper::ResolutionHelper::getResolP(parametrization_, covmatrix_, p4);
}
double pat::CandKinResolution::resolPt(const pat::CandKinResolution::LorentzVector &p4)    const
{
    return pat::helper::ResolutionHelper::getResolPt(parametrization_, covmatrix_, p4);
}
double pat::CandKinResolution::resolPInv(const pat::CandKinResolution::LorentzVector &p4)  const
{
    return pat::helper::ResolutionHelper::getResolPInv(parametrization_, covmatrix_, p4);
}
double pat::CandKinResolution::resolPx(const pat::CandKinResolution::LorentzVector &p4)    const
{
    return pat::helper::ResolutionHelper::getResolPx(parametrization_, covmatrix_, p4);
}
double pat::CandKinResolution::resolPy(const pat::CandKinResolution::LorentzVector &p4)    const
{
    return pat::helper::ResolutionHelper::getResolPy(parametrization_, covmatrix_, p4);
}
double pat::CandKinResolution::resolPz(const pat::CandKinResolution::LorentzVector &p4)    const
{
    return pat::helper::ResolutionHelper::getResolPz(parametrization_, covmatrix_, p4);
}

void pat::CandKinResolution::fillVector() { 
    if (dimension() == 3) {
        AlgebraicSymMatrix33 sub = covmatrix_.Sub<AlgebraicSymMatrix33>(0,0);
        covariances_.insert(covariances_.end(), sub.begin(), sub.end());
    } else {
        covariances_.insert(covariances_.end(), covmatrix_.begin(), covmatrix_.end());
    }
}

void pat::CandKinResolution::fillMatrixFrom(Parametrization parametrization, 
                                            const std::vector<Scalar>& covariances,
                                            AlgebraicSymMatrix44& covmatrix) { 
    int dimension = dimensionFrom(parametrization);
    if (dimension == 3) {
        if (covariances.size() == 3) {
            for (int i = 0; i < 3; ++i) covmatrix(i,i) = covariances[i];
        } else {
            covmatrix.Place_at(AlgebraicSymMatrix33(covariances.begin(), covariances.end()), 0, 0);
        }
    } else if (dimension == 4) {
        if (covariances.size() == 4) {
            for (int i = 0; i < 4; ++i) covmatrix(i,i) = covariances[i];
        } else {
            covmatrix = AlgebraicSymMatrix44(covariances.begin(), covariances.end());
        }
    }
}


void pat::CandKinResolution::fillMatrix() { 
    fillMatrixFrom(parametrization_, covariances_, covmatrix_);
}
