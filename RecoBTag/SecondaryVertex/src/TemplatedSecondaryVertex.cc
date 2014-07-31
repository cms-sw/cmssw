#include <cmath>

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

#include "RecoBTag/SecondaryVertex/interface/TemplatedSecondaryVertex.h"

namespace reco {

template <>
TemplatedSecondaryVertex<reco::VertexCompositePtrCandidate>::operator reco::Vertex() {
CovarianceMatrix err;
fillVertexCovariance(err);
return reco::Vertex(Vertex::Point(vertex()), err, vertexChi2(), vertexNdof(),numberOfSourceCandidatePtrs());
}

//it should be better without if it compiles
/*template <>
TemplatedSecondaryVertex<reco::Vertex>::operator reco::Vertex() {
return reco::Vertex(this);
}*/

template <>
Measurement1D
TemplatedSecondaryVertex<reco::VertexCompositePtrCandidate>::computeDist2d(const Vertex &pv, const VertexCompositePtrCandidate &sv,
                               const GlobalVector &direction, bool withPVError)
{
	typedef ROOT::Math::SVector<double, 2> SVector2;
	typedef ROOT::Math::SMatrix<double, 2, 2,
			ROOT::Math::MatRepSym<double, 2> > SMatrixSym2;
	CovarianceMatrix covariance;
	sv.fillVertexCovariance(covariance);

	SMatrixSym2 cov = covariance.Sub<SMatrixSym2>(0, 0);
	if (withPVError)
		cov += pv.covariance().Sub<SMatrixSym2>(0, 0);

	SVector2 vector(sv.vertex().X() - pv.position().X(),
	                sv.vertex().Y() - pv.position().Y());

	double dist = ROOT::Math::Mag(vector);
	double error = ROOT::Math::Similarity(cov, vector);
	if (error > 0.0 && dist > 1.0e-9)
		error = std::sqrt(error) / dist;
	else
		error = -1.0;

	if ((vector[0] * direction.x() +
	     vector[1] * direction.y()) < 0.0)
		dist = -dist;

	return Measurement1D(dist, error);
}
template <>
Measurement1D
TemplatedSecondaryVertex<reco::VertexCompositePtrCandidate>::computeDist3d(const Vertex &pv, const VertexCompositePtrCandidate &sv,
                               const GlobalVector &direction, bool withPVError)
{
	typedef ROOT::Math::SVector<double, 3> SVector3;
	typedef ROOT::Math::SMatrix<double, 3, 3,
			ROOT::Math::MatRepSym<double, 3> > SMatrixSym3;
	CovarianceMatrix covariance;
        sv.fillVertexCovariance(covariance);

	SMatrixSym3 cov = covariance;
	if (withPVError)
		cov += pv.covariance();

	SVector3 vector(sv.vertex().X() - pv.position().X(),
	                sv.vertex().Y() - pv.position().Y(),
	                sv.vertex().Z() - pv.position().Z());

	double dist = ROOT::Math::Mag(vector);
	double error = ROOT::Math::Similarity(cov, vector);
	if (error > 0.0 && dist > 1.0e-9)
		error = std::sqrt(error) / dist;
	else
		error = -1.0;

	if ((vector[0] * direction.x() +
	     vector[1] * direction.y() +
	     vector[2] * direction.z()) < 0.0)
		dist = -dist;

	return Measurement1D(dist, error);
}


template <>
Measurement1D
TemplatedSecondaryVertex<reco::Vertex>::computeDist2d(const Vertex &pv, const Vertex &sv,
                               const GlobalVector &direction, bool withPVError)
{
	typedef ROOT::Math::SVector<double, 2> SVector2;
	typedef ROOT::Math::SMatrix<double, 2, 2,
			ROOT::Math::MatRepSym<double, 2> > SMatrixSym2;

	SMatrixSym2 cov = sv.covariance().Sub<SMatrixSym2>(0, 0);
	if (withPVError)
		cov += pv.covariance().Sub<SMatrixSym2>(0, 0);

	SVector2 vector(sv.position().X() - pv.position().X(),
	                sv.position().Y() - pv.position().Y());

	double dist = ROOT::Math::Mag(vector);
	double error = ROOT::Math::Similarity(cov, vector);
	if (error > 0.0 && dist > 1.0e-9)
		error = std::sqrt(error) / dist;
	else
		error = -1.0;

	if ((vector[0] * direction.x() +
	     vector[1] * direction.y()) < 0.0)
		dist = -dist;

	return Measurement1D(dist, error);
}
template <>
Measurement1D
TemplatedSecondaryVertex<reco::Vertex>::computeDist3d(const Vertex &pv, const Vertex &sv,
                               const GlobalVector &direction, bool withPVError)
{
	typedef ROOT::Math::SVector<double, 3> SVector3;
	typedef ROOT::Math::SMatrix<double, 3, 3,
			ROOT::Math::MatRepSym<double, 3> > SMatrixSym3;

	SMatrixSym3 cov = sv.covariance();
	if (withPVError)
		cov += pv.covariance();

	SVector3 vector(sv.position().X() - pv.position().X(),
	                sv.position().Y() - pv.position().Y(),
	                sv.position().Z() - pv.position().Z());

	double dist = ROOT::Math::Mag(vector);
	double error = ROOT::Math::Similarity(cov, vector);
	if (error > 0.0 && dist > 1.0e-9)
		error = std::sqrt(error) / dist;
	else
		error = -1.0;

	if ((vector[0] * direction.x() +
	     vector[1] * direction.y() +
	     vector[2] * direction.z()) < 0.0)
		dist = -dist;

	return Measurement1D(dist, error);
}

}
