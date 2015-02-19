/*
 * GblPoint.cpp
 *
 *  Created on: Aug 18, 2011
 *      Author: kleinwrt
 */

#include "Alignment/ReferenceTrajectories/interface/GblPoint.h"

//! Namespace for the general broken lines package
namespace gbl {

/// Create a point.
/**
 * Create point on (initial) trajectory. Needs transformation jacobian from previous point.
 * \param [in] aJacobian Transformation jacobian from previous point
 */
GblPoint::GblPoint(const TMatrixD &aJacobian) :
		theLabel(0), theOffset(0), measDim(0), transFlag(false), measTransformation(), scatFlag(
				false), localDerivatives(), globalLabels(), globalDerivatives() {

	for (unsigned int i = 0; i < 5; ++i) {
		for (unsigned int j = 0; j < 5; ++j) {
			p2pJacobian(i, j) = aJacobian(i, j);
		}
	}
}

GblPoint::GblPoint(const SMatrix55 &aJacobian) :
		theLabel(0), theOffset(0), p2pJacobian(aJacobian), measDim(0), transFlag(
				false), measTransformation(), scatFlag(false), localDerivatives(), globalLabels(), globalDerivatives() {

}

GblPoint::~GblPoint() {
}

/// Add a measurement to a point.
/**
 * Add measurement (in meas. system) with diagonal precision (inverse covariance) matrix.
 * ((up to) 2D: position, 4D: slope+position, 5D: curvature+slope+position)
 * \param [in] aProjection Projection from local to measurement system
 * \param [in] aResiduals Measurement residuals
 * \param [in] aPrecision Measurement precision (diagonal)
 * \param [in] minPrecision Minimal precision to accept measurement
 */
void GblPoint::addMeasurement(const TMatrixD &aProjection,
		const TVectorD &aResiduals, const TVectorD &aPrecision,
		double minPrecision) {
	measDim = aResiduals.GetNrows();
	unsigned int iOff = 5 - measDim;
	for (unsigned int i = 0; i < measDim; ++i) {
		measResiduals(iOff + i) = aResiduals[i];
		measPrecision(iOff + i) = (
				aPrecision[i] >= minPrecision ? aPrecision[i] : 0.);
		for (unsigned int j = 0; j < measDim; ++j) {
			measProjection(iOff + i, iOff + j) = aProjection(i, j);
		}
	}
}

/// Add a measurement to a point.
/**
 * Add measurement (in meas. system) with arbitrary precision (inverse covariance) matrix.
 * Will be diagonalized.
 * ((up to) 2D: position, 4D: slope+position, 5D: curvature+slope+position)
 * \param [in] aProjection Projection from local to measurement system
 * \param [in] aResiduals Measurement residuals
 * \param [in] aPrecision Measurement precision (matrix)
 * \param [in] minPrecision Minimal precision to accept measurement
 */
void GblPoint::addMeasurement(const TMatrixD &aProjection,
		const TVectorD &aResiduals, const TMatrixDSym &aPrecision,
		double minPrecision) {
	measDim = aResiduals.GetNrows();
	TMatrixDSymEigen measEigen(aPrecision);
	measTransformation.ResizeTo(measDim, measDim);
	measTransformation = measEigen.GetEigenVectors();
	measTransformation.T();
	transFlag = true;
	TVectorD transResiduals = measTransformation * aResiduals;
	TVectorD transPrecision = measEigen.GetEigenValues();
	TMatrixD transProjection = measTransformation * aProjection;
	unsigned int iOff = 5 - measDim;
	for (unsigned int i = 0; i < measDim; ++i) {
		measResiduals(iOff + i) = transResiduals[i];
		measPrecision(iOff + i) = (
				transPrecision[i] >= minPrecision ? transPrecision[i] : 0.);
		for (unsigned int j = 0; j < measDim; ++j) {
			measProjection(iOff + i, iOff + j) = transProjection(i, j);
		}
	}
}

/// Add a measurement to a point.
/**
 * Add measurement in local system with diagonal precision (inverse covariance) matrix.
 * ((up to) 2D: position, 4D: slope+position, 5D: curvature+slope+position)
 * \param [in] aResiduals Measurement residuals
 * \param [in] aPrecision Measurement precision (diagonal)
 * \param [in] minPrecision Minimal precision to accept measurement
 */
void GblPoint::addMeasurement(const TVectorD &aResiduals,
		const TVectorD &aPrecision, double minPrecision) {
	measDim = aResiduals.GetNrows();
	unsigned int iOff = 5 - measDim;
	for (unsigned int i = 0; i < measDim; ++i) {
		measResiduals(iOff + i) = aResiduals[i];
		measPrecision(iOff + i) = (
				aPrecision[i] >= minPrecision ? aPrecision[i] : 0.);
	}
	measProjection = ROOT::Math::SMatrixIdentity();
}

/// Add a measurement to a point.
/**
 * Add measurement in local system with arbitrary precision (inverse covariance) matrix.
 * Will be diagonalized.
 * ((up to) 2D: position, 4D: slope+position, 5D: curvature+slope+position)
 * \param [in] aResiduals Measurement residuals
 * \param [in] aPrecision Measurement precision (matrix)
 * \param [in] minPrecision Minimal precision to accept measurement
 */
void GblPoint::addMeasurement(const TVectorD &aResiduals,
		const TMatrixDSym &aPrecision, double minPrecision) {
	measDim = aResiduals.GetNrows();
	TMatrixDSymEigen measEigen(aPrecision);
	measTransformation.ResizeTo(measDim, measDim);
	measTransformation = measEigen.GetEigenVectors();
	measTransformation.T();
	transFlag = true;
	TVectorD transResiduals = measTransformation * aResiduals;
	TVectorD transPrecision = measEigen.GetEigenValues();
	unsigned int iOff = 5 - measDim;
	for (unsigned int i = 0; i < measDim; ++i) {
		measResiduals(iOff + i) = transResiduals[i];
		measPrecision(iOff + i) = (
				transPrecision[i] >= minPrecision ? transPrecision[i] : 0.);
		for (unsigned int j = 0; j < measDim; ++j) {
			measProjection(iOff + i, iOff + j) = measTransformation(i, j);
		}
	}
}

/// Check for measurement at a point.
/**
 * Get dimension of measurement (0 = none).
 * \return measurement dimension
 */
unsigned int GblPoint::hasMeasurement() const {
	return measDim;
}

/// Retrieve measurement of a point.
/**
 * \param [out] aProjection Projection from (diagonalized) measurement to local system
 * \param [out] aResiduals Measurement residuals
 * \param [out] aPrecision Measurement precision (diagonal)
 */
void GblPoint::getMeasurement(SMatrix55 &aProjection, SVector5 &aResiduals,
		SVector5 &aPrecision) const {
	aProjection = measProjection;
	aResiduals = measResiduals;
	aPrecision = measPrecision;
}

/// Get measurement transformation (from diagonalization).
/**
 * \param [out] aTransformation Transformation matrix
 */
void GblPoint::getMeasTransformation(TMatrixD &aTransformation) const {
	aTransformation.ResizeTo(measDim, measDim);
	if (transFlag) {
		aTransformation = measTransformation;
	} else {
		aTransformation.UnitMatrix();
	}
}

/// Add a (thin) scatterer to a point.
/**
 * Add scatterer with diagonal precision (inverse covariance) matrix.
 * Changes local track direction.
 *
 * \param [in] aResiduals Scatterer residuals
 * \param [in] aPrecision Scatterer precision (diagonal of inverse covariance matrix)
 */
void GblPoint::addScatterer(const TVectorD &aResiduals,
		const TVectorD &aPrecision) {
	scatFlag = true;
	scatResiduals(0) = aResiduals[0];
	scatResiduals(1) = aResiduals[1];
	scatPrecision(0) = aPrecision[0];
	scatPrecision(1) = aPrecision[1];
	scatTransformation = ROOT::Math::SMatrixIdentity();
}

/// Add a (thin) scatterer to a point.
/**
 * Add scatterer with arbitrary precision (inverse covariance) matrix.
 * Will be diagonalized. Changes local track direction.
 *
 * The precision matrix for the local slopes is defined by the
 * angular scattering error theta_0 and the scalar products c_1, c_2 of the
 * offset directions in the local frame with the track direction:
 *
 *            (1 - c_1*c_1 - c_2*c_2)   |  1 - c_1*c_1     - c_1*c_2  |
 *       P =  ----------------------- * |                             |
 *                theta_0*theta_0       |    - c_1*c_2   1 - c_2*c_2  |
 *
 * \param [in] aResiduals Scatterer residuals
 * \param [in] aPrecision Scatterer precision (matrix)
 */
void GblPoint::addScatterer(const TVectorD &aResiduals,
		const TMatrixDSym &aPrecision) {
	scatFlag = true;
	TMatrixDSymEigen scatEigen(aPrecision);
	TMatrixD aTransformation = scatEigen.GetEigenVectors();
	aTransformation.T();
	TVectorD transResiduals = aTransformation * aResiduals;
	TVectorD transPrecision = scatEigen.GetEigenValues();
	for (unsigned int i = 0; i < 2; ++i) {
		scatResiduals(i) = transResiduals[i];
		scatPrecision(i) = transPrecision[i];
		for (unsigned int j = 0; j < 2; ++j) {
			scatTransformation(i, j) = aTransformation(i, j);
		}
	}
}

/// Check for scatterer at a point.
bool GblPoint::hasScatterer() const {
	return scatFlag;
}

/// Retrieve scatterer of a point.
/**
 * \param [out] aTransformation Scatterer transformation from diagonalization
 * \param [out] aResiduals Scatterer residuals
 * \param [out] aPrecision Scatterer precision (diagonal)
 */
void GblPoint::getScatterer(SMatrix22 &aTransformation, SVector2 &aResiduals,
		SVector2 &aPrecision) const {
	aTransformation = scatTransformation;
	aResiduals = scatResiduals;
	aPrecision = scatPrecision;
}

/// Get scatterer transformation (from diagonalization).
/**
 * \param [out] aTransformation Transformation matrix
 */
void GblPoint::getScatTransformation(TMatrixD &aTransformation) const {
	aTransformation.ResizeTo(2, 2);
	if (scatFlag) {
		for (unsigned int i = 0; i < 2; ++i) {
			for (unsigned int j = 0; j < 2; ++j) {
				aTransformation(i, j) = scatTransformation(i, j);
			}
		}
	} else {
		aTransformation.UnitMatrix();
	}
}

/// Add local derivatives to a point.
/**
 * Point needs to have a measurement.
 * \param [in] aDerivatives Local derivatives (matrix)
 */
void GblPoint::addLocals(const TMatrixD &aDerivatives) {
	if (measDim) {
		localDerivatives.ResizeTo(aDerivatives);
		if (transFlag) {
			localDerivatives = measTransformation * aDerivatives;
		} else {
			localDerivatives = aDerivatives;
		}
	}
}

/// Retrieve number of local derivatives from a point.
unsigned int GblPoint::getNumLocals() const {
	return localDerivatives.GetNcols();
}

/// Retrieve local derivatives from a point.
const TMatrixD& GblPoint::getLocalDerivatives() const {
	return localDerivatives;
}

/// Add global derivatives to a point.
/**
 * Point needs to have a measurement.
 * \param [in] aLabels Global derivatives labels
 * \param [in] aDerivatives Global derivatives (matrix)
 */
void GblPoint::addGlobals(const std::vector<int> &aLabels,
		const TMatrixD &aDerivatives) {
	if (measDim) {
		globalLabels = aLabels;
		globalDerivatives.ResizeTo(aDerivatives);
		if (transFlag) {
			globalDerivatives = measTransformation * aDerivatives;
		} else {
			globalDerivatives = aDerivatives;
		}

	}
}

/// Retrieve number of global derivatives from a point.
unsigned int GblPoint::getNumGlobals() const {
	return globalDerivatives.GetNcols();
}

/// Retrieve global derivatives labels from a point.
std::vector<int> GblPoint::getGlobalLabels() const {
	return globalLabels;
}

/// Retrieve global derivatives from a point.
const TMatrixD& GblPoint::getGlobalDerivatives() const {
	return globalDerivatives;
}

/// Define label of point (by GBLTrajectory constructor)
/**
 * \param [in] aLabel Label identifying point
 */
void GblPoint::setLabel(unsigned int aLabel) {
	theLabel = aLabel;
}

/// Retrieve label of point
unsigned int GblPoint::getLabel() const {
	return theLabel;
}

/// Define offset for point (by GBLTrajectory constructor)
/**
 * \param [in] anOffset Offset number
 */
void GblPoint::setOffset(int anOffset) {
	theOffset = anOffset;
}

/// Retrieve offset for point
int GblPoint::getOffset() const {
	return theOffset;
}

/// Retrieve point-to-(previous)point jacobian
const SMatrix55& GblPoint::getP2pJacobian() const {
	return p2pJacobian;
}

/// Define jacobian to previous scatterer (by GBLTrajectory constructor)
/**
 * \param [in] aJac Jacobian
 */
void GblPoint::addPrevJacobian(const SMatrix55 &aJac) {
	int ifail = 0;
// to optimize: need only two last rows of inverse
//	prevJacobian = aJac.InverseFast(ifail);
//  block matrix algebra
	SMatrix23 CA = aJac.Sub<SMatrix23>(3, 0)
			* aJac.Sub<SMatrix33>(0, 0).InverseFast(ifail); // C*A^-1
	SMatrix22 DCAB = aJac.Sub<SMatrix22>(3, 3) - CA * aJac.Sub<SMatrix32>(0, 3); // D - C*A^-1 *B
	DCAB.InvertFast();
	prevJacobian.Place_at(DCAB, 3, 3);
	prevJacobian.Place_at(-DCAB * CA, 3, 0);
}

/// Define jacobian to next scatterer (by GBLTrajectory constructor)
/**
 * \param [in] aJac Jacobian
 */
void GblPoint::addNextJacobian(const SMatrix55 &aJac) {
	nextJacobian = aJac;
}

/// Retrieve derivatives of local track model
/**
 * Linearized track model: F_u(q/p,u',u) = J*u + S*u' + d*q/p,
 * W is inverse of S, negated for backward propagation.
 * \param [in] aDirection Propagation direction (>0 forward, else backward)
 * \param [out] matW W
 * \param [out] matWJ W*J
 * \param [out] vecWd W*d
 * \exception std::overflow_error : matrix S is singular.
 */
void GblPoint::getDerivatives(int aDirection, SMatrix22 &matW, SMatrix22 &matWJ,
		SVector2 &vecWd) const {

	SMatrix22 matJ;
	SVector2 vecd;
	if (aDirection < 1) {
		matJ = prevJacobian.Sub<SMatrix22>(3, 3);
		matW = -prevJacobian.Sub<SMatrix22>(3, 1);
		vecd = prevJacobian.SubCol<SVector2>(0, 3);
	} else {
		matJ = nextJacobian.Sub<SMatrix22>(3, 3);
		matW = nextJacobian.Sub<SMatrix22>(3, 1);
		vecd = nextJacobian.SubCol<SVector2>(0, 3);
	}

	if (!matW.InvertFast()) {
		std::cout << " GblPoint::getDerivatives failed to invert matrix: "
				<< matW << std::endl;
		std::cout
				<< " Possible reason for singular matrix: multiple GblPoints at same arc-length"
				<< std::endl;
		throw std::overflow_error("Singular matrix inversion exception");
	}
	matWJ = matW * matJ;
	vecWd = matW * vecd;

}

/// Print GblPoint
/**
 * \param [in] level print level (0: minimum, >0: more)
 */
void GblPoint::printPoint(unsigned int level) const {
	std::cout << " GblPoint";
	if (theLabel) {
		std::cout << ", label " << theLabel;
		if (theOffset >= 0) {
			std::cout << ", offset " << theOffset;
		}
	}
	if (measDim) {
		std::cout << ", " << measDim << " measurements";
	}
	if (scatFlag) {
		std::cout << ", scatterer";
	}
	if (transFlag) {
		std::cout << ", diagonalized";
	}
	if (localDerivatives.GetNcols()) {
		std::cout << ", " << localDerivatives.GetNcols()
				<< " local derivatives";
	}
	if (globalDerivatives.GetNcols()) {
		std::cout << ", " << globalDerivatives.GetNcols()
				<< " global derivatives";
	}
	std::cout << std::endl;
	if (level > 0) {
		if (measDim) {
			std::cout << "  Measurement" << std::endl;
			std::cout << "   Projection: " << std::endl << measProjection
					<< std::endl;
			std::cout << "   Residuals: " << measResiduals << std::endl;
			std::cout << "   Precision: " << measPrecision << std::endl;
		}
		if (scatFlag) {
			std::cout << "  Scatterer" << std::endl;
			std::cout << "   Residuals: " << scatResiduals << std::endl;
			std::cout << "   Precision: " << scatPrecision << std::endl;
		}
		if (localDerivatives.GetNcols()) {
			std::cout << "  Local Derivatives:" << std::endl;
			localDerivatives.Print();
		}
		if (globalDerivatives.GetNcols()) {
			std::cout << "  Global Labels:";
			for (unsigned int i = 0; i < globalLabels.size(); ++i) {
				std::cout << " " << globalLabels[i];
			}
			std::cout << std::endl;
			std::cout << "  Global Derivatives:" << std::endl;
			globalDerivatives.Print();
		}
		std::cout << "  Jacobian " << std::endl;
		std::cout << "   Point-to-point " << std::endl << p2pJacobian
				<< std::endl;
		if (theLabel) {
			std::cout << "   To previous offset " << std::endl << prevJacobian
					<< std::endl;
			std::cout << "   To next offset " << std::endl << nextJacobian
					<< std::endl;
		}
	}
}

}
