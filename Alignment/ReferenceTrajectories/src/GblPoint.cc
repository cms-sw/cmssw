/*
 * GblPoint.cpp
 *
 *  Created on: Aug 18, 2011
 *      Author: kleinwrt
 */

/** \file
 *  GblPoint methods.
 *
 *  \author Claus Kleinwort, DESY, 2011 (Claus.Kleinwort@desy.de)
 *
 *  \copyright
 *  Copyright (c) 2011 - 2016 Deutsches Elektronen-Synchroton,
 *  Member of the Helmholtz Association, (DESY), HAMBURG, GERMANY \n\n
 *  This library is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU Library General Public License as
 *  published by the Free Software Foundation; either version 2 of the
 *  License, or (at your option) any later version. \n\n
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Library General Public License for more details. \n\n
 *  You should have received a copy of the GNU Library General Public
 *  License along with this program (see the file COPYING.LIB for more
 *  details); if not, write to the Free Software Foundation, Inc.,
 *  675 Mass Ave, Cambridge, MA 02139, USA.
 */

#include "Alignment/ReferenceTrajectories/interface/GblPoint.h"
using namespace Eigen;

//! Namespace for the general broken lines package
namespace gbl {

/// Create a point.
/**
 * Create point on (initial) trajectory. Needs transformation jacobian from previous point.
 * \param [in] aJacobian Transformation jacobian from previous point
 */
GblPoint::GblPoint(const Matrix5d &aJacobian) :
		theLabel(0), theOffset(0), p2pJacobian(aJacobian), measDim(0), measPrecMin(
				0.), transFlag(false), measTransformation(), scatFlag(false), localDerivatives(), globalLabels(), globalDerivatives() {

}

#ifdef GBL_EIGEN_SUPPORT_ROOT
/// Create a point.
/**
 * Create point on (initial) trajectory. Needs transformation jacobian from previous point.
 * \param [in] aJacobian Transformation jacobian from previous point
 */
GblPoint::GblPoint(const TMatrixD &aJacobian) :
		theLabel(0), theOffset(0), measDim(0), measPrecMin(0.), transFlag(
				false), measTransformation(), scatFlag(false), localDerivatives(), globalLabels(), globalDerivatives() {

	for (unsigned int i = 0; i < 5; ++i) {
		for (unsigned int j = 0; j < 5; ++j) {
			p2pJacobian(i, j) = aJacobian(i, j);
		}
	}
}
#endif

GblPoint::~GblPoint() {
}

#ifdef GBL_EIGEN_SUPPORT_ROOT
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
	measPrecMin = minPrecision;
	unsigned int iOff = 5 - measDim;
	for (unsigned int i = 0; i < measDim; ++i) {
		measResiduals(iOff + i) = aResiduals[i];
		measPrecision(iOff + i) = aPrecision[i];
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
	measPrecMin = minPrecision;
	TMatrixDSymEigen measEigen(aPrecision);
	TMatrixD tmpTransformation(measDim, measDim);
	tmpTransformation = measEigen.GetEigenVectors();
	tmpTransformation.T();
	transFlag = true;
	TVectorD transResiduals = tmpTransformation * aResiduals;
	TVectorD transPrecision = measEigen.GetEigenValues();
	TMatrixD transProjection = tmpTransformation * aProjection;
	measTransformation.resize(measDim, measDim);
	unsigned int iOff = 5 - measDim;
	for (unsigned int i = 0; i < measDim; ++i) {
		measResiduals(iOff + i) = transResiduals[i];
		measPrecision(iOff + i) = transPrecision[i];
		for (unsigned int j = 0; j < measDim; ++j) {
			measTransformation(i, j) = tmpTransformation(i, j);
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
	measPrecMin = minPrecision;
	unsigned int iOff = 5 - measDim;
	for (unsigned int i = 0; i < measDim; ++i) {
		measResiduals(iOff + i) = aResiduals[i];
		measPrecision(iOff + i) = aPrecision[i];
	}
	measProjection.setIdentity();
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
	measPrecMin = minPrecision;
	TMatrixDSymEigen measEigen(aPrecision);
	TMatrixD tmpTransformation(measDim, measDim);
	tmpTransformation = measEigen.GetEigenVectors();
	tmpTransformation.T();
	transFlag = true;
	TVectorD transResiduals = tmpTransformation * aResiduals;
	TVectorD transPrecision = measEigen.GetEigenValues();
	measTransformation.resize(measDim, measDim);
	unsigned int iOff = 5 - measDim;
	for (unsigned int i = 0; i < measDim; ++i) {
		measResiduals(iOff + i) = transResiduals[i];
		measPrecision(iOff + i) = transPrecision[i];
		for (unsigned int j = 0; j < measDim; ++j) {
			measTransformation(i, j) = tmpTransformation(i, j);
			measProjection(iOff + i, iOff + j) = measTransformation(i, j);
		}
	}
}
#endif

/// Check for measurement at a point.
/**
 * Get dimension of measurement (0 = none).
 * \return measurement dimension
 */
unsigned int GblPoint::hasMeasurement() const {
	return measDim;
}

/// get precision cutoff.
/**
 * \return minimal measurement precision (for usage)
 */
double GblPoint::getMeasPrecMin() const {
	return measPrecMin;
}

/// Retrieve measurement of a point.
/**
 * \param [out] aProjection Projection from (diagonalized) measurement to local system
 * \param [out] aResiduals Measurement residuals
 * \param [out] aPrecision Measurement precision (diagonal)
 */
void GblPoint::getMeasurement(Matrix5d &aProjection, Vector5d &aResiduals,
		Vector5d &aPrecision) const {
	aProjection.bottomRightCorner(measDim, measDim) =
			measProjection.bottomRightCorner(measDim, measDim);
	aResiduals.tail(measDim) = measResiduals.tail(measDim);
	aPrecision.tail(measDim) = measPrecision.tail(measDim);
}

/// Get measurement transformation (from diagonalization).
/**
 * \param [out] aTransformation Transformation matrix
 */
void GblPoint::getMeasTransformation(MatrixXd &aTransformation) const {
	aTransformation.resize(measDim, measDim);
	if (transFlag) {
		aTransformation = measTransformation;
	} else {
		aTransformation.setIdentity();
	}
}

#ifdef GBL_EIGEN_SUPPORT_ROOT
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
	scatTransformation.setIdentity();
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
	scatTransformation.resize(2, 2);
	for (unsigned int i = 0; i < 2; ++i) {
		scatResiduals(i) = transResiduals[i];
		scatPrecision(i) = transPrecision[i];
		for (unsigned int j = 0; j < 2; ++j) {
			scatTransformation(i, j) = aTransformation(i, j);
		}
	}
}
#endif

/// Add a (thin) scatterer to a point.
/**
 * Add scatterer with diagonal or arbitrary precision (inverse covariance) matrix.
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
 * \param [in] aPrecision Scatterer precision (vector (with diagonal) or (full) matrix)
 */
void GblPoint::addScatterer(const Vector2d &aResiduals,
		const Vector2d& aPrecision) {
	scatFlag = true;
	scatResiduals = aResiduals;
	scatPrecision = aPrecision;
	scatTransformation.setIdentity();
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
void GblPoint::getScatterer(Matrix2d &aTransformation, Vector2d &aResiduals,
		Vector2d &aPrecision) const {
	aTransformation = scatTransformation;
	aResiduals = scatResiduals;
	aPrecision = scatPrecision;
}

/// Get scatterer transformation (from diagonalization).
/**
 * \param [out] aTransformation Transformation matrix
 */
void GblPoint::getScatTransformation(Matrix2d &aTransformation) const {
	if (scatFlag) {
		aTransformation = scatTransformation;
	} else {
		aTransformation.setIdentity();
	}
}

#ifdef GBL_EIGEN_SUPPORT_ROOT
/// Add local derivatives to a point.
/**
 * Point needs to have a measurement.
 * \param [in] aDerivatives Local derivatives (matrix)
 */
void GblPoint::addLocals(const TMatrixD &aDerivatives) {
	if (measDim) {
		unsigned int numDer = aDerivatives.GetNcols();
		localDerivatives.resize(measDim, numDer);
		// convert from ROOT
		MatrixXd tmpDerivatives(measDim, numDer);
		for (unsigned int i = 0; i < measDim; ++i) {
			for (unsigned int j = 0; j < numDer; ++j)
				tmpDerivatives(i, j) = aDerivatives(i, j);
		}
		if (transFlag) {
			localDerivatives = measTransformation * tmpDerivatives;
		} else {
			localDerivatives = tmpDerivatives;
		}
	}
}
#endif

/// Retrieve number of local derivatives from a point.
unsigned int GblPoint::getNumLocals() const {
	return localDerivatives.cols();
}

/// Retrieve local derivatives from a point.
const MatrixXd& GblPoint::getLocalDerivatives() const {
	return localDerivatives;
}

#ifdef GBL_EIGEN_SUPPORT_ROOT
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
		unsigned int numDer = aDerivatives.GetNcols();
		globalDerivatives.resize(measDim, numDer);
		// convert from ROOT
		MatrixXd tmpDerivatives(measDim, numDer);
		for (unsigned int i = 0; i < measDim; ++i) {
			for (unsigned int j = 0; j < numDer; ++j)
				tmpDerivatives(i, j) = aDerivatives(i, j);
		}
		if (transFlag) {
			globalDerivatives = measTransformation * tmpDerivatives;
		} else {
			globalDerivatives = tmpDerivatives;
		}

	}
}
#endif

/// Retrieve number of global derivatives from a point.
unsigned int GblPoint::getNumGlobals() const {
	return globalDerivatives.cols();
}

/// Retrieve global derivatives labels from a point.
/**
 * \param [out] aLabels Global labels
 */
void GblPoint::getGlobalLabels(std::vector<int> &aLabels) const {
	aLabels = globalLabels;
}

/// Retrieve global derivatives from a point.
/**
 * \param [out] aDerivatives  Global derivatives
 */
void GblPoint::getGlobalDerivatives(MatrixXd &aDerivatives) const {
	aDerivatives = globalDerivatives;
}

/// Retrieve global derivatives from a point for a single row.
/**
 * \param [in] aRow  Row number
 * \param [out] aLabels Global labels
 * \param [out] aDerivatives  Global derivatives
 */
void GblPoint::getGlobalLabelsAndDerivatives(unsigned int aRow,
		std::vector<int> &aLabels, std::vector<double> &aDerivatives) const {
	aLabels.resize(globalDerivatives.cols());
	aDerivatives.resize(globalDerivatives.cols());
	for (unsigned int i = 0; i < globalDerivatives.cols(); ++i) {
		aLabels[i] = globalLabels[i];
		aDerivatives[i] = globalDerivatives(aRow, i);
	}
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
const Matrix5d& GblPoint::getP2pJacobian() const {
	return p2pJacobian;
}

/// Define jacobian to previous scatterer (by GBLTrajectory constructor)
/**
 * \param [in] aJac Jacobian
 */
void GblPoint::addPrevJacobian(const Matrix5d &aJac) {
// to optimize: need only two last rows of inverse
//	prevJacobian = aJac.inverse();
//  block matrix algebra
	Matrix23d CA = aJac.block<2, 3>(3, 0) * aJac.block<3, 3>(0, 0).inverse(); // C*A^-1
	Matrix2d DCAB = aJac.block<2, 2>(3, 3) - CA * aJac.block<3, 2>(0, 3); // D - C*A^-1 *B
	Matrix2d DCABInv = DCAB.inverse();
	prevJacobian.block<2, 2>(3, 3) = DCABInv;
	prevJacobian.block<2, 3>(3, 0) = -DCABInv * CA;
}

/// Define jacobian to next scatterer (by GBLTrajectory constructor)
/**
 * \param [in] aJac Jacobian
 */
void GblPoint::addNextJacobian(const Matrix5d &aJac) {
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
void GblPoint::getDerivatives(int aDirection, Matrix2d &matW, Matrix2d &matWJ,
		Vector2d &vecWd) const {

	Matrix2d matJ;
	Vector2d vecd;
	if (aDirection < 1) {
		matJ = prevJacobian.block<2, 2>(3, 3);
		matW = -prevJacobian.block<2, 2>(3, 1);
		vecd = prevJacobian.block<2, 1>(3, 0);
	} else {
		matJ = nextJacobian.block<2, 2>(3, 3);
		matW = nextJacobian.block<2, 2>(3, 1);
		vecd = nextJacobian.block<2, 1>(3, 0);
	}

	if (!matW.determinant()) {
		std::cout << " GblPoint::getDerivatives failed to invert matrix "
				<< std::endl;
		std::cout
				<< " Possible reason for singular matrix: multiple GblPoints at same arc-length"
				<< std::endl;
		throw std::overflow_error("Singular matrix inversion exception");
	}
	matW = matW.inverse().eval();
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
	if (localDerivatives.cols()) {
		std::cout << ", " << localDerivatives.cols() << " local derivatives";
	}
	if (globalDerivatives.cols()) {
		std::cout << ", " << globalDerivatives.cols() << " global derivatives";
	}
	std::cout << std::endl;
	if (level > 0) {
		IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
		if (measDim) {
			std::cout << "  Measurement" << std::endl;
			std::cout << "   Projection: " << std::endl
					<< measProjection.format(CleanFmt) << std::endl;
			std::cout << "   Residuals: "
					<< measResiduals.transpose().format(CleanFmt) << std::endl;
			std::cout << "   Precision (min.: " << measPrecMin << "): "
					<< measPrecision.transpose().format(CleanFmt) << std::endl;
		}
		if (scatFlag) {
			std::cout << "  Scatterer" << std::endl;
			std::cout << "   Residuals: "
					<< scatResiduals.transpose().format(CleanFmt) << std::endl;
			std::cout << "   Precision: "
					<< scatPrecision.transpose().format(CleanFmt) << std::endl;
		}
		if (localDerivatives.cols()) {
			std::cout << "  Local Derivatives:" << std::endl
					<< localDerivatives.format(CleanFmt) << std::endl;
		}
		if (globalDerivatives.cols()) {
			std::cout << "  Global Labels:";
			for (unsigned int i = 0; i < globalLabels.size(); ++i) {
				std::cout << " " << globalLabels[i];
			}
			std::cout << std::endl;
			std::cout << "  Global Derivatives:"
					<< globalDerivatives.format(CleanFmt) << std::endl;
		}
		std::cout << "  Jacobian " << std::endl;
		std::cout << "   Point-to-point " << std::endl
				<< p2pJacobian.format(CleanFmt) << std::endl;
		if (theLabel) {
			std::cout << "   To previous offset " << std::endl
					<< prevJacobian.format(CleanFmt) << std::endl;
			std::cout << "   To next offset " << std::endl
					<< nextJacobian.format(CleanFmt) << std::endl;
		}
	}
}

}
