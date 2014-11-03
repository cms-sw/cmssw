/*
 * GblPoint.h
 *
 *  Created on: Aug 18, 2011
 *      Author: kleinwrt
 */

#ifndef GBLPOINT_H_
#define GBLPOINT_H_

#include<iostream>
#include<vector>
#include<math.h>
#include <stdexcept>
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"

#include "Math/SMatrix.h"
#include "Math/SVector.h"
typedef ROOT::Math::SMatrix<double, 2> SMatrix22;
typedef ROOT::Math::SMatrix<double, 2, 3> SMatrix23;
typedef ROOT::Math::SMatrix<double, 2, 5> SMatrix25;
typedef ROOT::Math::SMatrix<double, 2, 7> SMatrix27;
typedef ROOT::Math::SMatrix<double, 3, 2> SMatrix32;
typedef ROOT::Math::SMatrix<double, 3> SMatrix33;
typedef ROOT::Math::SMatrix<double, 5> SMatrix55;
typedef ROOT::Math::SVector<double, 2> SVector2;
typedef ROOT::Math::SVector<double, 5> SVector5;

namespace gbl {

/// Point on trajectory
/**
 * User supplied point on (initial) trajectory.
 *
 * Must have jacobian for propagation from previous point. May have:
 *
 *   -# Measurement (1D - 5D)
 *   -# Scatterer (thin, 2D kinks)
 *   -# Additional local parameters (with derivatives). Fitted together with track parameters.
 *   -# Additional global parameters (with labels and derivatives). Not fitted, only passed
 *      on to (binary) file for fitting with Millepede-II.
 */
class GblPoint {
public:
	GblPoint(const TMatrixD &aJacobian);
	GblPoint(const SMatrix55 &aJacobian);
	virtual ~GblPoint();
	void addMeasurement(const TMatrixD &aProjection, const TVectorD &aResiduals,
			const TVectorD &aPrecision, double minPrecision = 0.);
	void addMeasurement(const TMatrixD &aProjection, const TVectorD &aResiduals,
			const TMatrixDSym &aPrecision, double minPrecision = 0.);
	void addMeasurement(const TVectorD &aResiduals, const TVectorD &aPrecision,
			double minPrecision = 0.);
	void addMeasurement(const TVectorD &aResiduals,
			const TMatrixDSym &aPrecision, double minPrecision = 0.);
	unsigned int hasMeasurement() const;
	void getMeasurement(SMatrix55 &aProjection, SVector5 &aResiduals,
			SVector5 &aPrecision) const;
	void getMeasTransformation(TMatrixD &aTransformation) const;
	void addScatterer(const TVectorD &aResiduals, const TVectorD &aPrecision);
	void addScatterer(const TVectorD &aResiduals,
			const TMatrixDSym &aPrecision);
	bool hasScatterer() const;
	void getScatterer(SMatrix22 &aTransformation, SVector2 &aResiduals,
			SVector2 &aPrecision) const;
	void getScatTransformation(TMatrixD &aTransformation) const;
	void addLocals(const TMatrixD &aDerivatives);
	unsigned int getNumLocals() const;
	const TMatrixD& getLocalDerivatives() const;
	void addGlobals(const std::vector<int> &aLabels,
			const TMatrixD &aDerivatives);
	unsigned int getNumGlobals() const;
	std::vector<int> getGlobalLabels() const;
	const TMatrixD& getGlobalDerivatives() const;
	void setLabel(unsigned int aLabel);
	unsigned int getLabel() const;
	void setOffset(int anOffset);
	int getOffset() const;
	const SMatrix55& getP2pJacobian() const;
	void addPrevJacobian(const SMatrix55 &aJac);
	void addNextJacobian(const SMatrix55 &aJac);
	void getDerivatives(int aDirection, SMatrix22 &matW, SMatrix22 &matWJ,
			SVector2 &vecWd) const;
	void printPoint(unsigned int level = 0) const;

private:
	unsigned int theLabel; ///< Label identifying point
	int theOffset; ///< Offset number at point if not negative (else interpolation needed)
	SMatrix55 p2pJacobian; ///< Point-to-point jacobian from previous point
	SMatrix55 prevJacobian; ///< Jacobian to previous scatterer (or first measurement)
	SMatrix55 nextJacobian; ///< Jacobian to next scatterer (or last measurement)
	unsigned int measDim; ///< Dimension of measurement (1-5), 0 indicates absence of measurement
	SMatrix55 measProjection; ///< Projection from measurement to local system
	SVector5 measResiduals; ///< Measurement residuals
	SVector5 measPrecision; ///< Measurement precision (diagonal of inverse covariance matrix)
	bool transFlag; ///< Transformation exists?
	TMatrixD measTransformation; ///< Transformation of diagonalization (of meas. precision matrix)
	bool scatFlag; ///< Scatterer present?
	SMatrix22 scatTransformation; ///< Transformation of diagonalization (of scat. precision matrix)
	SVector2 scatResiduals; ///< Scattering residuals (initial kinks if iterating)
	SVector2 scatPrecision; ///< Scattering precision (diagonal of inverse covariance matrix)
	TMatrixD localDerivatives; ///< Derivatives of measurement vs additional local (fit) parameters
	std::vector<int> globalLabels; ///< Labels of global (MP-II) derivatives
	TMatrixD globalDerivatives; ///< Derivatives of measurement vs additional global (MP-II) parameters
};
}
#endif /* GBLPOINT_H_ */
