/*
 * GblPoint.h
 *
 *  Created on: Aug 18, 2011
 *      Author: kleinwrt
 */

/** \file
 *  GblPoint definition.
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

#ifndef GBLPOINT_H_
#define GBLPOINT_H_

#include<iostream>
#include<vector>
#include<math.h>
#include <stdexcept>
#ifdef GBL_EIGEN_SUPPORT_ROOT
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#endif

#include "Eigen/Dense"
typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<double, 2, 3> Matrix23d;
typedef Eigen::Matrix<double, 2, 5> Matrix25d;
typedef Eigen::Matrix<double, 2, 7> Matrix27d;
typedef Eigen::Matrix<double, 3, 2> Matrix32d;
typedef Eigen::Matrix<double, 5, 5> Matrix5d;

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
	GblPoint(const Matrix5d &aJacobian);
	virtual ~GblPoint();
#ifdef GBL_EIGEN_SUPPORT_ROOT
	// input via ROOT
	GblPoint(const TMatrixD &aJacobian);
	void addMeasurement(const TMatrixD &aProjection, const TVectorD &aResiduals,
			const TVectorD &aPrecision, double minPrecision = 0.);
	void addMeasurement(const TMatrixD &aProjection, const TVectorD &aResiduals,
			const TMatrixDSym &aPrecision, double minPrecision = 0.);
	void addMeasurement(const TVectorD &aResiduals, const TVectorD &aPrecision,
			double minPrecision = 0.);
	void addMeasurement(const TVectorD &aResiduals,
			const TMatrixDSym &aPrecision, double minPrecision = 0.);
	void addScatterer(const TVectorD &aResiduals, const TVectorD &aPrecision);
	void addScatterer(const TVectorD &aResiduals,
			const TMatrixDSym &aPrecision);
	void addLocals(const TMatrixD &aDerivatives);
	void addGlobals(const std::vector<int> &aLabels,
			const TMatrixD &aDerivatives);
#endif
	// input via Eigen
	void addMeasurement(const Eigen::MatrixXd &aProjection, const Eigen::VectorXd &aResiduals,
			const Eigen::MatrixXd &aPrecision, double minPrecision = 0.);
	void addMeasurement(const Eigen::VectorXd &aResiduals, const Eigen::MatrixXd &aPrecision,
			double minPrecision = 0.);
	void addScatterer(const Eigen::Vector2d &aResiduals, const Eigen::MatrixXd &aPrecision);
	//
	unsigned int hasMeasurement() const;
	double getMeasPrecMin() const;
	void getMeasurement(Matrix5d &aProjection, Vector5d &aResiduals,
			Vector5d &aPrecision) const;
	void getMeasTransformation(Eigen::MatrixXd &aTransformation) const;
	bool hasScatterer() const;
	void getScatterer(Eigen::Matrix2d &aTransformation, Eigen::Vector2d &aResiduals,
			Eigen::Vector2d &aPrecision) const;
	void getScatTransformation(Eigen::Matrix2d &aTransformation) const;
	void addLocals(const Eigen::MatrixXd &aDerivatives);
	unsigned int getNumLocals() const;
	const Eigen::MatrixXd& getLocalDerivatives() const;
	void addGlobals(const std::vector<int> &aLabels,
			const Eigen::MatrixXd &aDerivatives);
	unsigned int getNumGlobals() const;
	void getGlobalLabels(std::vector<int> &aLabels) const;
	void getGlobalDerivatives(Eigen::MatrixXd &aDerivatives) const;
	void getGlobalLabelsAndDerivatives(unsigned int aRow,
			std::vector<int> &aLabels, std::vector<double> &aDerivatives) const;
	void setLabel(unsigned int aLabel);
	unsigned int getLabel() const;
	void setOffset(int anOffset);
	int getOffset() const;
	const Matrix5d& getP2pJacobian() const;
	void addPrevJacobian(const Matrix5d &aJac);
	void addNextJacobian(const Matrix5d &aJac);
	void getDerivatives(int aDirection, Eigen::Matrix2d &matW, Eigen::Matrix2d &matWJ,
			Eigen::Vector2d &vecWd) const;
	void printPoint(unsigned int level = 0) const;

private:
	unsigned int theLabel; ///< Label identifying point
	int theOffset; ///< Offset number at point if not negative (else interpolation needed)
	Matrix5d p2pJacobian; ///< Point-to-point jacobian from previous point
	Matrix5d prevJacobian; ///< Jacobian to previous scatterer (or first measurement)
	Matrix5d nextJacobian; ///< Jacobian to next scatterer (or last measurement)
	unsigned int measDim; ///< Dimension of measurement (1-5), 0 indicates absence of measurement
	double measPrecMin; ///< Minimal measurement precision (for usage)
	Matrix5d measProjection; ///< Projection from measurement to local system
	Vector5d measResiduals; ///< Measurement residuals
	Vector5d measPrecision; ///< Measurement precision (diagonal of inverse covariance matrix)
	bool transFlag; ///< Transformation exists?
	Eigen::MatrixXd measTransformation; ///< Transformation of diagonalization (of meas. precision matrix)
	bool scatFlag; ///< Scatterer present?
	Eigen::Matrix2d scatTransformation; ///< Transformation of diagonalization (of scat. precision matrix)
	Eigen::Vector2d scatResiduals; ///< Scattering residuals (initial kinks if iterating)
	Eigen::Vector2d scatPrecision; ///< Scattering precision (diagonal of inverse covariance matrix)
	Eigen::MatrixXd localDerivatives; ///< Derivatives of measurement vs additional local (fit) parameters
	std::vector<int> globalLabels; ///< Labels of global (MP-II) derivatives
	Eigen::MatrixXd globalDerivatives; ///< Derivatives of measurement vs additional global (MP-II) parameters
};
}
#endif /* GBLPOINT_H_ */
