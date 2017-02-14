/*
 * GblData.h
 *
 *  Created on: Aug 18, 2011
 *      Author: kleinwrt
 */

/** \file
 *  GblData definition.
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

#ifndef GBLDATA_H_
#define GBLDATA_H_

#include<iostream>
#include<vector>
#include<math.h>
#include "Alignment/ReferenceTrajectories/interface/VMatrix.h"

#include "Eigen/Core"
typedef Eigen::Matrix<double, 5, 5> Matrix5d;
typedef Eigen::Matrix<double, 2, 7> Matrix27d;

//! Namespace for the general broken lines package
namespace gbl {

enum dataBlockType {
	None, InternalMeasurement, InternalKink, ExternalSeed, ExternalMeasurement
};

/// Data (block) for independent scalar measurement
/**
 * Data (block) containing value, precision and derivatives for measurements, kinks and seeds.
 * Created from attributes of GblPoints, used to construct linear equation system for track fit.
 */
class GblData {
public:
	GblData(unsigned int aLabel, dataBlockType aType, double aMeas,
			double aPrec, unsigned int aTraj = 0, unsigned int aPoint = 0);
	virtual ~GblData();
	void addDerivatives(unsigned int iRow,
			const std::vector<unsigned int> &labDer, const Matrix5d &matDer,
			unsigned int iOff, const Eigen::MatrixXd &derLocal, unsigned int nLocal,
			const Eigen::MatrixXd &derTrans);
	void addDerivatives(unsigned int iRow,
			const std::vector<unsigned int> &labDer, const Matrix27d &matDer,
			unsigned int nLocal, const Eigen::MatrixXd &derTrans);
	void addDerivatives(const std::vector<unsigned int> &index,
			const std::vector<double> &derivatives);

	void setPrediction(const VVector &aVector);
	double setDownWeighting(unsigned int aMethod);
	double getChi2() const;
	void printData() const;
	unsigned int getLabel() const;
	dataBlockType getType() const;
	unsigned int getNumSimple() const;
	void getLocalData(double &aValue, double &aWeight, unsigned int &numLocal,
			unsigned int* &indLocal, double* &derLocal);
	void getAllData(double &aValue, double &aErr, unsigned int &numLocal,
			unsigned int* &indLocal, double* &derLocal, unsigned int &aTraj,
			unsigned int &aPoint, unsigned int &aRow);
	void getResidual(double &aResidual, double &aVariance, double &aDownWeight,
			unsigned int &numLocal, unsigned int* &indLocal, double* &derLocal);

private:
	unsigned int theLabel; ///< Label (of corresponding point)
	unsigned int theRow; ///< Row number (of measurement)
	dataBlockType theType; ///< Type (None, InternalMeasurement, InternalKink, ExternalSeed, ExternalMeasurement)
	double theValue; ///< Value (residual)
	double thePrecision; ///< Precision (1/sigma**2)
	unsigned int theTrajectory; ///< Trajectory number
	unsigned int thePoint; ///< Point number (on trajectory)
	double theDownWeight; ///< Down-weighting factor (0-1)
	double thePrediction; ///< Prediction from fit
	// standard local parameters (curvature, offsets), fixed size
	unsigned int theNumLocal; ///< Number of (non zero) local derivatives (max 7 for kinks)
	unsigned int theParameters[7]; ///< List of parameters (with non zero derivatives)
	double theDerivatives[7]; ///< List of derivatives for fit
	// more local parameters, dynamic size
	std::vector<unsigned int> moreParameters; ///< List of fit parameters (with non zero derivatives)
	std::vector<double> moreDerivatives; ///< List of derivatives for fit
};
}
#endif /* GBLDATA_H_ */
