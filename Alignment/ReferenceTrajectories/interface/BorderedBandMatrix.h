/*
 * BorderedBandMatrix.h
 *
 *  Created on: Aug 14, 2011
 *      Author: kleinwrt
 */

/** \file
 *  BorderedBandMatrix definition.
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

#ifndef BORDEREDBANDMATRIX_H_
#define BORDEREDBANDMATRIX_H_

#include<iostream>
#include<vector>
#include<math.h>
#include<cstdlib>
#include "Alignment/ReferenceTrajectories/interface/VMatrix.h"
#include "Eigen/Core"

//! Namespace for the general broken lines package
namespace gbl {

/// (Symmetric) Bordered Band Matrix.
/**
 *  Separate storage of border, mixed and band parts (as vector<double>).
 *
 *\verbatim
 *  Example for matrix size=8 with border size and band width of two
 *
 *     +-                                 -+
 *     |  B11 B12 M13 M14 M15 M16 M17 M18  |
 *     |  B12 B22 M23 M24 M25 M26 M27 M28  |
 *     |  M13 M23 C33 C34 C35  0.  0.  0.  |
 *     |  M14 M24 C34 C44 C45 C46  0.  0.  |
 *     |  M15 M25 C35 C45 C55 C56 C57  0.  |
 *     |  M16 M26  0. C46 C56 C66 C67 C68  |
 *     |  M17 M27  0.  0. C57 C67 C77 C78  |
 *     |  M18 M28  0.  0.  0. C68 C78 C88  |
 *     +-                                 -+
 *
 *  Is stored as::
 *
 *     +-         -+     +-                         -+
 *     |  B11 B12  |     |  M13 M14 M15 M16 M17 M18  |
 *     |  B12 B22  |     |  M23 M24 M25 M26 M27 M28  |
 *     +-         -+     +-                         -+
 *
 *                       +-                         -+
 *                       |  C33 C44 C55 C66 C77 C88  |
 *                       |  C34 C45 C56 C67 C78  0.  |
 *                       |  C35 C46 C57 C68  0.  0.  |
 *                       +-                         -+
 *\endverbatim
 */

class BorderedBandMatrix {
public:
	BorderedBandMatrix();
	virtual ~BorderedBandMatrix();
	void resize(unsigned int nSize, unsigned int nBorder = 1,
			unsigned int nBand = 5);
	void solveAndInvertBorderedBand(const VVector &aRightHandSide,
			VVector &aSolution);
	void addBlockMatrix(double aWeight,
			const std::vector<unsigned int>* anIndex,
			const std::vector<double>* aVector);
	void addBlockMatrix(double aWeight, unsigned int nSimple,
			unsigned int* anIndex, double* aVector);
	Eigen::MatrixXd getBlockMatrix(const std::vector<unsigned int> anIndex) const;
	Eigen::MatrixXd getBlockMatrix(unsigned int aSize, unsigned int* anIndex) const;
	void printMatrix() const;

private:
	unsigned int numSize; ///< Matrix size
	unsigned int numBorder; ///< Border size
	unsigned int numBand; ///< Band width
	unsigned int numCol; ///< Band matrix size
	VSymMatrix theBorder; ///< Border part
	VMatrix theMixed; ///< Mixed part
	VMatrix theBand; ///< Band part

	void decomposeBand();
	VVector solveBand(const VVector &aRightHandSide) const;
	VMatrix solveBand(const VMatrix &aRightHandSide) const;
	VMatrix invertBand();
	VMatrix bandOfAVAT(const VMatrix &anArray,
			const VSymMatrix &aSymArray) const;
};
}
#endif /* BORDEREDBANDMATRIX_H_ */
