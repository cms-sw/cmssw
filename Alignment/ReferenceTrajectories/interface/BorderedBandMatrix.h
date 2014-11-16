/*
 * BorderedBandMatrix.h
 *
 *  Created on: Aug 14, 2011
 *      Author: kleinwrt
 */

#ifndef BORDEREDBANDMATRIX_H_
#define BORDEREDBANDMATRIX_H_

#include<iostream>
#include<vector>
#include<math.h>
#include<cstdlib>
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "VMatrix.h"

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
	TMatrixDSym getBlockMatrix(const std::vector<unsigned int> anIndex) const;
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
