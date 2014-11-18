/*
 * BorderedBandMatrix.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kleinwrt
 */

#include "Alignment/ReferenceTrajectories/interface/BorderedBandMatrix.h"

//! Namespace for the general broken lines package
namespace gbl {

/// Create bordered band matrix.
BorderedBandMatrix::BorderedBandMatrix() : numSize(0), numBorder(0), numBand(0), numCol(0) {
}

BorderedBandMatrix::~BorderedBandMatrix() {
}

/// Resize bordered band matrix.
/**
 * \param nSize [in] Size of matrix
 * \param nBorder [in] Size of border (=1 for q/p + additional local parameters)
 * \param nBand [in] Band width (usually = 5, for simplified jacobians = 4)
 */
void BorderedBandMatrix::resize(unsigned int nSize, unsigned int nBorder,
		unsigned int nBand) {
	numSize = nSize;
	numBorder = nBorder;
	numCol = nSize - nBorder;
	numBand = 0;
	theBorder.resize(numBorder);
	theMixed.resize(numBorder, numCol);
	theBand.resize((nBand + 1), numCol);
}

/// Add symmetric block matrix.
/**
 * Add (extended) block matrix defined by 'aVector * aWeight * aVector.T'
 * to bordered band matrix:
 * BBmatrix(anIndex(i),anIndex(j)) += aVector(i) * aWeight * aVector(j).
 * \param aWeight [in] Weight
 * \param anIndex [in] List of rows/colums to be used
 * \param aVector [in] Vector
 */
void BorderedBandMatrix::addBlockMatrix(double aWeight,
		const std::vector<unsigned int>* anIndex,
		const std::vector<double>* aVector) {
	int nBorder = numBorder;
	for (unsigned int i = 0; i < anIndex->size(); ++i) {
		int iIndex = (*anIndex)[i] - 1; // anIndex has to be sorted
		for (unsigned int j = 0; j <= i; ++j) {
			int jIndex = (*anIndex)[j] - 1;
			if (iIndex < nBorder) {
				theBorder(iIndex, jIndex) += (*aVector)[i] * aWeight
						* (*aVector)[j];
			} else if (jIndex < nBorder) {
				theMixed(jIndex, iIndex - nBorder) += (*aVector)[i] * aWeight
						* (*aVector)[j];
			} else {
				unsigned int nBand = iIndex - jIndex;
				theBand(nBand, jIndex - nBorder) += (*aVector)[i] * aWeight
						* (*aVector)[j];
				numBand = std::max(numBand, nBand); // update band width
			}
		}
	}
}

/// Retrieve symmetric block matrix.
/**
 * Get (compressed) block from bordered band matrix: aMatrix(i,j) = BBmatrix(anIndex(i),anIndex(j)).
 * \param anIndex [in] List of rows/colums to be used
 */
TMatrixDSym BorderedBandMatrix::getBlockMatrix(
		const std::vector<unsigned int> anIndex) const {

	TMatrixDSym aMatrix(anIndex.size());
	int nBorder = numBorder;
	for (unsigned int i = 0; i < anIndex.size(); ++i) {
		int iIndex = anIndex[i] - 1; // anIndex has to be sorted
		for (unsigned int j = 0; j <= i; ++j) {
			int jIndex = anIndex[j] - 1;
			if (iIndex < nBorder) {
				aMatrix(i, j) = theBorder(iIndex, jIndex); // border part of inverse
			} else if (jIndex < nBorder) {
				aMatrix(i, j) = -theMixed(jIndex, iIndex - nBorder); // mixed part of inverse
			} else {
				unsigned int nBand = iIndex - jIndex;
				aMatrix(i, j) = theBand(nBand, jIndex - nBorder); // band part of inverse
			}
			aMatrix(j, i) = aMatrix(i, j);
		}
	}
	return aMatrix;
}

/// Solve linear equation system, partially calculate inverse.
/**
 * Solve linear equation A*x=b system with bordered band matrix A,
 * calculate bordered band part of inverse of A. Use decomposition
 * in border and band part for block matrix algebra:
 *
 *     | A  Ct |   | x1 |   | b1 |        , A  is the border part
 *     |       | * |    | = |    |        , Ct is the mixed part
 *     | C  D  |   | x2 |   | b2 |        , D  is the band part
 *
 * Explicit inversion of D is avoided by using solution X of D*X=C (X=D^-1*C,
 * obtained from Cholesky decomposition and forward/backward substitution)
 *
 *     | x1 |   | E*b1 - E*Xt*b2 |        , E^-1 = A-Ct*D^-1*C = A-Ct*X
 *     |    | = |                |
 *     | x2 |   |  x   - X*x1    |        , x is solution of D*x=b2 (x=D^-1*b2)
 *
 * Inverse matrix is:
 *
 *     |  E   -E*Xt          |
 *     |                     |            , only band part of (D^-1 + X*E*Xt)
 *     | -X*E  D^-1 + X*E*Xt |              is calculated
 *
 *
 * \param [in] aRightHandSide Right hand side (vector) 'b' of A*x=b
 * \param [out] aSolution Solution (vector) x of A*x=b
 */
void BorderedBandMatrix::solveAndInvertBorderedBand(
		const VVector &aRightHandSide, VVector &aSolution) {

	// decompose band
	decomposeBand();
	// invert band
	VMatrix inverseBand = invertBand();
	if (numBorder > 0) { // need to use block matrix decomposition to solve
		// solve for mixed part
		const VMatrix auxMat = solveBand(theMixed); // = Xt
		const VMatrix auxMatT = auxMat.transpose(); // = X
		// solve for border part
		const VVector auxVec = aRightHandSide.getVec(numBorder)
				- auxMat * aRightHandSide.getVec(numCol, numBorder); // = b1 - Xt*b2
		VSymMatrix inverseBorder = theBorder - theMixed * auxMatT;
		inverseBorder.invert(); // = E
		const VVector borderSolution = inverseBorder * auxVec; // = x1
		// solve for band part
		const VVector bandSolution = solveBand(
				aRightHandSide.getVec(numCol, numBorder)); // = x
		aSolution.putVec(borderSolution);
		aSolution.putVec(bandSolution - auxMatT * borderSolution, numBorder); // = x2
		// parts of inverse
		theBorder = inverseBorder; // E
		theMixed = inverseBorder * auxMat; // E*Xt (-mixed part of inverse) !!!
		theBand = inverseBand + bandOfAVAT(auxMatT, inverseBorder); // band(D^-1 + X*E*Xt)
	} else {
		aSolution.putVec(solveBand(aRightHandSide));
		theBand = inverseBand;
	}
}

/// Print bordered band matrix.
void BorderedBandMatrix::printMatrix() const {
	std::cout << "Border part " << std::endl;
	theBorder.print();
	std::cout << "Mixed  part " << std::endl;
	theMixed.print();
	std::cout << "Band   part " << std::endl;
	theBand.print();
}

/*============================================================================
 from Dbandmatrix.F (MillePede-II by V. Blobel, Univ. Hamburg)
 ============================================================================*/
/// (root free) Cholesky decomposition of band part: C=LDL^T
/**
 * Decompose band matrix into diagonal matrix D and lower triangular band matrix
 * L (diagonal=1). Overwrite band matrix with D and off-diagonal part of L.
 *  \exception 2 : matrix is singular.
 *  \exception 3 : matrix is not positive definite.
 */
void BorderedBandMatrix::decomposeBand() {

	int nRow = numBand + 1;
	int nCol = numCol;
	VVector auxVec(nCol);
	for (int i = 0; i < nCol; ++i) {
		auxVec(i) = theBand(0, i) * 16.0; // save diagonal elements
	}
	for (int i = 0; i < nCol; ++i) {
		if ((theBand(0, i) + auxVec(i)) != theBand(0, i)) {
			theBand(0, i) = 1.0 / theBand(0, i);
			if (theBand(0, i) < 0.) {
				throw 3; // not positive definite
			}
		} else {
			theBand(0, i) = 0.0;
			throw 2; // singular
		}
		for (int j = 1; j < std::min(nRow, nCol - i); ++j) {
			double rxw = theBand(j, i) * theBand(0, i);
			for (int k = 0; k < std::min(nRow, nCol - i) - j; ++k) {
				theBand(k, i + j) -= theBand(k + j, i) * rxw;
			}
			theBand(j, i) = rxw;
		}
	}
}

/// Solve for band part.
/**
 * Solve C*x=b for band part using decomposition C=LDL^T
 * and forward (L*z=b) and backward substitution (L^T*x=D^-1*z).
 * \param [in] aRightHandSide Right hand side (vector) 'b' of C*x=b
 * \return Solution (vector) 'x' of C*x=b
 */
VVector BorderedBandMatrix::solveBand(const VVector &aRightHandSide) const {

	int nRow = theBand.getNumRows();
	int nCol = theBand.getNumCols();
	VVector aSolution(aRightHandSide);
	for (int i = 0; i < nCol; ++i) // forward substitution
			{
		for (int j = 1; j < std::min(nRow, nCol - i); ++j) {
			aSolution(j + i) -= theBand(j, i) * aSolution(i);
		}
	}
	for (int i = nCol - 1; i >= 0; i--) // backward substitution
			{
		double rxw = theBand(0, i) * aSolution(i);
		for (int j = 1; j < std::min(nRow, nCol - i); ++j) {
			rxw -= theBand(j, i) * aSolution(j + i);
		}
		aSolution(i) = rxw;
	}
	return aSolution;
}

/// solve band part for mixed part (border rows).
/**
 * Solve C*X=B for mixed part using decomposition C=LDL^T
 * and forward and backward substitution.
 * \param [in] aRightHandSide Right hand side (matrix) 'B' of C*X=B
 * \return Solution (matrix) 'X' of C*X=B
 */
VMatrix BorderedBandMatrix::solveBand(const VMatrix &aRightHandSide) const {

	int nRow = theBand.getNumRows();
	int nCol = theBand.getNumCols();
	VMatrix aSolution(aRightHandSide);
	for (unsigned int iBorder = 0; iBorder < numBorder; iBorder++) {
		for (int i = 0; i < nCol; ++i) // forward substitution
				{
			for (int j = 1; j < std::min(nRow, nCol - i); ++j) {
				aSolution(iBorder, j + i) -= theBand(j, i)
						* aSolution(iBorder, i);
			}
		}
		for (int i = nCol - 1; i >= 0; i--) // backward substitution
				{
			double rxw = theBand(0, i) * aSolution(iBorder, i);
			for (int j = 1; j < std::min(nRow, nCol - i); ++j) {
				rxw -= theBand(j, i) * aSolution(iBorder, j + i);
			}
			aSolution(iBorder, i) = rxw;
		}
	}
	return aSolution;
}

/// Invert band part.
/**
 * \return Inverted band
 */
VMatrix BorderedBandMatrix::invertBand() {

	int nRow = numBand + 1;
	int nCol = numCol;
	VMatrix inverseBand(nRow, nCol);

	for (int i = nCol - 1; i >= 0; i--) {
		double rxw = theBand(0, i);
		for (int j = i; j >= std::max(0, i - nRow + 1); j--) {
			for (int k = j + 1; k < std::min(nCol, j + nRow); ++k) {
				rxw -= inverseBand(abs(i - k), std::min(i, k))
						* theBand(k - j, j);
			}
			inverseBand(i - j, j) = rxw;
			rxw = 0.;
		}
	}
	return inverseBand;
}

/// Calculate band part of: 'anArray * aSymArray * anArray.T'.
/**
 * \return Band part of product
 */
VMatrix BorderedBandMatrix::bandOfAVAT(const VMatrix &anArray,
		const VSymMatrix &aSymArray) const {
	int nBand = numBand;
	int nCol = numCol;
	int nBorder = numBorder;
	double sum;
	VMatrix aBand((nBand + 1), nCol);
	for (int i = 0; i < nCol; ++i) {
		for (int j = std::max(0, i - nBand); j <= i; ++j) {
			sum = 0.;
			for (int l = 0; l < nBorder; ++l) { // diagonal
				sum += anArray(i, l) * aSymArray(l, l) * anArray(j, l);
				for (int k = 0; k < l; ++k) { // off diagonal
					sum += anArray(i, l) * aSymArray(l, k) * anArray(j, k)
							+ anArray(i, k) * aSymArray(l, k) * anArray(j, l);
				}
			}
			aBand(i - j, j) = sum;
		}
	}
	return aBand;
}

}
