/*
 * VMatrix.cpp
 *
 *  Created on: Feb 15, 2012
 *      Author: kleinwrt
 */

#include "Alignment/ReferenceTrajectories/interface/VMatrix.h"

//! Namespace for the general broken lines package
namespace gbl {

/*********** simple Matrix based on std::vector<double> **********/

VMatrix::VMatrix(const unsigned int nRows, const unsigned int nCols) :
		numRows(nRows), numCols(nCols), theVec(nRows * nCols) {
}

VMatrix::VMatrix(const VMatrix &aMatrix) :
		numRows(aMatrix.numRows), numCols(aMatrix.numCols), theVec(
				aMatrix.theVec) {

}

VMatrix::~VMatrix() {
}

/// Resize Matrix.
/**
 * \param [in] nRows Number of rows.
 * \param [in] nCols Number of columns.
 */
void VMatrix::resize(const unsigned int nRows, const unsigned int nCols) {
	numRows = nRows;
	numCols = nCols;
	theVec.resize(nRows * nCols);
}

/// Get transposed matrix.
/**
 * \return Transposed matrix.
 */
VMatrix VMatrix::transpose() const {
	VMatrix aResult(numCols, numRows);
	for (unsigned int i = 0; i < numRows; ++i) {
		for (unsigned int j = 0; j < numCols; ++j) {
			aResult(j, i) = theVec[numCols * i + j];
		}
	}
	return aResult;
}

/// Get number of rows.
/**
 * \return Number of rows.
 */
unsigned int VMatrix::getNumRows() const {
	return numRows;
}

/// Get number of columns.
/**
 * \return Number of columns.
 */
unsigned int VMatrix::getNumCols() const {
	return numCols;
}

/// Print matrix.
void VMatrix::print() const {
	std::cout << " VMatrix: " << numRows << "*" << numCols << std::endl;
	for (unsigned int i = 0; i < numRows; ++i) {
		for (unsigned int j = 0; j < numCols; ++j) {
			if (j % 5 == 0) {
				std::cout << std::endl << std::setw(4) << i << ","
						<< std::setw(4) << j << "-" << std::setw(4)
						<< std::min(j + 4, numCols) << ":";
			}
			std::cout << std::setw(13) << theVec[numCols * i + j];
		}
	}
	std::cout << std::endl << std::endl;
}

/// Multiplication Matrix*Vector.
VVector VMatrix::operator*(const VVector &aVector) const {
	VVector aResult(numRows);
	for (unsigned int i = 0; i < this->numRows; ++i) {
		double sum = 0.0;
		for (unsigned int j = 0; j < this->numCols; ++j) {
			sum += theVec[numCols * i + j] * aVector(j);
		}
		aResult(i) = sum;
	}
	return aResult;
}

/// Multiplication Matrix*Matrix.
VMatrix VMatrix::operator*(const VMatrix &aMatrix) const {

	VMatrix aResult(numRows, aMatrix.numCols);
	for (unsigned int i = 0; i < numRows; ++i) {
		for (unsigned int j = 0; j < aMatrix.numCols; ++j) {
			double sum = 0.0;
			for (unsigned int k = 0; k < numCols; ++k) {
				sum += theVec[numCols * i + k] * aMatrix(k, j);
			}
			aResult(i, j) = sum;
		}
	}
	return aResult;
}

/// Addition Matrix+Matrix.
VMatrix VMatrix::operator+(const VMatrix &aMatrix) const {
	VMatrix aResult(numRows, numCols);
	for (unsigned int i = 0; i < numRows; ++i) {
		for (unsigned int j = 0; j < numCols; ++j) {
			aResult(i, j) = theVec[numCols * i + j] + aMatrix(i, j);
		}
	}
	return aResult;
}

/// Assignment Matrix=Matrix.
VMatrix &VMatrix::operator=(const VMatrix &aMatrix) {
	if (this != &aMatrix) {   // Gracefully handle self assignment
		numRows = aMatrix.getNumRows();
		numCols = aMatrix.getNumCols();
		theVec.resize(numRows * numCols);
		for (unsigned int i = 0; i < numRows; ++i) {
			for (unsigned int j = 0; j < numCols; ++j) {
				theVec[numCols * i + j] = aMatrix(i, j);
			}
		}
	}
	return *this;
}

/*********** simple symmetric Matrix based on std::vector<double> **********/

VSymMatrix::VSymMatrix(const unsigned int nRows) :
		numRows(nRows), theVec((nRows * nRows + nRows) / 2) {
}

VSymMatrix::~VSymMatrix() {
}

/// Resize symmetric matrix.
/**
 * \param [in] nRows Number of rows.
 */
void VSymMatrix::resize(const unsigned int nRows) {
	numRows = nRows;
	theVec.resize((nRows * nRows + nRows) / 2);
}

/// Get number of rows (= number of colums).
/**
 * \return Number of rows.
 */
unsigned int VSymMatrix::getNumRows() const {
	return numRows;
}

/// Print matrix.
void VSymMatrix::print() const {
	std::cout << " VSymMatrix: " << numRows << "*" << numRows << std::endl;
	for (unsigned int i = 0; i < numRows; ++i) {
		for (unsigned int j = 0; j <= i; ++j) {
			if (j % 5 == 0) {
				std::cout << std::endl << std::setw(4) << i << ","
						<< std::setw(4) << j << "-" << std::setw(4)
						<< std::min(j + 4, i) << ":";
			}
			std::cout << std::setw(13) << theVec[(i * i + i) / 2 + j];
		}
	}
	std::cout << std::endl << std::endl;
}

/// Subtraction SymMatrix-(sym)Matrix.
VSymMatrix VSymMatrix::operator-(const VMatrix &aMatrix) const {
	VSymMatrix aResult(numRows);
	for (unsigned int i = 0; i < numRows; ++i) {
		for (unsigned int j = 0; j <= i; ++j) {
			aResult(i, j) = theVec[(i * i + i) / 2 + j] - aMatrix(i, j);
		}
	}
	return aResult;
}

/// Multiplication SymMatrix*Vector.
VVector VSymMatrix::operator*(const VVector &aVector) const {
	VVector aResult(numRows);
	for (unsigned int i = 0; i < numRows; ++i) {
		aResult(i) = theVec[(i * i + i) / 2 + i] * aVector(i);
		for (unsigned int j = 0; j < i; ++j) {
			aResult(j) += theVec[(i * i + i) / 2 + j] * aVector(i);
			aResult(i) += theVec[(i * i + i) / 2 + j] * aVector(j);
		}
	}
	return aResult;
}

/// Multiplication SymMatrix*Matrix.
VMatrix VSymMatrix::operator*(const VMatrix &aMatrix) const {
	unsigned int nCol = aMatrix.getNumCols();
	VMatrix aResult(numRows, nCol);
	for (unsigned int l = 0; l < nCol; ++l) {
		for (unsigned int i = 0; i < numRows; ++i) {
			aResult(i, l) = theVec[(i * i + i) / 2 + i] * aMatrix(i, l);
			for (unsigned int j = 0; j < i; ++j) {
				aResult(j, l) += theVec[(i * i + i) / 2 + j] * aMatrix(i, l);
				aResult(i, l) += theVec[(i * i + i) / 2 + j] * aMatrix(j, l);
			}
		}
	}
	return aResult;
}

/*********** simple Vector based on std::vector<double> **********/

VVector::VVector(const unsigned int nRows) :
		numRows(nRows), theVec(nRows) {
}

VVector::VVector(const VVector &aVector) :
		numRows(aVector.numRows), theVec(aVector.theVec) {

}

VVector::~VVector() {
}

/// Resize vector.
/**
 * \param [in] nRows Number of rows.
 */
void VVector::resize(const unsigned int nRows) {
	numRows = nRows;
	theVec.resize(nRows);
}

/// Get part of vector.
/**
 * \param [in] len Length of part.
 * \param [in] start Offset of part.
 * \return Part of vector.
 */
VVector VVector::getVec(unsigned int len, unsigned int start) const {
	VVector aResult(len);
	std::memcpy(&aResult.theVec[0], &theVec[start], sizeof(double) * len);
	return aResult;
}

/// Put part of vector.
/**
 * \param [in] aVector Vector with part.
 * \param [in] start Offset of part.
 */
void VVector::putVec(const VVector &aVector, unsigned int start) {
	std::memcpy(&theVec[start], &aVector.theVec[0],
			sizeof(double) * aVector.numRows);
}

/// Get number of rows.
/**
 * \return Number of rows.
 */
unsigned int VVector::getNumRows() const {
	return numRows;
}

/// Print vector.
void VVector::print() const {
	std::cout << " VVector: " << numRows << std::endl;
	for (unsigned int i = 0; i < numRows; ++i) {

		if (i % 5 == 0) {
			std::cout << std::endl << std::setw(4) << i << "-" << std::setw(4)
					<< std::min(i + 4, numRows) << ":";
		}
		std::cout << std::setw(13) << theVec[i];
	}
	std::cout << std::endl << std::endl;
}

/// Subtraction Vector-Vector.
VVector VVector::operator-(const VVector &aVector) const {
	VVector aResult(numRows);
	for (unsigned int i = 0; i < numRows; ++i) {
		aResult(i) = theVec[i] - aVector(i);
	}
	return aResult;
}

/// Assignment Vector=Vector.
VVector &VVector::operator=(const VVector &aVector) {
	if (this != &aVector) {   // Gracefully handle self assignment
		numRows = aVector.getNumRows();
		theVec.resize(numRows);
		for (unsigned int i = 0; i < numRows; ++i) {
			theVec[i] = aVector(i);
		}
	}
	return *this;
}

/*============================================================================
 from mpnum.F (MillePede-II by V. Blobel, Univ. Hamburg)
 ============================================================================*/
/// Matrix inversion.
/**
 *     Invert symmetric N-by-N matrix V in symmetric storage mode
 *               V(1) = V11, V(2) = V12, V(3) = V22, V(4) = V13, . . .
 *               replaced by inverse matrix
 *
 *     Method of solution is by elimination selecting the  pivot  on  the
 *     diagonal each stage. The rank of the matrix is returned in  NRANK.
 *     For NRANK ne N, all remaining  rows  and  cols  of  the  resulting
 *     matrix V are  set  to  zero.
 *  \exception 1 : matrix is singular.
 *  \return Rank of matrix.
 */
unsigned int VSymMatrix::invert() {

	const double eps = 1.0E-10;
	unsigned int aSize = numRows;
	std::vector<int> next(aSize);
	std::vector<double> diag(aSize);
	int nSize = aSize;

	int first = 1;
	for (int i = 1; i <= nSize; ++i) {
		next[i - 1] = i + 1; // set "next" pointer
		diag[i - 1] = fabs(theVec[(i * i + i) / 2 - 1]); // save abs of diagonal elements
	}
	next[aSize - 1] = -1; // end flag

	unsigned int nrank = 0;
	for (int i = 1; i <= nSize; ++i) { // start of loop
		int k = 0;
		double vkk = 0.0;

		int j = first;
		int previous = 0;
		int last = previous;
		// look for pivot
		while (j > 0) {
			int jj = (j * j + j) / 2 - 1;
			if (fabs(theVec[jj]) > std::max(fabs(vkk), eps * diag[j - 1])) {
				vkk = theVec[jj];
				k = j;
				last = previous;
			}
			previous = j;
			j = next[j - 1];
		}
		// pivot found
		if (k > 0) {
			int kk = (k * k + k) / 2 - 1;
			if (last <= 0) {
				first = next[k - 1];
			} else {
				next[last - 1] = next[k - 1];
			}
			next[k - 1] = 0; // index is used, reset
			nrank++; // increase rank and ...

			vkk = 1.0 / vkk;
			theVec[kk] = -vkk;
			int jk = kk - k;
			int jl = -1;
			for (int j = 1; j <= nSize; ++j) { // elimination
				if (j == k) {
					jk = kk;
					jl += j;
				} else {
					if (j < k) {
						++jk;
					} else {
						jk += j - 1;
					}

					double vjk = theVec[jk];
					theVec[jk] = vkk * vjk;
					int lk = kk - k;
					if (j >= k) {
						for (int l = 1; l <= k - 1; ++l) {
							++jl;
							++lk;
							theVec[jl] -= theVec[lk] * vjk;
						}
						++jl;
						lk = kk;
						for (int l = k + 1; l <= j; ++l) {
							++jl;
							lk += l - 1;
							theVec[jl] -= theVec[lk] * vjk;
						}
					} else {
						for (int l = 1; l <= j; ++l) {
							++jl;
							++lk;
							theVec[jl] -= theVec[lk] * vjk;
						}
					}
				}
			}
		} else {
			for (int k = 1; k <= nSize; ++k) {
				if (next[k - 1] >= 0) {
					int kk = (k * k - k) / 2 - 1;
					for (int j = 1; j <= nSize; ++j) {
						if (next[j - 1] >= 0) {
							theVec[kk + j] = 0.0; // clear matrix row/col
						}
					}
				}
			}
			throw 1; // singular
		}
	}
	for (int ij = 0; ij < (nSize * nSize + nSize) / 2; ++ij) {
		theVec[ij] = -theVec[ij]; // finally reverse sign of all matrix elements
	}
	return nrank;
}
}
