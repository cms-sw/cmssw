/*
 * VMatrix.h
 *
 *  Created on: Feb 15, 2012
 *      Author: kleinwrt
 */

/** \file
 *  VMatrix definition.
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

#ifndef VMATRIX_H_
#define VMATRIX_H_

#include<iostream>
#include<iomanip>
#include<vector>
#include<cstring>
#include<math.h>

//! Namespace for the general broken lines package
namespace gbl {

/// Simple Vector based on std::vector<double>
class VVector {
public:
	VVector(const unsigned int nRows = 0);
	VVector(const VVector &aVector);
	virtual ~VVector();
	void resize(const unsigned int nRows);
	VVector getVec(unsigned int len, unsigned int start = 0) const;
	void putVec(const VVector &aVector, unsigned int start = 0);
	inline double &operator()(unsigned int i);
	inline double operator()(unsigned int i) const;
	unsigned int getNumRows() const;
	void print() const;
	VVector operator-(const VVector &aVector) const;
	VVector &operator=(const VVector &aVector);
private:
	unsigned int numRows; ///< Number of rows
	std::vector<double> theVec; ///< Data
};

/// Simple Matrix based on std::vector<double>
class VMatrix {
public:
	VMatrix(const unsigned int nRows = 0, const unsigned int nCols = 0);
	VMatrix(const VMatrix &aMatrix);
	virtual ~VMatrix();
	void resize(const unsigned int nRows, const unsigned int nCols);
	VMatrix transpose() const;
	inline double &operator()(unsigned int i, unsigned int j);
	inline double operator()(unsigned int i, unsigned int j) const;
	unsigned int getNumRows() const;
	unsigned int getNumCols() const;
	void print() const;
	VVector operator*(const VVector &aVector) const;
	VMatrix operator*(const VMatrix &aMatrix) const;
	VMatrix operator+(const VMatrix &aMatrix) const;
	VMatrix &operator=(const VMatrix &aMatrix);
private:
	unsigned int numRows; ///< Number of rows
	unsigned int numCols; ///< Number of columns
	std::vector<double> theVec; ///< Data
};

/// Simple symmetric Matrix based on std::vector<double>
class VSymMatrix {
public:
	VSymMatrix(const unsigned int nRows = 0);
	virtual ~VSymMatrix();
	void resize(const unsigned int nRows);
	unsigned int invert();
	inline double &operator()(unsigned int i, unsigned int j);
	inline double operator()(unsigned int i, unsigned int j) const;
	unsigned int getNumRows() const;
	void print() const;
	VSymMatrix operator-(const VMatrix &aMatrix) const;
	VVector operator*(const VVector &aVector) const;
	VMatrix operator*(const VMatrix &aMatrix) const;
private:
	unsigned int numRows; ///< Number of rows
	std::vector<double> theVec; ///< Data (symmetric storage)
};

/// access element (i,j)
inline double &VMatrix::operator()(unsigned int iRow, unsigned int iCol) {
	return theVec[numCols * iRow + iCol];
}

/// access element (i,j)
inline double VMatrix::operator()(unsigned int iRow, unsigned int iCol) const {
	return theVec[numCols * iRow + iCol];
}

/// access element (i)
inline double &VVector::operator()(unsigned int iRow) {
	return theVec[iRow];
}

/// access element (i)
inline double VVector::operator()(unsigned int iRow) const {
	return theVec[iRow];
}

/// access element (i,j) assuming i>=j
inline double &VSymMatrix::operator()(unsigned int iRow, unsigned int iCol) {
	return theVec[(iRow * iRow + iRow) / 2 + iCol]; // assuming iCol <= iRow
}

/// access element (i,j) assuming i>=j
inline double VSymMatrix::operator()(unsigned int iRow,
		unsigned int iCol) const {
	return theVec[(iRow * iRow + iRow) / 2 + iCol]; // assuming iCol <= iRow
}
}
#endif /* VMATRIX_H_ */
