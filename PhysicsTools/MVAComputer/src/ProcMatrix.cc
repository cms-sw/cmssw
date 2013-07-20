// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcMatrix
// 

// Implementation:
//     Variable processor to apply a matrix transformation to the input
//     variables. An n x m matrix applied to n input variables results in
//     m output variables.
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: ProcMatrix.cc,v 1.4 2009/06/03 09:50:14 saout Exp $
//

#include <stdlib.h>

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

using namespace PhysicsTools;

namespace { // anonymous

class ProcMatrix : public VarProcessor {
    public:
	typedef VarProcessor::Registry::Registry<ProcMatrix,
					Calibration::ProcMatrix> Registry;

	ProcMatrix(const char *name,
	           const Calibration::ProcMatrix *calib,
	           const MVAComputer *computer);
	virtual ~ProcMatrix() {}      

	virtual void configure(ConfIterator iter, unsigned int n);
	virtual void eval(ValueIterator iter, unsigned int n) const;
	virtual std::vector<double> deriv(ValueIterator iter,
	                                  unsigned int n) const;

    private:
	class Matrix {
	    public:
		inline Matrix(const Calibration::Matrix *calib) :
			rows(calib->rows), cols(calib->columns),
			coeffs(calib->elements) {}

		inline unsigned int getRows() const { return rows; }
		inline unsigned int getCols() const { return cols; }

		inline double operator () (unsigned int row,
		                           unsigned int col) const
		{ return coeffs[row * cols + col]; }

	    private:
		unsigned int		rows;
		unsigned int		cols;
		std::vector<double>	coeffs;
	};

	Matrix	matrix;
};

static ProcMatrix::Registry registry("ProcMatrix");

ProcMatrix::ProcMatrix(const char *name,
                       const Calibration::ProcMatrix *calib,
                       const MVAComputer *computer) :
	VarProcessor(name, calib, computer),
	matrix(&calib->matrix)
{
}

void ProcMatrix::configure(ConfIterator iter, unsigned int n)
{
	if (n != matrix.getCols())
		return;

	for(unsigned int col = 0; col < matrix.getCols(); col++)
		iter++(Variable::FLAG_NONE);

	for(unsigned int row = 0; row < matrix.getRows(); row++)
		iter << Variable::FLAG_NONE;
}

void ProcMatrix::eval(ValueIterator iter, unsigned int n) const
{
	double *sums = (double*)alloca(matrix.getRows() * sizeof(double));
	for(unsigned int row = 0; row < matrix.getRows(); row++)
		sums[row] = 0.0;

	for(unsigned int col = 0; col < matrix.getCols(); col++) {
		double val = *iter++;
		for(unsigned int row = 0; row < matrix.getRows(); row++)
			sums[row] += matrix(row, col) * val;
	}

	for(unsigned int row = 0; row < matrix.getRows(); row++)
		iter(sums[row]);
}

std::vector<double> ProcMatrix::deriv(ValueIterator iter,
                                      unsigned int n) const
{
	std::vector<double> result;
	result.reserve(matrix.getRows() * matrix.getCols());

	for(unsigned int row = 0; row < matrix.getRows(); row++)
		for(unsigned int col = 0; col < matrix.getCols(); col++)
			result.push_back(matrix(row, col));

	return result;
}

} // anonymous namespace
