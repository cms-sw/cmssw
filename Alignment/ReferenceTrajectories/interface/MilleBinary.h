/*
 * MilleBinary.h
 *
 *  Created on: Aug 31, 2011
 *      Author: kleinwrt
 */

#ifndef MILLEBINARY_H_
#define MILLEBINARY_H_

#include<fstream>
#include<vector>

//! Namespace for the general broken lines package
namespace gbl {

///  Millepede-II (binary) record.
/**
 *  Containing information for local (track) and global fit.
 *
 *  The data blocks are collected in two arrays, a real array
 *  (containing float or double values) and integer array, of same length.
 *  A positive record length indicate _float_ and a negative one _double_ values.
 *  The content of the record is:
 *\verbatim
 *         real array              integer array
 *     0   0.0                     error count (this record)
 *     1   RMEAS, measured value   0                            -+
 *     2   local derivative        index of local derivative     |
 *     3   local derivative        index of local derivative     |
 *     4    ...                                                  | block
 *         SIGMA, error (>0)       0                             |
 *         global derivative       label of global derivative    |
 *         global derivative       label of global derivative   -+
 *         RMEAS, measured value   0
 *         local derivative        index of local derivative
 *         local derivative        index of local derivative
 *         ...
 *         SIGMA, error            0
 *         global derivative       label of global derivative
 *         global derivative       label of global derivative
 *         ...
 *         global derivative       label of global derivative
 *\endverbatim
 */
class MilleBinary {
public:
	MilleBinary(const std::string fileName = "milleBinaryISN.dat",
			bool doublePrec = false, unsigned int aSize = 2000);
	virtual ~MilleBinary();
	void addData(double aMeas, double aPrec,
			const std::vector<unsigned int> &indLocal,
			const std::vector<double> &derLocal,
			const std::vector<int> &labGlobal,
			const std::vector<double> &derGlobal);
	void writeRecord();

private:
	std::ofstream binaryFile; ///< Binary File
	std::vector<int> intBuffer; ///< Integer buffer
	std::vector<float> floatBuffer; ///< Float buffer
	std::vector<double> doubleBuffer; ///< Double buffer
	bool doublePrecision; ///< Flag for storage in as *double* values
};
}
#endif /* MILLEBINARY_H_ */
