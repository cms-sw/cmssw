/*
 * MilleBinary.cpp
 *
 *  Created on: Aug 31, 2011
 *      Author: kleinwrt
 */

#include "Alignment/ReferenceTrajectories/interface/MilleBinary.h"

//! Namespace for the general broken lines package
namespace gbl {

/// Create binary file.
/**
 * \param [in] fileName File name
 * \param [in] doublePrec Flag for storage as double values
 * \param [in] aSize Buffer size
 */
MilleBinary::MilleBinary(const std::string fileName, bool doublePrec,
		unsigned int aSize) :
		binaryFile(fileName.c_str(), std::ios::binary | std::ios::out), intBuffer(), floatBuffer(), doubleBuffer(), doublePrecision(
				doublePrec) {
	intBuffer.reserve(aSize);
	intBuffer.push_back(0); // first word is error counter
	if (doublePrecision) {
		doubleBuffer.reserve(aSize);
		doubleBuffer.push_back(0.);

	} else {
		floatBuffer.reserve(aSize);
		floatBuffer.push_back(0.);
	}
}

MilleBinary::~MilleBinary() {
	binaryFile.close();
}

/// Add data block to (end of) record.
/**
 * \param [in] aMeas Value
 * \param [in] aErr Error
 * \param [in] indLocal List of labels of local parameters
 * \param [in] derLocal List of derivatives for local parameters
 * \param [in] labGlobal List of labels of global parameters
 * \param [in] derGlobal List of derivatives for global parameters
 */
void MilleBinary::addData(double aMeas, double aErr,
		const std::vector<unsigned int> &indLocal,
		const std::vector<double> &derLocal, const std::vector<int> &labGlobal,
		const std::vector<double> &derGlobal) {

	if (doublePrecision) {
		// double values
		intBuffer.push_back(0);
		doubleBuffer.push_back(aMeas);
		for (unsigned int i = 0; i < indLocal.size(); ++i) {
			intBuffer.push_back(indLocal[i]);
			doubleBuffer.push_back(derLocal[i]);
		}
		intBuffer.push_back(0);
		doubleBuffer.push_back(aErr);
		for (unsigned int i = 0; i < labGlobal.size(); ++i) {
			if (derGlobal[i]) {
				intBuffer.push_back(labGlobal[i]);
				doubleBuffer.push_back(derGlobal[i]);
			}
		}
	} else {
		// float values
		intBuffer.push_back(0);
		floatBuffer.push_back(aMeas);
		for (unsigned int i = 0; i < indLocal.size(); ++i) {
			intBuffer.push_back(indLocal[i]);
			floatBuffer.push_back(derLocal[i]);
		}
		intBuffer.push_back(0);
		floatBuffer.push_back(aErr);
		for (unsigned int i = 0; i < labGlobal.size(); ++i) {
			if (derGlobal[i]) {
				intBuffer.push_back(labGlobal[i]);
				floatBuffer.push_back(derGlobal[i]);
			}
		}
	}
}

/// Write record to file.
void MilleBinary::writeRecord() {

	const int recordLength =
			(doublePrecision) ? -intBuffer.size() * 2 : intBuffer.size() * 2;
	binaryFile.write(reinterpret_cast<const char*>(&recordLength),
			sizeof(recordLength));
	if (doublePrecision)
		binaryFile.write(reinterpret_cast<char*>(&doubleBuffer[0]),
				doubleBuffer.size() * sizeof(doubleBuffer[0]));
	else
		binaryFile.write(reinterpret_cast<char*>(&floatBuffer[0]),
				floatBuffer.size() * sizeof(floatBuffer[0]));
	binaryFile.write(reinterpret_cast<char*>(&intBuffer[0]),
			intBuffer.size() * sizeof(intBuffer[0]));
// start with new record
	intBuffer.resize(1);
	if (doublePrecision)
		doubleBuffer.resize(1);
	else
		floatBuffer.resize(1);
}
}
