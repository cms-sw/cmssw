/*
 * GblData.h
 *
 *  Created on: Aug 18, 2011
 *      Author: kleinwrt
 */

#ifndef GBLDATA_H_
#define GBLDATA_H_

#include<iostream>
#include<vector>
#include<math.h>
#include "VMatrix.h"
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"

#include "Math/SMatrix.h"
#include "Math/SVector.h"
typedef ROOT::Math::SMatrix<double, 2, 5> SMatrix25;
typedef ROOT::Math::SMatrix<double, 2, 7> SMatrix27;
typedef ROOT::Math::SMatrix<double, 5, 5> SMatrix55;

//! Namespace for the general broken lines package
namespace gbl {

/// Data (block) for independent scalar measurement
/**
 * Data (block) containing value, precision and derivatives for measurements and kinks.
 * Created from attributes of GblPoints, used to construct linear equation system for track fit.
 */
class GblData {
public:
	GblData(unsigned int aLabel, double aMeas, double aPrec);
	virtual ~GblData();
	void addDerivatives(unsigned int iRow,
			const std::vector<unsigned int> &labDer, const SMatrix55 &matDer,
			unsigned int iOff, const TMatrixD &derLocal,
			const std::vector<int> &labGlobal, const TMatrixD &derGlobal,
			unsigned int nLocal, const TMatrixD &derTrans);
	void addDerivatives(unsigned int iRow,
			const std::vector<unsigned int> &labDer, const SMatrix27 &matDer,
			unsigned int nLocal, const TMatrixD &derTrans);
	void addDerivatives(const std::vector<unsigned int> &index,
			const std::vector<double> &derivatives);

	void setPrediction(const VVector &aVector);
	double setDownWeighting(unsigned int aMethod);
	double getChi2() const;
	void printData() const;
	void getLocalData(double &aValue, double &aWeight,
			std::vector<unsigned int>* &indLocal,
			std::vector<double>* &derLocal);
	void getAllData(double &aValue, double &aErr,
			std::vector<unsigned int>* &indLocal,
			std::vector<double>* &derLocal, std::vector<int>* &labGlobal,
			std::vector<double>* &derGlobal);
	void getResidual(double &aResidual, double &aVariance, double &aDownWeight,
			std::vector<unsigned int>* &indLocal,
			std::vector<double>* &derLocal);

private:
	unsigned int theLabel; ///< Label (of measurements point)
	double theValue; ///< Value (residual)
	double thePrecision; ///< Precision (1/sigma**2)
	double theDownWeight; ///< Down-weighting factor (0-1)
	double thePrediction; ///< Prediction from fit
	std::vector<unsigned int> theParameters; ///< List of fit parameters (with non zero derivatives)
	std::vector<double> theDerivatives; ///< List of derivatives for fit
	std::vector<int> globalLabels; ///< Labels for global derivatives
	std::vector<double> globalDerivatives; ///< Global derivatives
};
}
#endif /* GBLDATA_H_ */
