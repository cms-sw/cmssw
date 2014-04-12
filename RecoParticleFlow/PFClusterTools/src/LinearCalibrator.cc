#include "RecoParticleFlow/PFClusterTools/interface/LinearCalibrator.h"
#include <cassert>
#include <cmath>
#include "TVector.h"
#include "TDecompLU.h"
#include "TDecompSVD.h"

#include "TDecompBK.h"

#include <iostream>

using namespace pftools;

/* These utility functions allow you to print matrices and vectors to a stream,
 * and copy and paste the output directly into Octave or Matlab
 */
void printMat(std::ostream& s, const TMatrixD& input) {
	s << "\t[";
	for (int i(0); i < input.GetNrows(); i++) {
		for (int j(0); j < input.GetNcols(); j++) {
			s << input[i][j]<< ", \t";
		}
		if (i != (input.GetNrows() - 1)) {
			s << ";\n";
			s << "\t";
		}
	}
	s << "\t]\n";
}

void printVec(std::ostream& s, const TVectorD& input) {
	s << "\t[";
	for (int i(0); i < input.GetNrows(); i++) {
		s << input[i];
		if (i != (input.GetNrows() - 1)) {
			s << ",\n\t";
		}
	}
	s << "\t]\n";
}

LinearCalibrator::LinearCalibrator() {

}

LinearCalibrator::~LinearCalibrator() {
}
LinearCalibrator* LinearCalibrator::clone() const {
	return new LinearCalibrator(*this);
}

LinearCalibrator::LinearCalibrator(const LinearCalibrator& lc) {
	myDetectorElements = lc.myDetectorElements;
	myParticleDeposits = lc.myParticleDeposits;
}

LinearCalibrator* LinearCalibrator::create() const {
	return new LinearCalibrator();
}

std::map<DetectorElementPtr, double> LinearCalibrator::getCalibrationCoefficientsCore()
		throw(PFToolsException&) {
//	std::cout << __PRETTY_FUNCTION__
//			<< ": determining linear calibration coefficients...\n";
	if (!hasParticles()) {
		//I have no particles to calibrate to - throw exception.
		PFToolsException me("Calibrator has no particles for calibration!");
		throw me;
	}
	//std::cout << "\tGetting eij matrix...\n";
	TMatrixD eij;
	TVectorD truthE;
	initEijMatrix(eij, truthE);

//	std::cout << "\tEij matrix:\n";
//	printMat(std::cout, eij);

	//std::cout << "\tGetting projections...\n";
	TVectorD proj;
	TMatrixD hess;

	getProjections(eij, proj, truthE);
	//std::cout << "\tProjections:\n";
	//printVec(std::cout, proj);
	getHessian(eij, hess, truthE);

	TDecompLU lu;
	lu.SetMatrix(hess);
	//std::cout << "\tHessian:\n";
	//printMat(std::cout, hess);

	lu.SetTol(1e-25);
	TMatrixD hessInv = lu.Invert();
	//std::cout <<"\tInverse Hessian:\n";
	//printMat(std::cout, hessInv);
	TVectorD calibsSolved(eij.GetNcols());

	bool ok(true);
	calibsSolved = lu.Solve(proj, ok);
	if (ok) {
		//std::cout << "\tLU reports ok.\n";
	}
	else {
		std::cout << "\tWARNING: LU reports NOT ok.\n";
		//This usually happens when you've asked the calibrator to solve for the 'a' term, without including
		//a dummy 'a' term in each particle deposit.
		//Make sure you do
		//Deposition dOffset(offset, eta, phi, 1.0);
		//particle->addRecDeposition(dOffset);
		//particle->addTruthDeposition(dOffset);
		std::cout
				<< "\tDid you forget to add a dummy offset deposition to each particle?\n";
		std::cout << "\tThrowing an exception!"<< std::endl;
		PFToolsException
				me("TDecompLU did not converge successfully when finding calibrations. Did you forget to add a dummy offset deposition to each particle?");
		throw me;
	}

	//std::cout << "\tCalibrations: \n";
	//printVec(std::cout, calibsSolved);

	std::map<DetectorElementPtr, double> answers;
	for (std::map<DetectorElementPtr, unsigned>::iterator it =
			myDetElIndex.begin(); it != myDetElIndex.end(); ++it) {
		DetectorElementPtr de = (*it).first;
		answers[de] = calibsSolved[(*it).second];
	}
	return answers;
}

void LinearCalibrator::initEijMatrix(TMatrixD& eij, TVectorD& truthE) {
	//std::cout << __PRETTY_FUNCTION__ << "\n";
	//std::cout << "\tGetting detector element indices...\n";
	populateDetElIndex();
	eij.Clear();
	eij.Zero();

	truthE.ResizeTo(myParticleDeposits.size());
	truthE.Zero();

	//First determine number of calibration constants.

	eij.ResizeTo(myParticleDeposits.size(), myDetElIndex.size());

	//loop over all particle deposits
	unsigned index(0);
	for (std::vector<ParticleDepositPtr>::const_iterator cit =
			myParticleDeposits.begin(); cit != myParticleDeposits.end(); ++cit) {
		ParticleDepositPtr p = *cit;
		//get each of the relevant detector elements

		for (std::vector<DetectorElementPtr>::const_iterator detElIt =
				myDetectorElements.begin(); detElIt != myDetectorElements.end(); ++detElIt) {
			DetectorElementPtr de = *detElIt;
			eij[index][myDetElIndex[de]] = p->getRecEnergy(de);
			//truthE[p->getId()] += p->getTruthEnergy(de);
		}
		truthE[index] = p->getTruthEnergy();
		++index;
	}

}

TVectorD& LinearCalibrator::getProjections(const TMatrixD& eij, TVectorD& proj,
		const TVectorD& truthE) const {
	//std::cout << __PRETTY_FUNCTION__ << "\n";
	proj.ResizeTo(eij.GetNcols());
	proj.Zero();

	for (int j(0); j < eij.GetNcols(); ++j) {
		for (int i(0); i < eij.GetNrows(); ++i) {
			proj[j] += eij[i][j] / truthE[i];
		}
	}

	return proj;
}

TMatrixD& LinearCalibrator::getHessian(const TMatrixD& eij, TMatrixD& hess,
		const TVectorD& truthE) const {
	//std::cout << __PRETTY_FUNCTION__ << "\n";
	unsigned nCalibs(eij.GetNcols());
	hess.ResizeTo(nCalibs, nCalibs);
	hess.Zero();

	for (unsigned i(0); i < nCalibs; ++i) {
		for (unsigned j(0); j < nCalibs; ++j) {
			for (int n(0); n < eij.GetNrows(); ++n) {
				hess[i][j] += eij[n][i] * eij[n][j]/ pow(truthE[n], 2.0);
			}
		}
	}

	return hess;
}

void LinearCalibrator::populateDetElIndex() {
	//reserve index = 0 for the constant term, if we're told to compute it
	unsigned index(0);

	myDetElIndex.clear();
	//loop over known detector elements, and assign a unique row/column index to each
	for (std::vector<DetectorElementPtr>::const_iterator cit =
			myDetectorElements.begin(); cit != myDetectorElements.end(); ++cit) {
		DetectorElementPtr de = *cit;
		//std::cout << "\tGot element: "<< *de;
		//check we don't have duplicate detector elements
		if (myDetElIndex.count(de) == 0) {
			myDetElIndex[de] = index;
			++index;
		}
	}

}

