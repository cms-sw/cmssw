#include "RecoParticleFlow/PFClusterTools/interface/SpaceManager.h"
#include <cassert>
#include <algorithm>
#include <cmath>
#include "TROOT.h"
#include <string>
#include "TH3F.h"
#include "TVirtualFitter.h"
#include "TProfile.h"
using namespace pftools;

SpaceManager::SpaceManager(std::string name) :
	name_(name), barrelLimit_(1.4), transitionLimit_(1.4), endcapLimit_(5.0) {
	regionsToSVs_[BARREL_POS] = barrelPosRegion_;
	regionsToSVs_[ENDCAP_POS] = endcapPosRegion_;
}

SpaceManager::~SpaceManager() {

}

void SpaceManager::clear() {
	for (std::map<SpaceVoxelPtr, CalibratorPtr>::iterator it =
			myAddressBook.begin(); it!= myAddressBook.end(); ++it) {
		SpaceVoxelPtr s = (*it).first;
		CalibratorPtr c = (*it).second;
	}
}

void SpaceManager::createCalibrators(const Calibrator& toClone,
		const double etaSeg, const double phiSeg, const double energySeg) {
	std::cout << __PRETTY_FUNCTION__
			<< ": this method has not yet been implemented!\n";
	PFToolsException me("Unimplemented method! Sorry!");
	throw me;

}

void SpaceManager::createCalibrators(const Calibrator& toClone) {
	clear();
	std::cout << __PRETTY_FUNCTION__
			<< ": creating default calibration schema.\n";

	SpaceVoxelPtr sv(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 0, 3.0));
	barrelPosRegion_.push_back(sv);
	SpaceVoxelPtr sv1(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 3, 9.0));
	barrelPosRegion_.push_back(sv1);
	SpaceVoxelPtr sv2(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 9.0, 16.0));
	barrelPosRegion_.push_back(sv2);
	SpaceVoxelPtr sv3(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 16.0, 25.0));
	barrelPosRegion_.push_back(sv3);
	SpaceVoxelPtr sv4(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 25.0, 100.0));
	barrelPosRegion_.push_back(sv4);
	SpaceVoxelPtr sv5(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 100.0, 200.0));
	barrelPosRegion_.push_back(sv5);
	SpaceVoxelPtr sv6(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 200.0, 400.0));
	barrelPosRegion_.push_back(sv6);

	//	SpaceVoxelPtr sv(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 0, 2.0));
	//	barrelPosRegion_.push_back(sv);
	//	SpaceVoxelPtr sv1(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 1, 1.5));
	//	barrelPosRegion_.push_back(sv1);
	//	SpaceVoxelPtr sv2(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 1.5, 2.0));
	//	barrelPosRegion_.push_back(sv2);
	//	SpaceVoxelPtr sv3(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 2.0, 2.5));
	//	barrelPosRegion_.push_back(sv3);
	//	SpaceVoxelPtr sv4(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 2.5, 4.0));
	//	barrelPosRegion_.push_back(sv4);
	//	SpaceVoxelPtr sv41(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 4.0, 5.0));
	//	barrelPosRegion_.push_back(sv41);
	//	SpaceVoxelPtr sv5(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 5.0, 6.5));
	//	barrelPosRegion_.push_back(sv5);
	//	SpaceVoxelPtr sv51(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 6.5, 8.0));
	//	barrelPosRegion_.push_back(sv51);
	//	SpaceVoxelPtr sv6(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 8.0, 10.0));
	//	barrelPosRegion_.push_back(sv6);
	//	SpaceVoxelPtr sv61(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 10.0, 12.0));
	//	barrelPosRegion_.push_back(sv61);
	//	SpaceVoxelPtr sv7(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 12.0, 16.0));
	//	barrelPosRegion_.push_back(sv7);
	//	SpaceVoxelPtr sv8(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 16.0, 25.0));
	//	barrelPosRegion_.push_back(sv8);
	//	SpaceVoxelPtr sv9(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 25.0, 40.0));
	//	barrelPosRegion_.push_back(sv9);
	//	SpaceVoxelPtr sv10(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 40.0, 60.0));
	//	barrelPosRegion_.push_back(sv10);
	//	SpaceVoxelPtr sv11(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 60.0, 100.0));
	//	barrelPosRegion_.push_back(sv11);
	//	SpaceVoxelPtr sv12(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 100.0, 200.0));
	//	barrelPosRegion_.push_back(sv12);
	//	SpaceVoxelPtr sv13(new SpaceVoxel(-1.0*barrelLimit_, barrelLimit_, -3.2, 3.2, 200.0, 400.0));
	//	barrelPosRegion_.push_back(sv13);

	SpaceVoxelPtr sve(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 0, 3.0));
	endcapPosRegion_.push_back(sve);
	SpaceVoxelPtr sve0(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 3.0, 9.0));
	endcapPosRegion_.push_back(sve0);
	SpaceVoxelPtr sve1(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 9.0, 16.0));
	endcapPosRegion_.push_back(sve1);
	SpaceVoxelPtr sve2(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 16.0, 25.0));
	endcapPosRegion_.push_back(sve2);
	SpaceVoxelPtr sve3(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 25.0, 100.0));
	endcapPosRegion_.push_back(sve3);
	SpaceVoxelPtr sve4(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 100.0, 200.0));
	endcapPosRegion_.push_back(sve4);
	SpaceVoxelPtr sve5(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 200.0, 400.0));
	endcapPosRegion_.push_back(sve5);

	//	SpaceVoxelPtr sve(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 0, 0.5));
	//	endcapPosRegion_.push_back(sve);
	//	SpaceVoxelPtr sve0(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 0.5, 1.0));
	//	endcapPosRegion_.push_back(sve0);
	//	SpaceVoxelPtr sve1(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 1, 1.5));
	//	endcapPosRegion_.push_back(sve1);
	//	SpaceVoxelPtr sve2(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 1.5, 2.0));
	//	endcapPosRegion_.push_back(sve2);
	//	SpaceVoxelPtr sve3(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 2.0, 2.5));
	//	endcapPosRegion_.push_back(sve3);
	//	SpaceVoxelPtr sve4(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 2.5, 4.0));
	//	endcapPosRegion_.push_back(sve4);
	//	SpaceVoxelPtr sve5(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 4.0, 5.0));
	//	endcapPosRegion_.push_back(sve5);
	//	SpaceVoxelPtr sve51(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 5.0, 6.5));
	//	endcapPosRegion_.push_back(sve51);
	//	SpaceVoxelPtr sve6(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 6.5, 8.0));
	//	endcapPosRegion_.push_back(sve6);
	//	SpaceVoxelPtr sve61(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 8.0, 10.0));
	//	endcapPosRegion_.push_back(sve61);
	//	SpaceVoxelPtr sve62(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 10.0, 12.0));
	//	endcapPosRegion_.push_back(sve62);
	//	SpaceVoxelPtr sve7(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 12.0, 16.0));
	//	endcapPosRegion_.push_back(sve7);
	//	SpaceVoxelPtr sve8(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 16.0, 25.0));
	//	endcapPosRegion_.push_back(sve8);
	//	SpaceVoxelPtr sve9(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 25.0, 40.0));
	//	endcapPosRegion_.push_back(sve9);
	//	SpaceVoxelPtr sve10(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 40.0, 60.0));
	//	endcapPosRegion_.push_back(sve10);
	//	SpaceVoxelPtr sve11(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 60.0, 100.0));
	//	endcapPosRegion_.push_back(sve11);
	//	SpaceVoxelPtr sve12(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 100.0, 200.0));
	//	endcapPosRegion_.push_back(sve12);
	//	SpaceVoxelPtr sve13(new SpaceVoxel(barrelLimit_, endcapLimit_, -3.2, 3.2, 200.0, 400.0));
	//	endcapPosRegion_.push_back(sve13);

	for (std::vector<SpaceVoxelPtr>::iterator it = barrelPosRegion_.begin(); it
			!= barrelPosRegion_.end(); ++it) {
		myKnownSpaceVoxels.push_back(*it);
	}

	for (std::vector<SpaceVoxelPtr>::iterator it = endcapPosRegion_.begin(); it
			!= endcapPosRegion_.end(); ++it) {
		myKnownSpaceVoxels.push_back(*it);
	}

	for (std::vector<SpaceVoxelPtr>::iterator it = myKnownSpaceVoxels.begin(); it
			!= myKnownSpaceVoxels.end(); ++it) {
		CalibratorPtr c(toClone.clone());
		myAddressBook[*it] = c;
	}
	std::cout << "Address book size: \t\t"<< myAddressBook.size() << "\n";
	std::cout << "Known space voxels size: \t"<< myKnownSpaceVoxels.size()
			<< "\n";
	assert(myAddressBook.size() == myKnownSpaceVoxels.size());

}

void SpaceManager::createCalibrators(const Calibrator& toClone,
		const unsigned nEta, const double etaMin, const double etaMax,
		const unsigned nPhi, const double phiMin, const double phiMax,
		const unsigned nEnergy, const double energyMin, const double energyMax)
		throw(PFToolsException&) {
	clear();

	if (nEta == 0|| nPhi ==0|| nEnergy == 0) {
		PFToolsException
				me("Can't create calibrators with zero values for nEta, nPhi or nEnergy!");
		throw me;
	}

	double etaSeg = (etaMax - etaMin) / nEta;
	double phiSeg = (phiMax - phiMin) / nPhi;
	double energySeg = (energyMax - energyMin) / nEnergy;

	double eta1, eta2, phi1, phi2, energy1, energy2;
	for (unsigned k(0); k < nEta; ++k) {
		for (unsigned l(0); l < nPhi; ++l) {
			for (unsigned m(0); m < nEnergy; ++m) {
				eta1 = etaMin + k * etaSeg;
				eta2 = eta1 + etaSeg;

				phi1 = phiMin + l * phiSeg;
				phi2 = phi1 + phiSeg;

				energy1 = energyMin + m * energySeg;
				energy2 = energy1 + energySeg;
				SpaceVoxelPtr sv(new SpaceVoxel(eta1, eta2, phi1, phi2, energy1, energy2));
				myKnownSpaceVoxels.push_back(sv);
				CalibratorPtr c(toClone.clone());
				myAddressBook[sv] = c;
			}
		}
	}
	unsigned nCalibrators = nEta * nPhi * nEnergy;
	std::cout << "Created "<< nCalibrators << " calibrators.\n";
	std::cout << "Address book size: \t\t"<< myAddressBook.size() << "\n";
	std::cout << "Known space voxels size: \t"<< myKnownSpaceVoxels.size()
			<< "\n";
	assert(myAddressBook.size() == myKnownSpaceVoxels.size());
	makeInverseAddressBook();

}
CalibratorPtr SpaceManager::createCalibrator(const Calibrator& toClone,
		SpaceVoxelPtr s) {
	CalibratorPtr c;
	int known = count(myKnownSpaceVoxels.begin(), myKnownSpaceVoxels.end(), s);
	if (known == 0) {
		myKnownSpaceVoxels.push_back(s);
		c.reset(toClone.clone());
		myAddressBook[s] = c;
	} else {
		c = myAddressBook[s];
	}
	assert(c != 0);
	return c;

}

CalibratorPtr SpaceManager::findCalibrator(const double eta, const double phi,
		const double energy) const {
	CalibratorPtr answer;
	for (std::vector<SpaceVoxelPtr>::const_iterator cit =
			myKnownSpaceVoxels.begin(); cit != myKnownSpaceVoxels.end(); ++cit) {
		SpaceVoxelPtr s = *cit;
		if (s->contains(eta, phi, energy)) {
			assert(count(myKnownSpaceVoxels.begin(), myKnownSpaceVoxels.end(), s) != 0);
			answer = (*myAddressBook.find(s)).second;
			break;
		} else {
			//assert(count(myKnownSpaceVoxels.begin(), myKnownSpaceVoxels.end(), s) == 0);
		}
	}
	return answer;
}

void SpaceManager::assignCalibration(const CalibratorPtr& c,
		const std::map<DetectorElementPtr, double>& result) {
	calibrationCoeffs_[c] = result;
	makeInverseAddressBook();
}

std::map<DetectorElementPtr, double> SpaceManager::getCalibration(CalibratorPtr c) {
	return calibrationCoeffs_[c];
}

TH1* SpaceManager::extractEvolution(DetectorElementPtr det, Region r, TF1& f1,
		bool useTruth) {

	std::vector<SpaceVoxelPtr> region;
	if (r == BARREL_POS)
		region = barrelPosRegion_;
	if (r == ENDCAP_POS)
		region = endcapPosRegion_;
	//region = regionsToSVs_[r];

	std::sort(region.begin(), region.end(), SpaceVoxel());

	std::string detElName = DetElNames[det->getType()];
	std::string name("hDist_");
	name.append(RegionNames[r]);
	name.append("_");
	name.append(DetElNames[det->getType()]);

	double minE(1000);
	double maxE(0);

	TH2F hDist(name.c_str(), name.c_str(), 100, 0, 300, 50, 0.0, 2.5);
	//	TH3F hSurf(nameSurf.c_str(), nameSurf.c_str(), 30, 0, 50, 10, 0.0, 3.0, 30,
	//			0.0, 2.5);
	for (std::vector<SpaceVoxelPtr>::iterator i = region.begin(); i
			!= region.end(); ++i) {
		SpaceVoxelPtr s = *i;
		//double midE = (s->maxEnergy() + s->minEnergy()) / 2.0;
		if (s->maxEnergy() > maxE)
			maxE = s->maxEnergy();
		if (s->minEnergy() < minE)
			minE = s->minEnergy();
		CalibratorPtr c = myAddressBook[s];
		double coeff = calibrationCoeffs_[c][det];
		if (coeff != 0.0) {
			std::vector<ParticleDepositPtr> particles = c->getParticles();
			for (std::vector<ParticleDepositPtr>::iterator it =
					particles.begin(); it != particles.end(); ++it) {
				if (useTruth)
					hDist.Fill((*it)->getTruthEnergy(), coeff);
				else
					hDist.Fill((*it)->getRecEnergy(), coeff);
			}
		}
	}

	hDist.FitSlicesY();
	hDist.ProfileX();
	hDist.Write();
	std::string nameProfile(name);
	nameProfile.append("_pfx");
	name.append("_1");

	TH1D* slices = (TH1D*) gDirectory->Get(name.c_str());
	//TH2D* slicesSurf = (TH2D*) gDirectory->Get(nameSurf.c_str());
	TProfile* profile = (TProfile*) gDirectory->Get(nameProfile.c_str());
	profile->Fit(&f1);
	slices->Fit(&f1);
	profile->Write();

	return slices;

}

double SpaceManager::evolveCoefficient(DetectorElementPtr det, double energy,
		double eta, double phi) {
	if (eta < barrelLimit_) {
		TF1& func = barrelPosEvolutions_[det];
		return func.Eval(energy);
	}
	TF1& func = endcapPosEvolutions_[det];
	return func.Eval(energy);
}

double SpaceManager::interpolateCoefficient(DetectorElementPtr det,
		double energy, double eta, double phi) {
	CalibratorPtr c = findCalibrator(eta, phi, energy);

	SpaceVoxelPtr s = inverseAddressBook_[c];

	double midEnergy = (s->minEnergy() + s->maxEnergy())/2.0;
	//interpolate left or right?
	double diffEnergy = energy - midEnergy;
	double thisCoeff = calibrationCoeffs_[c][det];

	double interpolatedCoeff = thisCoeff;
	double adjacentCoeff = 0.0;
	double adjacentEnergy = 0.0;
	if (diffEnergy > 0) {
		//look to higher energy calibrators
		CalibratorPtr adjC = findCalibrator(eta, phi, s->maxEnergy() + 0.1);
		if (adjC != 0) {
			SpaceVoxelPtr adjS = inverseAddressBook_[adjC];
			adjacentCoeff = calibrationCoeffs_[adjC][det];
			adjacentEnergy = (adjS->minEnergy() + adjS->maxEnergy()) / 2.0;
		}
	} else {
		//look to lower energy calibrations
		CalibratorPtr adjC = findCalibrator(eta, phi, s->minEnergy() - 0.1);
		if (adjC != 0) {
			SpaceVoxelPtr adjS = inverseAddressBook_[adjC];
			adjacentCoeff = calibrationCoeffs_[adjC][det];
			adjacentEnergy = (adjS->minEnergy() + adjS->maxEnergy()) / 2.0;
		}
	}
	if (adjacentCoeff != 0) {
		interpolatedCoeff = thisCoeff + diffEnergy* (adjacentCoeff - thisCoeff)
				/ (adjacentEnergy - midEnergy);
	}
	return interpolatedCoeff;
}

std::ostream& SpaceManager::printCalibrations(std::ostream& stream) {
	stream << "Calibration results: \n";
	//	std::sort(myKnownSpaceVoxels.begin(), myKnownSpaceVoxels.end(),
	//			SpaceVoxel());
	stream << "WARNING! Haven't sorted space voxels properly!\n";
	for (std::vector<SpaceVoxelPtr>::iterator it = myKnownSpaceVoxels.begin(); it
			!= myKnownSpaceVoxels.end(); ++it) {
		SpaceVoxelPtr s = *it;
		CalibratorPtr c = myAddressBook[s];
		stream << *s << "\n";
		stream << "\t[";
		std::map<DetectorElementPtr, double> result = calibrationCoeffs_[c];
		for (std::map<DetectorElementPtr, double>::iterator b = result.begin(); b
				!= result.end(); ++b) {
			DetectorElementPtr d = (*b).first;
			stream << *d << ": ";
			double ans = (*b).second;
			stream << ans << ", ";
		}
		stream << "]\n";
	}

	return stream;
}

void SpaceManager::makeInverseAddressBook() {
	inverseAddressBook_.clear();
	for (std::map<SpaceVoxelPtr, CalibratorPtr>::iterator it =
			myAddressBook.begin(); it != myAddressBook.end(); ++it) {
		SpaceVoxelPtr s = (*it).first;
		CalibratorPtr c = (*it).second;
		inverseAddressBook_[c] = s;
	}
}
