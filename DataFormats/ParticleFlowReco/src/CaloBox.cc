/*
 * CaloBox.cc
 *
 *  Created on: 11-May-2009
 *      Author: jamie
 */

#include "DataFormats/ParticleFlowReco/interface/CaloBox.h"

#include <iostream>
#include <cmath>
using namespace pftools;
using namespace std;

CaloBox::CaloBox(double centerEta, double centerPhi, double dEta, double dPhi,
		unsigned nEta, unsigned nPhi) :
	centerEta_(centerEta), centerPhi_(centerPhi), dEta_(dEta), dPhi_(dPhi),
			nEta_(nEta), nPhi_(nPhi) {
	if (nEta_ % 2 != 1) {
		cout << __PRETTY_FUNCTION__
				<< ": should use odd numbers for nEta and nPhi as CaloBox won't be symmetric otherwise!\n";
		etaPosBound_ = (nEta / 2);
		etaNegBound_ = -1 * static_cast<int> ((nEta / 2) + 1);

	} else {
		etaPosBound_ = (nEta - 1) / 2;
		etaNegBound_ = -1 * static_cast<int> ((nEta - 1) / 2);
	}

	if (nPhi_ % 2 != 1) {
		cout << __PRETTY_FUNCTION__
				<< ": should use odd numbers for nEta and nPhi as CaloBox won't be symmetric otherwise!\n";

		phiPosBound_ = (nPhi / 2);
		phiNegBound_ = -1 * static_cast<int> ((nPhi / 2) + 1);
	} else {
		phiPosBound_ = (nPhi - 1) / 2;
		phiNegBound_ = -1 * static_cast<int> ((nPhi - 1) / 2);
	}

	reset();

	//cout << "CaloBox with n+ " << etaPosBound_ << ", n- " << etaNegBound_
	//		<< ", p+ " << phiPosBound_ << ", p-" << phiNegBound_ << "\n";

}

void CaloBox::reset() {
	for (int r = phiPosBound_; r >= phiNegBound_; --r) {
		for (int c = etaNegBound_; c <= etaPosBound_; ++c) {
			std::pair<int, int> pos(c, r);
			energies_[pos] = 0.0;
		}
	}
}

CaloBox::~CaloBox() {

}

bool CaloBox::fill(double eta, double phi, double energy) {
	int etaDiv = static_cast<int>(floor((eta - centerEta_ + dEta_ / 2.0) / dEta_));
	int phiDiv = static_cast<int>(floor((phi - centerPhi_ + dPhi_ / 2.0) / dPhi_));
	//cout << "Testing for " << etaDiv << ", " << phiDiv << "\n";

	if (etaDiv >= 0 && etaDiv > etaPosBound_)
		return false;
	if (etaDiv < 0 && etaDiv < etaNegBound_)
		return false;
	if (phiDiv >= 0 && phiDiv > phiPosBound_)
		return false;
	if (phiDiv < 0 && phiDiv < phiNegBound_)
		return false;

	pair<int, int> key(etaDiv, phiDiv);

	energies_[key] += energy;
	return true;

}

std::ostream& CaloBox::dump(std::ostream& stream, double norm, string rowDelim) const {

	for (int r = phiPosBound_; r >= phiNegBound_; --r) {
		for (int c = etaNegBound_; c <= etaPosBound_; ++c) {
			pair<int, int> pos(c, r);
			double energy = energies_.find(pos)->second/norm;
			stream << energy << "\t";
		}
		stream << rowDelim;
	}
//
//	for (map<pair<int, int> , double>::const iterator i = energies_.begin(); i
//			!= energies_.end(); ++i) {
//		pair<int, int> loc = i->first;
//		double en = i->second;
//		stream << loc.first << ", " << loc.second << ": " << en << "\n";
//	}

	return stream;
}

void CaloBox::test() {

	CaloBox cb(0.0, 0.0, 1.0, 1.0, 5, 5);

	unsigned count(0);
	bool ok(false);

	ok = cb.fill(2, 1, 10);
	if (!ok)
		cout << "Box fill failed! Count = " << count << "\n";
	++count;

	ok = cb.fill(-1, 2, 20);
	if (!ok)
		cout << "Box fill failed! Count = " << count << "\n";
	++count;

	ok = cb.fill(-2, 5, 10);
	if (!ok)
		cout << "Box fill failed! Count = " << count << "\n";
	++count;

	ok = cb.fill(0.1, 1.3, 10);
	if (!ok)
		cout << "Box fill failed! Count = " << count << "\n";
	++count;

	ok = cb.fill(-1.4, 1.6, 10);
	if (!ok)
		cout << "Box fill failed! Count = " << count << "\n";
	++count;

	cout << cb;

}

std::ostream& pftools::operator<<(std::ostream& stream, const CaloBox& cb) {
	stream << "CaloBox at " << cb.centerEta_ << ", " << cb.centerPhi_ << ":\n";
	cb.dump(stream, 1.0, "\n");
	return stream;
}

