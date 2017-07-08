/*
 * CaloBox.h
 *
 *  Created on: 11-May-2009
 *      Author: jamie
 */

#ifndef CALOBOX_H_
#define CALOBOX_H_

#include <map>
#include <utility>
#include <iostream>
#include <string>
namespace pftools {

class CaloBox {
public:
	CaloBox(double centerEta, double centerPhi, double dEta, double dPhi,
			unsigned nEta, unsigned nPhi);
	virtual ~CaloBox();

	bool fill(double eta, double phi, double energy);

	void reset();

	double centerEta_;
	double centerPhi_;
	double dEta_;
	double dPhi_;
	unsigned nEta_;
	unsigned nPhi_;

	int etaPosBound_;
	int etaNegBound_;
	int phiPosBound_;
	int phiNegBound_;

	void test();

	const std::map<std::pair<int, int>, double>& energies() const {
		return energies_;
	}

	std::ostream& dump(std::ostream& stream, double norm = 1.0, std::string rowDelim = "\n") const;

private:
	std::map<std::pair<int, int>, double> energies_;
	CaloBox() = delete;

};

std::ostream& operator<<(std::ostream& stream, const CaloBox& cb);

}

#endif /* CALOBOX_H_ */
