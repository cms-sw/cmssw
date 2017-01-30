#include "DataFormats/ParticleFlowReco/interface/CaloWindow.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <TMath.h>

using namespace pftools;
//using namespace edm;
using namespace std;

//Let's do CaloRing first
CaloRing::CaloRing() :
	panes_(1) {
	myPanes_.push_back(0);
}

CaloRing::CaloRing(unsigned nPanes) :
	panes_(nPanes) {
	for (unsigned j(0); j < nPanes; ++j)
		myPanes_.push_back(0);
}

double CaloRing::getEnergy(unsigned pane) const {
	if (pane < panes_)
		return myPanes_[pane];
	else
		return 0;
}

bool CaloRing::setEnergy(unsigned pane, double energy) {
	if (pane < panes_) {
		myPanes_[pane] = energy;
		return true;
	}
	return false;
}

bool CaloRing::addEnergy(unsigned pane, double energy) {
	if (pane < panes_) {
		myPanes_[pane] += energy;
		return true;
	}
	return false;
}

double CaloRing::totalE() const {
	double ans(0.0);
	for (std::vector<double>::const_iterator cit = myPanes_.begin(); cit
			!= myPanes_.end(); ++cit) {
		ans += *cit;
	}
	return ans;
}

void CaloRing::reset() {
	for (std::vector<double>::iterator cit = myPanes_.begin(); cit
			!= myPanes_.end(); ++cit)
		*cit = 0.0;

}

void CaloRing::printEnergies(std::ostream& s, double range) {
	for (std::vector<double>::iterator cit = myPanes_.begin(); cit
			!= myPanes_.end(); ++cit)
		s << (*cit)/range << "\t";
}

std::ostream& pftools::operator<<(std::ostream& s,
		const pftools::CaloRing& caloRing) {
	for (auto const& e: caloRing.getEnergies()) {
		s << e << "\t";
	}
	s << " => ring E = " << caloRing.totalE();
	return s;
}

std::ostream& pftools::operator<<(std::ostream& s,
		const pftools::CaloWindow& caloWindow) {

	s << "CaloWindow at (" << caloWindow.baryEta() << ", "
	  << caloWindow.baryPhi() << "):\n";
	double totalE(0.0);
	for (std::map<unsigned, pftools::CaloRing>::const_iterator cit =
	      caloWindow.getRingDepositions().begin(); cit != caloWindow.getRingDepositions().end(); ++cit) {
		unsigned ring = (*cit).first;
		const CaloRing& cr = (*cit).second;
		s << "Ring " << ring << ":\t" << cr << "\n";
		totalE += cr.totalE();
	}

	s << "\tTotal E = " << totalE << std::endl;
	return s;

}

vector<double> CaloWindow::stream(double normalisation) const {
	vector<double> stream;
	for (map<unsigned, pftools::CaloRing>::const_iterator cit =
			energies_.begin(); cit != energies_.end(); ++cit) {
		CaloRing c = (*cit).second;
		std::vector<double> ringE = c.getEnergies();
		stream.insert(stream.end(), ringE.begin(), ringE.end());
	}
	if(normalisation != 1.0){
		for(vector<double>::iterator i = stream.begin(); i != stream.end(); ++i) {
			double& entry = *i;
			entry /= normalisation;
		}
	}
	return stream;

}

void CaloWindow::printEnergies(std::ostream& s, double range) {

	for (std::map<unsigned, pftools::CaloRing>::const_iterator cit =
			energies_.begin(); cit != energies_.end(); ++cit) {
		CaloRing c = (*cit).second;
		c.printEnergies(s, range);
		s << "\t";
	}

}

CaloWindow::CaloWindow() :
	baryEta_(0.0), baryPhi_(0.0), nRings_(1), deltaR_(0.1), nPanes_(1),  axis_(0.0) {

}

CaloWindow::CaloWindow(double eta, double phi, unsigned nRings, double deltaR, unsigned nPanes, double axis) {

	init(eta, phi, nRings, deltaR, nPanes, axis);
}

void CaloWindow::reset() {
	energies_.clear();
}

void CaloWindow::init(double eta, double phi, unsigned nRings, double deltaR, unsigned nPanes, double axis) {
	baryEta_ = eta;
	baryPhi_ = phi;
	nRings_ = nRings;
	deltaR_ = deltaR;
	nPanes_ = nPanes;
	axis_ = axis;

	energies_.clear();
	CaloRing c(1);
	energies_[0] = c;
	for (unsigned j(1); j < nRings_; ++j) {

		CaloRing r(nPanes);
		energies_[j] = r;
	}
}

CaloWindow::~CaloWindow() {

}

std::pair<unsigned, unsigned> CaloWindow::relativePosition(double eta,
		double phi) const {
	//How far is this hit from the barycentre in deltaR?
	double dEta = eta - baryEta_;
	double dPhi = deltaPhi(phi, baryPhi_);

	double dR = deltaR(eta, phi, baryEta_, baryPhi_);

	unsigned ring = static_cast<unsigned> (floor(dR / deltaR_));

	if (ring == 0) {
		//LogDebug("CaloWindow") << "Relative position: adding to central ring\n";
		return std::pair<unsigned, unsigned>(0, 0);
	}

	if (ring >= nRings_) {
		return std::pair<unsigned, unsigned>(ring, 0);
	}

	double dTheta = 0;

	if (dEta > 0) {
		if (dPhi > 0)
			dTheta = TMath::ATan(dPhi / dEta);
		else
			dTheta = 2 * TMath::Pi() + TMath::ATan(dPhi / dEta);
	} else {
		if (dPhi > 0)
			dTheta = TMath::Pi() + TMath::ATan(dPhi / dEta);
		else
			dTheta = TMath::Pi() + TMath::ATan(dPhi / dEta);
	}
	//Rotation. Rotate theta into axis of caloWindow
	// and also by half a window pane.
	//TODO: bug check!!
	//double dThetaOrig(dTheta);
	double paneOn2 = TMath::Pi() / (*energies_.find(ring)).second.getNPanes();
	dTheta =  dTheta - axis_ - paneOn2;
	//Original theta between 0 and 2 Pi, but transform above might move us beyond this, so...
	//double dThetaCopy(dTheta);
	//std::cout << "dTheta " << dThetaOrig << ", axis " << axis_ << ", paneOn2 " << paneOn2 << ", thetaPrime " << dTheta << "\n";
	if(dTheta > 2 *TMath::Pi()) {
		//std::cout << "To infinity and beyond...\n";
		while(dTheta > 2 * TMath::Pi()) {
			dTheta -= 2 * TMath::Pi();
			//std::cout << "dTheta1 " << dTheta << "\n";
		}
	}
	else if (dTheta < 0) {
		//std::cout << "To infinity and beyond 2... dTheta = " << dTheta << "\n";
		while(dTheta < 0) {
			dTheta += 2 * TMath::Pi();
			//std::cout << "dTheta2 " << dTheta << "\n";
		}
	}

//	std::cout << "\tdTheta is " << dTheta << " rad \n";
	unsigned division = static_cast<unsigned> (floor((*energies_.find(ring)).second.getNPanes() * dTheta
			/ (TMath::Pi() * 2)));
	//LogDebug("CaloWindow") << "Ring is " << ring << ", pane is "
	//		<< division << "\n";

	return std::pair<unsigned, unsigned>(ring, division);

}

bool CaloWindow::addHit(double eta, double phi, double energy) {

	std::pair<unsigned, unsigned> position = relativePosition(eta, phi);
	if (position.first >= nRings_) {
/*		double dEta = eta - baryEta_;
		double dPhi = deltaPhi(phi, baryPhi_);
		double dR = deltaR(eta, phi, baryEta_, baryPhi_);
		LogDebug("CaloWindow")
				<< "Hit is outside my range - it would be in ring "
				<< position.first << ", with dR = " << dR << std::endl;*/
		return false;
	}
//	std::cout << "Adding hit to ring " << position.first << " in position "
//			<< position.second << " with energy " << energy << "\n";

	CaloRing& c = energies_[position.first];
	c.addEnergy(position.second, energy);

	return true;
}
std::map<unsigned, double> CaloWindow::getRingEnergySummations() const {
	std::map<unsigned, double> answer;
	for (std::map<unsigned, CaloRing>::const_iterator cit = energies_.begin(); cit
			!= energies_.end(); ++cit) {
		std::pair<unsigned, CaloRing> pair = *cit;
		answer[pair.first] = pair.second.totalE();
	}

	return answer;
}

void TestCaloWindow::doTest() {

	CaloWindow cw(0.0, 0.0, 3, 0.1, 4);
	std::cout << cw << std::endl;
	cw.addHit(0, 0.05, 1.0);
	cw.addHit(0.22, 0.05, 2.0);
	cw.addHit(0, 0.8, 4.0);
	cw.addHit(0.2, 0, 1.0);
	cw.addHit(0.15, 0.15, 2.0);
	cw.addHit(0.0, 0.2, 3.0);
	cw.addHit(-0.15, 0.15, 4.0);
	cw.addHit(-0.2, 0, 5.0);
	cw.addHit(-0.15, -0.15, 6.0);
	cw.addHit(-0.0, -0.2, 7.0);
	cw.addHit(0.15, -0.15, 8.0);

	std::cout << cw << std::endl;
	std::cout << "bye bye" << std::endl;

}
