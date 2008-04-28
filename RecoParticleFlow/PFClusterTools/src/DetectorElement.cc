#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.hh"
#include "RecoParticleFlow/PFClusterTools/interface/MinimiserException.hh"
#include <iostream>
using namespace pftools;
//A comment
DetectorElement::DetectorElement(DetectorElementType type, double calib) :
	myType(type), myCalib(calib) {

}


void DetectorElement::setCalib(double calib) throw(MinimiserException&){
	//I'll tolerate very small negative numbers (artefacts of the minimisation algo
	//but otherwise this shouldn't be allowed.
//	if(calib > -0.01) {
		myCalib = calib;
//	}
//	else {
//		MinimiserException me("Setting calibration <= 0!");
//		throw me;
//	}
}

double DetectorElement::getCalib() const {
	return myCalib;
}

double DetectorElement::getCalib(double eta, double phi) const {
	return getCalib();
}

DetectorElement::~DetectorElement() {
}



std::ostream& pftools::operator<<(std::ostream& s, const DetectorElement& de) {
	s << "DetectorElement: " << pftools::DetElNames[de.getType()] << ", \tcalib: " << de.getCalib();

	return s;
}

//std::ostream& minimiser::operator<<(std::ostream& s, const minimiser::DetectorElement& de) {
//	s << "DetectorElement: " << minimiser::DetElNames[de.getType()]  << "\n";
//	return s;
//}


//void DetectorElement::getEnergyMap(TH2F& plot) const {
//	plot.SetTitle("Energy map;eta;phi");
//	plot.SetName("hEnergyMap");
//	plot.SetBins(100, -10, 10, 100, 0, 2*3.145);
//	for (std::vector<Deposition>::const_iterator cit = myDepositions.begin(); cit
//			!= myDepositions.end(); ++cit) {
//		Deposition d = *cit;
//		plot.Fill(d.getEta(), d.getPhi(), myCalib * d.getEnergy());
//	}
//}
//
//void DetectorElement::getHitMap(TH2F& plot) const {
//	plot.SetTitle("Hit map;eta;phi");
//	plot.SetName("hHitMap");
//	plot.SetBins(100, -10, 10, 100, 0, 2*3.145);
//	for (std::vector<Deposition>::const_iterator cit = myDepositions.begin(); cit
//			!= myDepositions.end(); ++cit) {
//		Deposition d = *cit;
//		plot.Fill(d.getEta(), d.getPhi());
//	}
//}
//
//void DetectorElement::getEnergySpectrum(TH1F& plot) const {
//	plot.SetTitle("Energy spectrum;Energy (GeV)");
//	plot.SetName("hEnergySpectrum");
//	plot.SetBins(100, 0, 200);
//	for (std::vector<Deposition>::const_iterator cit = myDepositions.begin(); cit
//			!= myDepositions.end(); ++cit) {
//		Deposition d = *cit;
//		plot.Fill(myCalib * d.getEnergy());
//	}
//}
