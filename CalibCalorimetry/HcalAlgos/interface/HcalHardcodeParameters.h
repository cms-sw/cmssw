#ifndef HcalAlgos_HcalHardcodeParameters_h
#define HcalAlgos_HcalHardcodeParameters_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

class HcalHardcodeParameters {
	public:
		//default constructor
		HcalHardcodeParameters() {}
		//construct from values
		HcalHardcodeParameters(double pedestal, double pedestalWidth, std::vector<double> gain, std::vector<double> gainWidth, 
							   int qieType, std::vector<double> qieOffset, std::vector<double> qieSlope, int mcShape, int recoShape,
							   double photoelectronsToAnalog, std::vector<double> darkCurrent);
		//construct from pset
		HcalHardcodeParameters(const edm::ParameterSet & p);
		
		//destructor
		virtual ~HcalHardcodeParameters() {}
		
		//accessors
		//note: all vector accessors use at() in order to throw exceptions for malformed conditions
		const double pedestal() const { return pedestal_; }
		const double pedestalWidth() const { return pedestalWidth_; }
		const double gain(unsigned index) const { return gain_.at(index); }
		const double gainWidth(unsigned index) const { return gainWidth_.at(index); }
		const int qieType() const { return qieType_; }
		const double qieOffset(unsigned range) const { return qieOffset_.at(range); }
		const double qieSlope(unsigned range) const { return qieSlope_.at(range); }
		const int mcShape() const { return mcShape_; }
		const int recoShape() const { return recoShape_; }
		const double photoelectronsToAnalog() const { return photoelectronsToAnalog_; }
		const double darkCurrent(unsigned index) const { return darkCurrent_.at(index); }
		
	private:
		//member variables
		double pedestal_, pedestalWidth_;
		std::vector<double> gain_, gainWidth_;
		int qieType_;
		std::vector<double> qieOffset_, qieSlope_;
		int mcShape_, recoShape_;
		double photoelectronsToAnalog_;
		std::vector<double> darkCurrent_;
};

#endif
