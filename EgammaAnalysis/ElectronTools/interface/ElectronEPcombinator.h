#ifndef ElectronEPcombinator_H
#define ElectronEPcombinator_H

#include "EgammaAnalysis/ElectronTools/interface/SimpleElectron.h"
#include <stdio.h>
#include <math.h>

class ElectronEPcombinator
{
	public:
		ElectronEPcombinator(){} 
		void combine(SimpleElectron & electron); 
		void setCombinationMode(int mode){mode_ = mode;}

	private:
		SimpleElectron electron_;
		void computeEPcombination();
		double combinedMomentum_;
		double combinedMomentumError_;
		double scEnergy_; 
		double scEnergyError_; 
		double trackerMomentum_; 
		double trackerMomentumError_;
		int elClass_;
		int mode_;
};

#endif
