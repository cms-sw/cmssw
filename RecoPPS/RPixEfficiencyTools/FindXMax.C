#include <iostream>
#include <cmath>
#include "Riostream.h"
#include "TF1.h"
#include "TMath.h"

using namespace std;
double findXMax(TF1* fitFunc,double xMin, double xLimit, double dist_fraction, double epsilon){
		double xMax = xLimit;
		double xMaxLow = xMin;
		double error = fitFunc->Integral(xMin,xMax)/fitFunc->Integral(xMin,xLimit) - dist_fraction;
		int cycle = 1;
		while(TMath::Abs(error)>epsilon){
			if(error > 0){
				xMax -= (xMax - xMaxLow)/2.;
			}
			else{
				double xMaxLow_tmp = xMaxLow;
				xMaxLow = xMax;
				xMax += (xMax - xMaxLow_tmp)/2.;

			}
			error = fitFunc->Integral(xMin,xMax)/fitFunc->Integral(xMin,xLimit) - dist_fraction;
			if(cycle > 1000){
				std::cout << "WARNING: No xMax found!" <<std::endl;
				return xLimit;
			}
			cycle++;
		}
		std::cout << "Converged after " << cycle << "cycles" << std::endl;
		return xMax;
}
