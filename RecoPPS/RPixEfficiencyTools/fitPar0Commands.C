// PAR 0 FIT

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

void UsefulCommandsSt2{

	.L FindXMax.C++
	// Canvas_1->Draw();

	map< std::pair<int,int>, vector<double> > fitRanges;
	fitRanges[std::pair<int,int>(0,0)] = {2.1,11,1.5,6,2.2,5};
	fitRanges[std::pair<int,int>(0,2)] = {2.1,11,1.5,6,2.2,5};
	fitRanges[std::pair<int,int>(1,0)] = {2.1,11,1.5,6,2.2,5};
	fitRanges[std::pair<int,int>(1,2)] = {2.1,11,1.5,6,2.2,5};
	TF1* amplitudeFitFunction2 = new TF1("amplitudeFitFunction2","[0]/(x-[1])+[2]",0,100);
	int station = 2;
	double max_fraction = 0.3;

	double xMin = h2TrackHitDistribution_arm0_st2_rp3_0->GetXaxis()->GetXmin();

	int arm = 0;
	// amplitudeFitFunction2->SetParameters(0,0,0);
	h2TrackHitDistribution_arm0_st2_rp3_0->Fit("amplitudeFitFunction2","","",fitRanges[std::pair<int,int>(arm,station)].at(0),fitRanges[std::pair<int,int>(arm,station)].at(1));
	amplitudeFitFunction2->SetLineColor(kBlue);
	amplitudeFitFunction2->DrawCopy("same");
	double x_max_0_2 = h2TrackHitDistribution_arm0_st2_rp3_0->GetBinCenter((h2TrackHitDistribution_arm0_st2_rp3_0->GetMaximumBin()))
	double max_0_2 = amplitudeFitFunction2->Eval(x_max_0_2)
	double x_limit_0_2 = findXMax(amplitudeFitFunction2,x_max_0_2,fitRanges[std::pair<int,int>(arm,station)].at(1),max_fraction,0.001); 
	double limit_0_2 = amplitudeFitFunction2->Eval(x_limit_0_2)

	lineVMax_0_2 = new TLine(x_max_0_2,0,x_max_0_2,max_0_2)
	lineVMax_0_2->SetLineColor(kBlue)
	lineVMax_0_2->SetLineWidth(3)
	lineVMax_0_2->SetLineStyle(9)
	lineVMax_0_2->Draw("same")

	lineHMax_0_2 = new TLine(xMin,max_0_2,x_max_0_2,max_0_2)
	lineHMax_0_2->SetLineColor(kBlue)
	lineHMax_0_2->SetLineWidth(3)
	lineHMax_0_2->SetLineStyle(9)
	lineHMax_0_2->Draw("same")

	lineVLimit_0_2 = new TLine(x_limit_0_2,0,x_limit_0_2,limit_0_2)
	lineVLimit_0_2->SetLineColor(kBlue)
	lineVLimit_0_2->SetLineWidth(3)
	lineVLimit_0_2->SetLineStyle(9)
	lineVLimit_0_2->Draw("same")

	lineHLimit_0_2 = new TLine(xMin,limit_0_2,x_limit_0_2,limit_0_2)
	lineHLimit_0_2->SetLineColor(kBlue)
	lineHLimit_0_2->SetLineWidth(3)
	lineHLimit_0_2->SetLineStyle(9)
	lineHLimit_0_2->Draw("same")
	
	double dist_frac_0_2 = (double)amplitudeFitFunction2->Integral(x_max_0_2,x_limit_0_2)/amplitudeFitFunction2->Integral(x_max_0_2,fitRanges[std::pair<int,int>(arm,station)].at(1))

	arm = 1
	amplitudeFitFunction2->SetParameters(0,0,0);
	h2TrackHitDistribution_arm1_st2_rp3_0->Fit("amplitudeFitFunction2","Q","",fitRanges[std::pair<int,int>(arm,station)].at(0),fitRanges[std::pair<int,int>(arm,station)].at(1));
	amplitudeFitFunction2->SetLineColor(kRed)
	amplitudeFitFunction2->DrawCopy("same");
	double x_max_1_2 = h2TrackHitDistribution_arm1_st2_rp3_0->GetBinCenter((h2TrackHitDistribution_arm1_st2_rp3_0->GetMaximumBin()))
	double max_1_2 = amplitudeFitFunction2->Eval(x_max_1_2)
	// x_limit_1_2 = amplitudeFitFunction2->GetX(max_fraction*max_1_2,x_max_1_2,fitRanges[std::pair<int,int>(arm,station)].at(1))
	double x_limit_1_2 = findXMax(amplitudeFitFunction2,x_max_1_2,fitRanges[std::pair<int,int>(arm,station)].at(1),max_fraction,0.001); 
	double limit_1_2 = amplitudeFitFunction2->Eval(x_limit_1_2)

	lineVMax_1_2 = new TLine(x_max_1_2,0,x_max_1_2,max_1_2)
	lineVMax_1_2->SetLineColor(kRed)
	lineVMax_1_2->SetLineWidth(3)
	lineVMax_1_2->SetLineStyle(9)
	lineVMax_1_2->Draw("same")

	lineHMax_1_2 = new TLine(xMin,max_1_2,x_max_1_2,max_1_2)
	lineHMax_1_2->SetLineColor(kRed)
	lineHMax_1_2->SetLineWidth(3)
	lineHMax_1_2->SetLineStyle(9)
	lineHMax_1_2->Draw("same")

	lineVLimit_1_2 = new TLine(x_limit_1_2,0,x_limit_1_2,limit_1_2)
	lineVLimit_1_2->SetLineColor(kRed)
	lineVLimit_1_2->SetLineWidth(3)
	lineVLimit_1_2->SetLineStyle(9)
	lineVLimit_1_2->Draw("same")

	lineHLimit_1_2 = new TLine(xMin,limit_1_2,x_limit_1_2,limit_1_2)
	lineHLimit_1_2->SetLineColor(kRed)
	lineHLimit_1_2->SetLineWidth(3)
	lineHLimit_1_2->SetLineStyle(9)
	lineHLimit_1_2->Draw("same")
	double dist_frac_1_2 = (double)amplitudeFitFunction2->Integral(x_max_1_2,x_limit_1_2)/amplitudeFitFunction2->Integral(x_max_1_2,fitRanges[std::pair<int,int>(arm,station)].at(1))
	cout << "\t Arm 0\t Arm1\n"<<"xMax\t"<<x_limit_0_2<<"\t"<<x_limit_1_2<<"\n"<<"Frac\t"<<dist_frac_0_2<<"\t"<<dist_frac_1_2<<endl; 

}

void UsefulCommandsSt0{

	.L FindXMax.C++
	Canvas_1->Draw();

	map< std::pair<int,int>, vector<double> > fitRanges;
	fitRanges[std::pair<int,int>(0,0)] = {7.1,18.0,7.1,12.1,7.0,11};
	fitRanges[std::pair<int,int>(0,2)] = {44.2,55.1,44.0,47.2,44,47};
	fitRanges[std::pair<int,int>(1,0)] = {6.3,17.2,6.1,11.1,6.0,8.5};
	fitRanges[std::pair<int,int>(1,2)] = {44.1,55,44.1,52.5,44,47};
	TF1* amplitudeFitFunction2 = new TF1("amplitudeFitFunction2","[0]/(x-[1])+[2]",0,100);
	amplitudeFitFunction2->SetParameters(0,0,0);
	int station = 0;
	double max_fraction = 0.3;

	double xMin = h2TrackHitDistribution_rotated_arm0_st0_rp3_0->GetXaxis()->GetXmin();

	int arm = 0;
	h2TrackHitDistribution_rotated_arm0_st0_rp3_0->Fit("amplitudeFitFunction2","N","",7.1,fitRanges[std::pair<int,int>(arm,station)].at(1));
	amplitudeFitFunction2->SetLineColor(kBlue);
	amplitudeFitFunction2->DrawCopy("same");
	double x_max_0_0 = h2TrackHitDistribution_rotated_arm0_st0_rp3_0->GetBinCenter((h2TrackHitDistribution_rotated_arm0_st0_rp3_0->GetMaximumBin()));
	double max_0_0 = amplitudeFitFunction2->Eval(x_max_0_0);
	double x_limit_0_0 = findXMax(amplitudeFitFunction2,x_max_0_0,fitRanges[std::pair<int,int>(arm,station)].at(1),max_fraction,0.001); 
	double limit_0_0 = amplitudeFitFunction2->Eval(x_limit_0_0);

	lineVMax_0_0 = new TLine(x_max_0_0,0,x_max_0_0,max_0_0)
	lineVMax_0_0->SetLineColor(kBlue)
	lineVMax_0_0->SetLineWidth(3)
	lineVMax_0_0->SetLineStyle(9)
	lineVMax_0_0->Draw("same")

	lineHMax_0_0 = new TLine(xMin,max_0_0,x_max_0_0,max_0_0)
	lineHMax_0_0->SetLineColor(kBlue)
	lineHMax_0_0->SetLineWidth(3)
	lineHMax_0_0->SetLineStyle(9)
	lineHMax_0_0->Draw("same")

	lineVLimit_0_0 = new TLine(x_limit_0_0,0,x_limit_0_0,limit_0_0)
	lineVLimit_0_0->SetLineColor(kBlue)
	lineVLimit_0_0->SetLineWidth(3)
	lineVLimit_0_0->SetLineStyle(9)
	lineVLimit_0_0->Draw("same")

	lineHLimit_0_0 = new TLine(xMin,limit_0_0,x_limit_0_0,limit_0_0)
	lineHLimit_0_0->SetLineColor(kBlue)
	lineHLimit_0_0->SetLineWidth(3)
	lineHLimit_0_0->SetLineStyle(9)
	lineHLimit_0_0->Draw("same")
	
	double dist_frac_0_0 = (double)amplitudeFitFunction2->Integral(x_max_0_0,x_limit_0_0)/amplitudeFitFunction2->Integral(x_max_0_0,fitRanges[std::pair<int,int>(arm,station)].at(1))

	arm = 1;
	amplitudeFitFunction2->SetParameters(0,0,0);
	h2TrackHitDistribution_rotated_arm1_st0_rp3_0->Fit("amplitudeFitFunction2","N","",6.3,fitRanges[std::pair<int,int>(arm,station)].at(1));
	amplitudeFitFunction2->SetLineColor(kRed);
	amplitudeFitFunction2->DrawCopy("same");
	double x_max_1_0 = h2TrackHitDistribution_rotated_arm1_st0_rp3_0->GetBinCenter((h2TrackHitDistribution_rotated_arm1_st0_rp3_0->GetMaximumBin()))
	double max_1_0 = amplitudeFitFunction2->Eval(x_max_1_0)
	// x_limit_1_2 = amplitudeFitFunction2->GetX(max_fraction*max_1_2,x_max_1_2,fitRanges[std::pair<int,int>(arm,station)].at(1))
	double x_limit_1_0 = findXMax(amplitudeFitFunction2,x_max_1_0,fitRanges[std::pair<int,int>(arm,station)].at(1),max_fraction,0.001); 
	double limit_1_0 = amplitudeFitFunction2->Eval(x_limit_1_0)

	lineVMax_1_0 = new TLine(x_max_1_0,0,x_max_1_0,max_1_0)
	lineVMax_1_0->SetLineColor(kRed)
	lineVMax_1_0->SetLineWidth(3)
	lineVMax_1_0->SetLineStyle(9)
	lineVMax_1_0->Draw("same")

	lineHMax_1_0 = new TLine(xMin,max_1_0,x_max_1_0,max_1_0)
	lineHMax_1_0->SetLineColor(kRed)
	lineHMax_1_0->SetLineWidth(3)
	lineHMax_1_0->SetLineStyle(9)
	lineHMax_1_0->Draw("same")

	lineVLimit_1_0 = new TLine(x_limit_1_0,0,x_limit_1_0,limit_1_0)
	lineVLimit_1_0->SetLineColor(kRed)
	lineVLimit_1_0->SetLineWidth(3)
	lineVLimit_1_0->SetLineStyle(9)
	lineVLimit_1_0->Draw("same")

	lineHLimit_1_0 = new TLine(xMin,limit_1_0,x_limit_1_0,limit_1_0)
	lineHLimit_1_0->SetLineColor(kRed)
	lineHLimit_1_0->SetLineWidth(3)
	lineHLimit_1_0->SetLineStyle(9)
	lineHLimit_1_0->Draw("same")
	double dist_frac_1_0 = (double)amplitudeFitFunction2->Integral(x_max_1_0,x_limit_1_0)/amplitudeFitFunction2->Integral(x_max_1_0,fitRanges[std::pair<int,int>(arm,station)].at(1))
	cout << "\t Arm 0\t Arm1\n"<<"xMax\t"<<x_limit_0_0<<"\t"<<x_limit_1_0<<"\n"<<"Frac\t"<<dist_frac_0_0<<"\t"<<dist_frac_1_0<<endl; 

}