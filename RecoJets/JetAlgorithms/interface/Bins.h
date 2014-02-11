#ifndef BINS_H
#define BINS_H

#include "TMath.h"


class Bins{
public:
const static int nRhoBins=40; //18
const static int nPtBins=20; //20
const static int Pt0=20;  //20
const static int Pt1=2000; //1000
const static int Rho0=0;  //2
const static int Rho1=40; //20
const static int PtLastExtend=4000; //3500

static int getBins(double  *bins,int nBins,double MinBin=15.0,double MaxBin=1000.,bool log=false);
static int getBin(int nBins,double  *bins,double value,double*x0=0,double*x1=0);
static void getBins_int( int nBins_total, Double_t* Lower, Double_t xmin, Double_t xmax, bool plotLog=true); 

};



#endif
