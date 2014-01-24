#ifndef _HOUGHLOCAL_H

#define _HOUGHLOCAL_H
#include <vector>
#include<stdint.h>
#include <TMath.h>
//#include <DCHistogramHandler.h>
#define PI 3.141592653589793
class RecoPoint;
class HoughLocal
{
public:
	
	HoughLocal(double thmin,double thmax,double rmin,double rmax,uint32_t nbintheta=8,uint32_t nbinr=8);
	~HoughLocal();
	void fill(double x,double y);
	void clear();
	double getTheta(int32_t i);
	double getR(int32_t i);
	uint16_t getValue(uint32_t i,uint32_t j);
	void findMaxima(std::vector< std::pair<uint32_t,uint32_t> >& maxval,uint32_t cut=3);
	void findMaxima(std::vector< std::pair<double,double> >& maxval,uint32_t cut=3);
	void findMaximumBins(std::vector< std::pair<double,double> >& maxval,uint32_t cut);
	void findThresholdBins(std::vector< std::pair<double,double> >& maxval,uint32_t cut);

	//	void draw(DCHistogramHandler* h,std::vector< std::pair<uint32_t,uint32_t> > *maxval=NULL);
	//	void draw(DCHistogramHandler* h,std::vector< std::pair<double,double> > *maxval);
  uint32_t getVoteMax();
	double getThetaBin(){return  theThetaBin_;}
	double getRBin(){return theRBin_;}
	double getThetaMin(){return  theThetaMin_;}
	double getRMin(){return theRMin_;}
	double getThetaMax(){return  theThetaMax_;}
	double getRMax(){return theRMax_;}
	uint32_t getNbinTheta(){ return theNbinTheta_;}
	uint32_t getNbinR(){ return theNbinR_;}
	static void PrintConvert(double theta,double r);
private:
	double theSin_[1024];
	double theCos_[1024];
	std::vector<double> theX_;
	std::vector<double> theY_;
	uint16_t theHoughImage_[1024][1024];
	double theThetaMin_;
	double theThetaMax_;
	double theRMin_,theRMax_;
	double theThetaBin_,theRBin_;
	uint32_t theNbinTheta_;
	uint32_t theNbinR_;
	uint16_t theVoteMax_;
};



#endif
