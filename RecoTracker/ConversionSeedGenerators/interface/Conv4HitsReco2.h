//
//	Simple photon conversion seeding class (inc)
//
//	Author: E Song
//
//	Version: 1; 	6 Aug 2012
//

#ifndef Conv4HitsReco2_h
#define Conv4HitsReco2_h

#include <iostream>
#include <iomanip>
#include <math.h>
#include "TMath.h"

#include <TVector3.h>

class Conv4HitsReco2 {

	public:
	Conv4HitsReco2(TVector3 &, TVector3 &, TVector3 &, TVector3 &, TVector3 &);
	Conv4HitsReco2();
	~Conv4HitsReco2();

	// Main publics
	int ConversionCandidate(TVector3&, double&, double&);
	void Reconstruct(); // Not directly called when in use
	void Dump();
	void Refresh(TVector3 &vPhotVertex, TVector3 &h1, TVector3 &h2, TVector3 &h3, TVector3 &h4);

	TVector3 GetPlusCenter(double &);
	TVector3 GetMinusCenter(double &);

	// Optional publics
	void SetMaxNumberOfIterations(int val) { fMaxNumberOfIterations=val; };
	void SetRadiusECut(double val) { fRadiusECut=val; };
	void SetPhiECut(double val) { fPhiECut=val; };
	void SetRECut(double val) { fRECut=val; };
	void SetFixedNumberOfIterations(double val) { fFixedNumberOfIterations=val; };
	void SetBField(double val) { fBField=val; };

	double GetRecPhi() { return fRecPhi; };
	double GetRecR() { return fRecR; };	
	double GetRecR1() { return fRecR1; };
	double GetRecR2() { return fRecR2; };
	int GetLoop() { return fLoop; };

	bool RegisterUnsolvable(int &num) { if (fSolved==1) return true; else {num+=1; return false;}};
	bool RegisterUnsolvable() { if (fSolved==1) return true; else return false; };
	bool RegisterBadSign(int &num) { if (fSignSatisfied==1) return true; else {num+=1; return false;}};
	bool RegisterBadSign() { if (fSignSatisfied==1) return true; else return false; };
	bool RegisterBadConverge(int &num) { if (fCutSatisfied==1) return true; else {num+=1; return false;}};
	bool RegisterBadConverge() { if (fCutSatisfied==1) return true; else return false;};

	private:
	void LocalTransformation(TVector3 v11, TVector3 v12, TVector3 v21, TVector3 v22,
				 TVector3 &V11, TVector3 &V12, TVector3 &V21, TVector3 &V22,
				 double Phi);
	TVector3 fHitv11, fHitv12, fHitv21, fHitv22;	
	TVector3 fPV;
	TVector3 fRecV, fRecC1, fRecC2;
	
	double fRecPhi;
	double fRecR;
	double fRecR1;
	double fRecR2;	// original input coordinates in cm,

	int fCutSatisfied;	// Target cut met within iters?
	int fSignSatisfied;	// All values positive?
	int fSolved;		// No break due to /0 or no real root?

	int fMaxNumberOfIterations;
	int fLoop;	// The number of loops actually performed
	int fFixedNumberOfIterations;	// Default 0: use cuts.  If > 0, employ fixed loop.
	double fRadiusECut;
	double fPhiECut;
	double fRECut;	// Note that these cuts are NOT independent.

	double fRadiusE;
	double fPhiE;
	double fRE;
	double fBField; // tesla
};

#endif
