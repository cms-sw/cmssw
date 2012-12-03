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
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"

class Conv4HitsReco2 {

	public:
	Conv4HitsReco2(math::XYZVector &, math::XYZVector &, math::XYZVector &, math::XYZVector &, math::XYZVector &);
	Conv4HitsReco2();
	~Conv4HitsReco2();

	// Main publics
	int ConversionCandidate(math::XYZVector&, double&, double&);
	void Reconstruct(); // Not directly called when in use
	void Dump();
	void Refresh(math::XYZVector &vPhotVertex, math::XYZVector &h1, math::XYZVector &h2, math::XYZVector &h3, math::XYZVector &h4);

	math::XYZVector GetPlusCenter(double &);
	math::XYZVector GetMinusCenter(double &);

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
	void LocalTransformation(math::XYZVector v11, math::XYZVector v12, math::XYZVector v21, math::XYZVector v22,
				 math::XYZVector &V11, math::XYZVector &V12, math::XYZVector &V21, math::XYZVector &V22,
				 double Phi);
	math::XYZVector fHitv11, fHitv12, fHitv21, fHitv22;
	math::XYZVector fPV;
	math::XYZVector fRecV, fRecC1, fRecC2;
	
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
