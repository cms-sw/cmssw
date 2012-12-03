//
//	Simple photon conversion seeding class (src)
//
//	Author: E Song
//
//	Version: 1; 	6 Aug 2012
//

#define Conv4HitsReco2_cxx

//#include "RecoTracker/ConversionSeedGenerators/interface/Conv4HitsReco2.h"
//#include "FWCore/MessegeLogger/interface/MessegeLogger.h"
#include "RecoTracker/ConversionSeedGenerators/interface/Conv4HitsReco2.h"
#include <time.h>

Conv4HitsReco2::Conv4HitsReco2(math::XYZVector &vPhotVertex, math::XYZVector &h1, math::XYZVector &h2, math::XYZVector &h3, math::XYZVector &h4)
{
	Refresh(vPhotVertex, h1, h2, h3, h4);
}

Conv4HitsReco2::~Conv4HitsReco2() { }
//Conv4HitsReco2::~Conv4HitsReco() { }

void Conv4HitsReco2::Refresh(math::XYZVector &vPhotVertex, math::XYZVector &h1, math::XYZVector &h2, math::XYZVector &h3, math::XYZVector &h4)
{
	// Fix 2D plane, make primary vertex the original point
	fPV = vPhotVertex;	fPV.SetZ(0.);
	fHitv11 = h3;		fHitv11.SetZ(0.);	fHitv11 = fHitv11 - fPV;
	fHitv21 = h4;		fHitv21.SetZ(0.);	fHitv21 = fHitv21 - fPV;
	fHitv12 = h2;		fHitv12.SetZ(0.);	fHitv12 = fHitv12 - fPV;
	fHitv22 = h1;		fHitv22.SetZ(0.);	fHitv22 = fHitv22 - fPV;

	// DEFAULT setup
	fMaxNumberOfIterations = 40;
	fFixedNumberOfIterations = 0;
	fRadiusECut = 10.0;//cm
	fPhiECut = 0.03;//rad
	fRECut = 0.5;//cm	
	fBField = 3.8;//T	

	// TRIVIAL initialization
	fCutSatisfied = 0;
	fSignSatisfied = 0;
	fSolved = 0;

	fRecPhi = 0.;
	fRecR = 0.;
	fRecR1 = 0.;
	fRecR2 = 0.;
	fRadiusE = 0.;
	fRE = 0.;
	fPhiE = 0.;
}

void Conv4HitsReco2::LocalTransformation(math::XYZVector v11, math::XYZVector v12, math::XYZVector v21, math::XYZVector v22,
				 	 math::XYZVector &V11, math::XYZVector &V12, math::XYZVector &V21, math::XYZVector &V22,
				 	 double NextPhi) 
{
	double x11,x12,x21,x22,y11,y12,y21,y22;

	x11 = v11.X(); y11 = v11.Y();
	x12 = v12.X(); y12 = v12.Y();
	x21 = v21.X(); y21 = v21.Y();
	x22 = v22.X(); y22 = v22.Y();

	double TANP = std::tan(NextPhi);
	double X11 = -std::fabs(x11*TANP - y11) / std::sqrt(1 + TANP*TANP);
	double Y11 = std::fabs(y11*TANP + x11) / std::sqrt(1 + TANP*TANP);

	double X21 = -std::fabs(x21*TANP - y21) / std::sqrt(1 + TANP*TANP);
	double Y21 = std::fabs(y21*TANP + x21) / std::sqrt(1 + TANP*TANP);

	double X12 = std::fabs(x12*TANP - y12) / std::sqrt(1 + TANP*TANP);
	double Y12 = std::fabs(y12*TANP + x12) / std::sqrt(1 + TANP*TANP);

	double X22 = std::fabs(x22*TANP - y22) / std::sqrt(1 + TANP*TANP);
	double Y22 = std::fabs(y22*TANP + x22) / std::sqrt(1 + TANP*TANP);

	V11.SetXYZ(X11,Y11,0.);
	V12.SetXYZ(X12,Y12,0.);
	V21.SetXYZ(X21,Y21,0.);
	V22.SetXYZ(X22,Y22,0.);
}


int Conv4HitsReco2::ConversionCandidate(math::XYZVector &vtx, double &ptplus, double &ptminus)
{
	Reconstruct();

	ptplus = fRecR2 * fBField * 0.01 * 0.3;		// 2 - positron
	ptminus = fRecR1 * fBField * 0.01 * 0.3;	// 1 - electron
	vtx = fRecV;
	//std::cout << ".";
	return fLoop;
}


void Conv4HitsReco2::Reconstruct()
{
	double x11,x12,x21,x22,y11,y12,y21,y22;
	//double X11,X12,X21,X22,Y11,Y12,Y21,Y22;
	x11 = fHitv11.X();	y11 = fHitv11.Y();
	x12 = fHitv12.X();	y12 = fHitv12.Y();
	x21 = fHitv21.X();	y21 = fHitv21.Y();
	x22 = fHitv22.X();	y22 = fHitv22.Y(); 

	if (fFixedNumberOfIterations==0)
	fLoop = fMaxNumberOfIterations;
	else fLoop = fFixedNumberOfIterations;

	// Setting Phi1, Phi2 initial guess range, and first guess
	double tempr1 = std::sqrt(y11*y11 + x11*x11);
	double tempr2 = std::sqrt(y12*y12 + x12*x12);

	double Phi1 = 2.0 * std::atan(y11 / (x11+tempr1));
	double Phi2 = 2.0 * std::atan(y12 / (x12+tempr2));

	if (Phi1<Phi2) Phi1 += 2.0 * 3.141592653;	// stupid Atan correction

	fPhiE = std::fabs((Phi1-Phi2)) / std::pow(2.0, fLoop + 1);

	double NextPhi = ( Phi1 + Phi2 ) / 2.0;	// first guess
	double D1, D2 = 0.0;
	double prevR1 = 0; double prevR2 = 0;
	double R1 = 0; double R2 = 0;

	// Iterations
	for (int i=0; i<fLoop; i++) {

		// LOCAL TRANFORMATION & EXTRACTION
		double SINP = std::sin(NextPhi);
		double COSP = std::cos(NextPhi);
		double SignCOSP = 1.; if(COSP < 0.) SignCOSP = -1.;
 		double AbsCOSP = std::fabs(COSP);

		double X11 = -std::fabs(x11*SINP*SignCOSP - y11*AbsCOSP);
		double Y11 =  std::fabs(y11*SINP*SignCOSP + x11*AbsCOSP);

		double X21 = -std::fabs(x21*SINP*SignCOSP - y21*AbsCOSP);
		double Y21 =  std::fabs(y21*SINP*SignCOSP + x21*AbsCOSP);

		double X12 =  std::fabs(x12*SINP*SignCOSP - y12*AbsCOSP);
		double Y12 =  std::fabs(y12*SINP*SignCOSP + x12*AbsCOSP);

		double X22 =  std::fabs(x22*SINP*SignCOSP - y22*AbsCOSP);
		double Y22 =  std::fabs(y22*SINP*SignCOSP + x22*AbsCOSP);
		// I'm not using LocalTransform() function because this direct way turns out to be faster




		// SOLVING EQUATIONS
		double d1 = Y21 - Y11;
		double d2 = Y22 - Y12;

		if ( ( (X11*X11*d1*d1/(X21-X11)/(X21-X11) + X11*X21 + X11*d1*d1/(X21-X11)) < 0 ) || 
			 ( (X12*X12*d2*d2/(X22-X12)/(X22-X12) + X12*X22 + X12*d2*d2/(X22-X12)) < 0 )    )	
			{ fSolved = -1;	fLoop = i;	return; } // No real root.  Break out.
	
		else {
			fSolved = 1;
			D1 = X11*d1/(X21-X11);
			D1 = D1 + std::sqrt(X11*X11*d1*d1/(X21-X11)/(X21-X11) + X11*X21 + X11*d1*d1/(X21-X11));
			D2 = X12*d2/(X22-X12);
			D2 = D2 + std::sqrt(X12*X12*d2*d2/(X22-X12)/(X22-X12) + X12*X22 + X12*d2*d2/(X22-X12));
	
			R1 = std::fabs((X11+X21)/2.0+(D1+d1/2.0)*d1/(X21-X11));
			R2 = std::fabs((X12+X22)/2.0+(D2+d2/2.0)*d2/(X22-X12));

			if ((Y11-D1)>=(Y12-D2)) {  // Moving RIGHT
				Phi1 = NextPhi;
				Phi2 = Phi2;
				NextPhi = (Phi1+Phi2)/2.0;
			}
			else if ((Y11-D1)<(Y12-D2)) {  // Moving LEFT
				Phi1 = Phi1;
				Phi2 = NextPhi;
				NextPhi = (Phi1+Phi2)/2.0; 
			} 
			
			// CHECK STOP CONDITION
			double tmpPhiE = std::fabs(Phi1-Phi2);
			double tmpRE = std::fabs( (Y11 - D1) - (Y12 - D2) );
			double tmpRadiusE = ( std::fabs(R1-prevR1) + std::fabs(R2-prevR2) ) / 2.;
			
			// A. Cut threshold satisfied - STOP - record
			if (( tmpPhiE <= fPhiECut ) && ( tmpRE <= fRECut ) && ( tmpRadiusE <= fRadiusECut ) && ( fFixedNumberOfIterations ==0 ))
			{
				fSolved = 1;  
				fCutSatisfied = 1; 
				fLoop = i+1;
				fPhiE = tmpPhiE; 
				fRE = tmpRE; 	
				fRadiusE = tmpRadiusE;				
				fRecR1 = R1; 
				fRecR2 = R2; 
				fRecR = ( (Y11 - D1) + (Y12 - D2) ) / 2.0; 
				fRecPhi = NextPhi;
				fRecV.SetX( fRecR * cos(fRecPhi) );
				fRecV.SetY( fRecR * sin(fRecPhi) );
				fRecC1.SetXYZ( fRecV.X()-fRecR1*sin(fRecPhi), fRecV.Y()+fRecR1*cos(fRecPhi), 0.);
				fRecC2.SetXYZ( fRecV.X()+fRecR2*sin(fRecPhi), fRecV.Y()-fRecR2*cos(fRecPhi), 0.);
				fRecV = fRecV + fPV;
				fRecC1 = fRecC1 + fPV;
				fRecC2 = fRecC2 + fPV;
				fCutSatisfied = 1;
				if ( (R1>0)&&(R2>0)&&(D1>0)&&(D2>0)&&((Y11-D1)>0)&&((Y12-D2)>0) ) fSignSatisfied = 1;
				else fSignSatisfied = 0;
			}
			else if (i==fLoop-1) {
				fSolved = 1;  
				fCutSatisfied = 1; 
				fLoop = i+1;
				fPhiE = tmpPhiE; 
				fRE = tmpRE; 	
				fRadiusE = tmpRadiusE;				
				fRecR1 = R1; 
				fRecR2 = R2; 
				fRecR = ( (Y11 - D1) + (Y12 - D2) ) / 2.0; 
				fRecPhi = NextPhi;
				fRecV.SetX( fRecR * cos(fRecPhi) );
				fRecV.SetY( fRecR * sin(fRecPhi) );
				fRecC1.SetXYZ( fRecV.X()-fRecR1*sin(fRecPhi), fRecV.Y()+fRecR1*cos(fRecPhi), 0.);
				fRecC2.SetXYZ( fRecV.X()+fRecR2*sin(fRecPhi), fRecV.Y()-fRecR2*cos(fRecPhi), 0.);
				fRecV = fRecV + fPV;
				fRecC1 = fRecC1 + fPV;
				fRecC2 = fRecC2 + fPV;
				fCutSatisfied = 0;
				if ( (R1>0)&&(R2>0)&&(D1>0)&&(D2>0)&&((Y11-D1)>0)&&((Y12-D2)>0) ) fSignSatisfied = 1;
				else fSignSatisfied = 0;
			}
			// B. Cut threshold NOT satisfied - prepare for next loop
			prevR1 = R1;   prevR2 = R2; 

		}
	}
}


void Conv4HitsReco2::Dump()
{
	std::cout << std::endl<< "================================================" << std::endl;
	std::cout << "	Nothing happend here.";
	if (fSolved==1) std::cout << "Solved.";
	if (fCutSatisfied==1) std::cout << "Cut good.";
	if (fSignSatisfied==1) std::cout << "Sign good.";
	
}

math::XYZVector Conv4HitsReco2::GetPlusCenter(double &plusR)
{
	plusR = fRecR1;
	return fRecC1;
	
}
math::XYZVector Conv4HitsReco2::GetMinusCenter(double &minusR)
{
	minusR = fRecR2;
	return fRecC2;
}
