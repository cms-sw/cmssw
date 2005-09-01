// jetCalibV1.cc
// 
// Description
// -----------
// This function will get jet calibrations that correct a recontructed 
// jets four vector, after pileup, back to the four vector of the generated 
// particles in the jet cone, before pileup.  They were determined for jets
// made from EcalPluHcalTowers with a threshold of 0.5 GeV, iterative 
// cone algorithm, cone size R=0.5, and pileup for lum=2x10^33.
// Determined from a sample of QCD jet production: DC04 PCP, Pythia, 
// Oscar 2_4_5, ORCA_8_1_3.  Although determined from a QCD sample,
// they are true response functions, so they should be applicable mainly
// for flat spectra or resonances: they were determined in bins of generated
// jet pt and therefore do not include feedown effects from a fallinq QCD
// spectrum. They have been tested on the sample from which they were 
// derived, and for Jet Pt>40 GeV, they flatten the response vs Pt to 
// within 1% in the barrel, and they flatten the response vs eta to within a
// few percent.
//
// Usage
// -----
// float jetCor = jetCalibV1(jetPt, jetEta)
//
// float jetCor = returned multiplicative jet correction.  To calibrate the
//                jet E, px, py, or pz, simply multiply them bv jetCor.
// float jetPt = input jet pt
// float jetEta = input jet pseudorapidity
//
// History
// -------
// 27-Sep-04   R. Harris   Initial version
//
#include "RecoJets/JetAlgorithms/interface/jetCalibV1.h"
#include "RecoJets/JetAlgorithms/interface/jetCalibV1Param.h"
#include <iostream>
using namespace std;

// Function to get jet corrections
// --------------------------------
float jetCalibV1(float jetPt, float jetEta)
{
  float jetCor;
  float corPt=jetPt;  //First approximation to gen Pt.
  // Since the response is parameterized as a function of gen Pt (corrected Pt),
  // and we only have the reconstructed Pt, we must iterate a few times to get 
  // the Pt correction.  The correction becomes stable to better than 0.1% after 
  // 5 iterations.
  float etaResponse = etaParam(jetPt,jetEta); //Response vs |eta|, relative to |eta|<1.0
  corPt = jetPt/etaResponse;                  //Correct the jet back to |eta|<1.0
  float ptResponse;
  for(int i=1; i<=5; i++){ 
    ptResponse = ptParam(corPt);        //Response vs pt for |eta|<1.0
    corPt=jetPt/(ptResponse*etaResponse);     //The corrected Pt 
    //cout << "pass " << i << ", pt Response = " << ptResponse << endl; 
  }
  jetCor=1.0/(ptResponse*etaResponse);       //Jet Correction at this pt and eta
  return jetCor;
}


// The jet response vs pt: (rec jet pt/ gen jet pt) vs gen jet pt for |rec eta|<1
// ------------------------------------------------------------------------------
float ptParam(float jetPt)
{
  // The ptParam used was a quadratic in the logarithm of the Pt, one for high Pt
  // where the response rose with increasing Pt, and one for low PT where the response
  // rose with decreasing Pt.  The parameterizations are constrained to the same value
  // at the cross point in Pt where the parameterizations meet.
  float response;
  if(jetPt<extrapLoPt){
    // if jet Pt beneath extrapolation region, hold correction at lowest point of extrapolation region.
    response=responseCross +  ptLoPar[0]*log(extrapLoPt/ptCross)+ptLoPar[1]*pow(log(extrapLoPt/ptCross),2);
  }
  else if(jetPt>=extrapLoPt && jetPt<ptCross){
    // jet Pt in region of lower Pt paramaterization
    response=responseCross + ptLoPar[0]*log(jetPt/ptCross)+ptLoPar[1]*pow(log(jetPt/ptCross),2);
  }
  else if(jetPt>=ptCross && jetPt<fitHiPt){
    // jet Pt in region of higher Pt paramaterization
    response=responseCross + ptHiPar[0]*log(jetPt/ptCross)+ptHiPar[1]*pow(log(jetPt/ptCross),2);
  }
  else if(jetPt>=fitHiPt){
    // jet Pt above fitted region, hold correction at highest fitted point.
    response=responseCross + ptHiPar[0]*log(fitHiPt/ptCross)+ptHiPar[1]*pow(log(fitHiPt/ptCross),2);
  }
  else{
    // this should never happen
    response = 1.;
    cout << "Error in determining jet correction. No correction applied." << endl;
  }     
  return response;
}


// The jet response vs rec eta, relative to the response for |rec eta|<1.0
float etaParam(float jetPt, float jetEta)
{
// Parameterization is a polynomial up to a maximum value of |eta|, and then flat at higher |eta|.
// We interpolate smoothly between the different polynomials.
   float response;
   int firstBin;
   int nextBin;
   float firstResponse;
   float nextResponse;
   if(jetPt < meanPt[0]){         firstBin=0;     nextBin=1;
     firstResponse=polynomial(etaParPt0, min(abs(jetEta),etaMaxPt0));
     nextResponse=firstResponse;
   }
   else if(jetPt < meanPt[1]){     firstBin=0;     nextBin=1;
     firstResponse=polynomial(etaParPt0, min(abs(jetEta),etaMaxPt0));
     nextResponse=polynomial(etaParPt1, min(abs(jetEta),etaMaxPt1));
   }
   else if(jetPt < meanPt[2]){     firstBin=1;     nextBin=2;
     firstResponse=polynomial(etaParPt1, min(abs(jetEta),etaMaxPt1));
     nextResponse=polynomial(etaParPt2, min(abs(jetEta),etaMaxPt2));
   }
   else if(jetPt < meanPt[3]){     firstBin=2;     nextBin=3;
     firstResponse=polynomial(etaParPt2, min(abs(jetEta),etaMaxPt2));
     nextResponse=polynomial(etaParPt3, min(abs(jetEta),etaMaxPt3));
   }
   else if(jetPt < meanPt[4]){     firstBin=3;     nextBin=4;
     firstResponse=polynomial(etaParPt3, min(abs(jetEta),etaMaxPt3));
     nextResponse=polynomial(etaParPt4, min(abs(jetEta),etaMaxPt4));
   }
   else if(jetPt < meanPt[5]){     firstBin=4;     nextBin=5;
     firstResponse=polynomial(etaParPt4, min(abs(jetEta),etaMaxPt4));
     nextResponse=polynomial(etaParPt5, min(abs(jetEta),etaMaxPt5));
   }
   else if(jetPt < meanPt[6]){     firstBin=5;     nextBin=6;
     firstResponse=polynomial(etaParPt5, min(abs(jetEta),etaMaxPt5));
     nextResponse=polynomial(etaParPt6, min(abs(jetEta),etaMaxPt6));
   }
   else if(jetPt < meanPt[7]){     firstBin=6;     nextBin=7;
     firstResponse=polynomial(etaParPt6, min(abs(jetEta),etaMaxPt6));
     nextResponse=polynomial(etaParPt7, min(abs(jetEta),etaMaxPt7));
   }
   else if(jetPt < meanPt[8]){     firstBin=7;     nextBin=8;
     firstResponse=polynomial(etaParPt7, min(abs(jetEta),etaMaxPt7));
     nextResponse=polynomial(etaParPt8, min(abs(jetEta),etaMaxPt8));
   }
   else                        {     firstBin=8;     nextBin=7;
     firstResponse=polynomial(etaParPt8, min(abs(jetEta),etaMaxPt8));
     nextResponse=firstResponse;
   }
   response= firstResponse + (nextResponse-firstResponse)*(jetPt-meanPt[firstBin])/(meanPt[nextBin]-meanPt[firstBin]);     
   return response;
}

// A General 8th order polynomial
// ------------------------------
float polynomial(float etaPar[9], float jetEta)
{ 
  return etaPar[0]+ \
         etaPar[1]*jetEta+ \
	 etaPar[2]*pow(jetEta,2)+\
	 etaPar[3]*pow(jetEta,3)+\
	 etaPar[4]*pow(jetEta,4)+\
	 etaPar[5]*pow(jetEta,5)+\
	 etaPar[6]*pow(jetEta,6)+\
	 etaPar[7]*pow(jetEta,7)+\
	 etaPar[8]*pow(jetEta,8);
}
	 
