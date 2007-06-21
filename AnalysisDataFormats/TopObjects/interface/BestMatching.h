#ifndef TopObjects_BestMatching_h
#define TopObjects_BestMatching_h

#include <vector>
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h" // Andrea

inline double theta(double eta) {
  return 2*atan(exp(-eta));
}

inline double deltaPhi(double phi1, double phi2) {
  double dPhi = phi1 - phi2;
  if (dPhi > 3.1415927)  dPhi =  2*3.1415927 - dPhi;
  if (dPhi < -3.1415927) dPhi = -2*3.1415927 - dPhi;
  return dPhi;
}


inline double deltaEta(double eta1, double eta2) {
  return eta1 - eta2;
}


inline double deltaTheta(double theta1, double theta2) {
  return theta1 - theta2;
}


inline double deltaR(double eta1, double phi1, double eta2, double phi2) {
  return sqrt(pow(deltaEta(eta1, eta2), 2) + pow(deltaPhi(phi1, phi2), 2));
}


inline double deltaSpA(double theta1, double phi1, double theta2, double phi2) {
  return acos(sin(theta1) * cos(phi1) * sin(theta2) * cos(phi2)
	      + sin(theta1) * sin(phi1) * sin(theta2) * sin(phi2)
	      + cos(theta1) * cos(theta2));
}


inline std::vector<double> BestMatch(TtSemiEvtSolution &sol, bool useSpaceAngle) {
  std::vector<double> output;
  output.clear(); 
  double dRHadb, dRHadpp, dRHadqq, dRHadpq, dRHadqp, dRLepb;
  if (useSpaceAngle) {
    dRHadpp = deltaSpA(sol.getGenHadp().theta(),sol.getGenHadp().phi(), 
    		       sol.getCalHadp().theta(),sol.getCalHadp().phi());
    dRHadqq = deltaSpA(sol.getGenHadq().theta(),sol.getGenHadq().phi(), 
    		       sol.getCalHadq().theta(),sol.getCalHadq().phi());
    dRHadpq = deltaSpA(sol.getGenHadp().theta(),sol.getGenHadp().phi(), 
    		       sol.getCalHadq().theta(),sol.getCalHadq().phi());
    dRHadqp = deltaSpA(sol.getGenHadq().theta(),sol.getGenHadq().phi(), 
    		       sol.getCalHadp().theta(),sol.getCalHadp().phi());
    dRHadb  = deltaSpA(sol.getGenHadb().theta(),sol.getGenHadb().phi(), 
    		       sol.getCalHadb().theta(),sol.getCalHadb().phi());
    dRLepb  = deltaSpA(sol.getGenLepb().theta(),sol.getGenLepb().phi(), 
    		       sol.getCalLepb().theta(),sol.getCalLepb().phi());
  } else {
    dRHadpp  =  deltaR(sol.getGenHadp().eta(),sol.getGenHadp().phi(), 
    		       sol.getCalHadp().eta(),sol.getCalHadp().phi());
    dRHadqq  =  deltaR(sol.getGenHadq().eta(),sol.getGenHadq().phi(), 
    		       sol.getCalHadq().eta(),sol.getCalHadq().phi());
    dRHadpq  =  deltaR(sol.getGenHadp().eta(),sol.getGenHadp().phi(), 
    		       sol.getCalHadq().eta(),sol.getCalHadq().phi());
    dRHadqp  =  deltaR(sol.getGenHadq().eta(),sol.getGenHadq().phi(), 
    		       sol.getCalHadp().eta(),sol.getCalHadp().phi());
    dRHadb   =  deltaR(sol.getGenHadb().eta(),sol.getGenHadb().phi(), 
    		       sol.getCalHadb().eta(),sol.getCalHadb().phi());
    dRLepb   =  deltaR(sol.getGenLepb().eta(),sol.getGenLepb().phi(), 
    		       sol.getCalLepb().eta(),sol.getCalLepb().phi());
  }
  Int_t change = 0;
  double totDR1 = dRLepb + dRHadpp + dRHadqq + dRHadb;
  double totDR2 = dRLepb + dRHadpq + dRHadqp + dRHadb;
  if (totDR1 > totDR2) {totDR1 = totDR2; change = 1; dRHadpp=dRHadpq; dRHadqq =dRHadqp;};  
  output.push_back(totDR1);
  output.push_back(change*1.);
  output.push_back(dRHadpp);
  output.push_back(dRHadqq);
  output.push_back(dRHadb);
  output.push_back(dRLepb);
  return output;
}

inline std::vector<double> BestMatch(StEvtSolution &sol, bool useSpaceAngle) { // Andrea
  std::vector<double> output;
  output.clear(); 
  double dRBB, dRLL, dRBL, dRLB;
  if (useSpaceAngle) {
    dRBB = deltaSpA(sol.getGenBottom().theta(),sol.getGenBottom().phi(), 
    		       sol.getCalBottom().theta(),sol.getCalBottom().phi());
    dRLL = deltaSpA(sol.getGenLight().theta(),sol.getGenLight().phi(), 
    		       sol.getCalLight().theta(),sol.getCalLight().phi());
    dRBL = deltaSpA(sol.getGenBottom().theta(),sol.getGenBottom().phi(), 
    		       sol.getCalLight().theta(),sol.getCalLight().phi());
    dRLB = deltaSpA(sol.getGenLight().theta(),sol.getGenLight().phi(), 
    		       sol.getCalBottom().theta(),sol.getCalBottom().phi());
  } else {
    dRBB = deltaSpA(sol.getGenBottom().eta(),sol.getGenBottom().phi(), 
    		       sol.getCalBottom().eta(),sol.getCalBottom().phi());
    dRLL = deltaSpA(sol.getGenLight().eta(),sol.getGenLight().phi(), 
    		       sol.getCalLight().eta(),sol.getCalLight().phi());
    dRBL = deltaSpA(sol.getGenBottom().eta(),sol.getGenBottom().phi(), 
    		       sol.getCalLight().eta(),sol.getCalLight().phi());
    dRLB = deltaSpA(sol.getGenLight().eta(),sol.getGenLight().phi(), 
    		       sol.getCalBottom().eta(),sol.getCalBottom().phi());
    /*
    dRHadpp  =  deltaR(sol.getGenHadp().eta(),sol.getGenHadp().phi(), 
    		       sol.getCalHadp().eta(),sol.getCalHadp().phi());
    dRHadqq  =  deltaR(sol.getGenHadq().eta(),sol.getGenHadq().phi(), 
    		       sol.getCalHadq().eta(),sol.getCalHadq().phi());
    dRHadpq  =  deltaR(sol.getGenHadp().eta(),sol.getGenHadp().phi(), 
    		       sol.getCalHadq().eta(),sol.getCalHadq().phi());
    dRHadqp  =  deltaR(sol.getGenHadq().eta(),sol.getGenHadq().phi(), 
    		       sol.getCalHadp().eta(),sol.getCalHadp().phi());
    dRHadb   =  deltaR(sol.getGenHadb().eta(),sol.getGenHadb().phi(), 
    		       sol.getCalHadb().eta(),sol.getCalHadb().phi());
    dRLepb   =  deltaR(sol.getGenLepb().eta(),sol.getGenLepb().phi(), 
    		       sol.getCalLepb().eta(),sol.getCalLepb().phi());
    */
  }
  Int_t change = 0;
  double totDR1 = dRBB + dRLL;
  double totDR2 = dRBL + dRLB;
  if (totDR1 > totDR2) {totDR1 = totDR2; change = 1; dRBB=dRBL; dRLL =dRLB;};  
  output.push_back(totDR1);
  output.push_back(change*1.);
  output.push_back(dRBB);
  output.push_back(dRLL);
  return output;
}

#endif
