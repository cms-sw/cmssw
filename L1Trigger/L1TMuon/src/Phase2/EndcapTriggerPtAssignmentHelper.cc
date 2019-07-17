#include "TF1.h"
#include "TGraphErrors.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "L1Trigger/L1TMuon/src/Phase2/EndcapTriggerPtAssignmentHelper.h"
#include <iostream>
#include <math.h>       /* atan */

int EndcapTriggerPtAssignmentHelper::GetEtaPartition(float eta)
{
  int neta=-1;
  if (fabs(eta)>=1.2 and fabs(eta)<1.4)
    neta=0;
  else if (fabs(eta)>=1.4 and fabs(eta)<1.6)
    neta=1;
  else if (fabs(eta)>=1.6 and fabs(eta)<1.8)
    neta=2;
  else if (fabs(eta)>=1.8 and fabs(eta)<2.0)
    neta=3;
  else if (fabs(eta)>=2.0 and fabs(eta)<2.2)
    neta=4;
  else if (fabs(eta)>=2.2 and fabs(eta)<2.4)
    neta=5;

  return neta;
}


float EndcapTriggerPtAssignmentHelper::ellipse(float a, float b, float alpha, float x0, float y0, float x, float y)
{
  float x1 = x*cos(alpha)+y*sin(alpha)-x0;
  float y1 = x*sin(alpha)-y*cos(alpha)-y0;
  return x1*x1/(a*a)+y1*y1/(b*b);
}


float EndcapTriggerPtAssignmentHelper::deltaYcalculation(const GlobalPoint& gp1,
                                                         const GlobalPoint& gp2)
{
  // ME2 is the reference station to which we want to compare
  const float ref_angle = gp2.phi();
  const float new_y_st1 = -gp1.x()*sin(ref_angle) + gp1.y()*cos(ref_angle);
  const float new_y_st2 = -gp2.x()*sin(ref_angle) + gp2.y()*cos(ref_angle);
  return (new_y_st2 - new_y_st1);
}


float EndcapTriggerPtAssignmentHelper::deltadeltaYcalculation(const GlobalPoint& gp1, const GlobalPoint& gp2, const GlobalPoint& gp3, float eta, enum EvenOdd123 parity)
{
  // ME2 is the reference station to which we want to compare
  const float ref_angle = gp2.phi();

  // rotate the coordinates of the other stations
  const float new_y_st1 = -gp1.x()*sin(ref_angle) + gp1.y()*cos(ref_angle);
  const float new_y_st2 = -gp2.x()*sin(ref_angle) + gp2.y()*cos(ref_angle);
  const float new_y_st3 = -gp3.x()*sin(ref_angle) + gp3.y()*cos(ref_angle);

  // calculate the new distances
  const float deltay12 = new_y_st2 - new_y_st1;
  const float deltay23 = new_y_st3 - new_y_st2;

  // eta sector in which this muon was found
  const int etaSector = EndcapTriggerPtAssignmentHelper::GetEtaPartition(eta);

  //FIXME: uncomment these special cases
  // if (meRing_st1 == 1 and etaSector == 1) etaSector =2;
  // if (meRing_st1 == 2 and etaSector == 2) etaSector =1;

  // do not consider an invalid parity  or an invalid eta sector
  if (parity<EndcapTriggerPtAssignmentHelper::EvenOdd123::OddEvenEven or
      parity>EndcapTriggerPtAssignmentHelper::EvenOdd123::EvenOddOdd or
      etaSector==-1) return -99;

  // get the geometric scale factor
  const float scaleFactor(EndcapTriggerPtAssignmentHelper::PositionEpLUT[int(parity)][etaSector][0]);

  // calculate
  const float deltaDeltaY123 = deltay23 - scaleFactor * deltay12;

  return deltaDeltaY123;
}

float EndcapTriggerPtAssignmentHelper::Ptassign_Direction(float bending_12, float eta, int par){
  int neta = GetEtaPartition(eta);
    if (par<0 or par>3 or neta==-1 or fabs(bending_12) > M_PI) return -1;

    float pt=(1/fabs(bending_12)+DirectionEpLUT[par][neta][1])/DirectionEpLUT[par][neta][0];
    //std::cout <<"Pt Direction, npar "<< par <<" neta "<<neta <<" slope "<< DirectionEpLUT[par][neta][0]<<" intercep "<< DirectionEpLUT[par][neta][1] << " bending_12 " << bending_12 <<" pt "<<pt <<std::endl;

    return pt;
}



float EndcapTriggerPtAssignmentHelper::PhiMomentum(float dphi, float phi_position, int st, bool evenodd){

    //even->0, odd->1
    int cham = (evenodd ? 0:1);
    float slope = BendingAngleLUT[st-1][cham];
    float phi_m = dphi*slope+phi_position;
    //std::cout <<"st "<< st <<" cham "<< cham <<" gemcsc dphi "<< dphi <<" phi position "<< phi_position <<" slope "<< slope <<" phi momentum "<< phi_m << std::endl;
    return phi_m;


}


float EndcapTriggerPtAssignmentHelper::PhiMomentum_Radius(float dphi, float phi_position, float radius_csc, float radius_gem)
{
     // usually radius_csc>radius_gem
     if (fabs(dphi) > M_PI or fabs(phi_position) > M_PI or radius_csc<radius_gem) return -9;
     float radius_diff = radius_gem-radius_csc*cos(dphi);
     float phi_diff = 0.0;
     if (fabs(radius_diff) > 0.0) phi_diff = atan(radius_csc*sin(dphi)/radius_diff);
     else phi_diff = M_PI/2.0;

     if (phi_diff <= -M_PI) phi_diff = phi_diff+2*M_PI;
     else if (phi_diff > M_PI) phi_diff = phi_diff-2*M_PI;

     float phiM = phi_position-phi_diff;
     if (phiM <= -M_PI) phiM = phiM+2*M_PI;
     else if (phiM > M_PI) phiM = phiM-2*M_PI;

     //std::cout <<" radius_csc "<< radius_csc <<" radius_gem "<< radius_gem <<" dphi "<< dphi << " phi_position "<< phi_position<<" radius_diff "<< radius_diff <<" phi_diff "<< phi_diff <<" phiM "<< phiM << std::endl;
     return phiM;

}


//dphi: local bending, phi_position: phi of GEM position, X: "x_factor"
//X: st1, X=D(GEM,CSC)*x_factor, st2: D(GEM,CSC)*x_factor/(D(ME11,ME21)*x+1)
float EndcapTriggerPtAssignmentHelper::PhiMomentum_Xfactor(float dphi, float phi_position, float X)
{

     if (fabs(dphi) > M_PI or fabs(phi_position) > M_PI) return -9;
     float y = 1.0-cos(dphi)- X;

     float phi_diff = 0.0;
     if (fabs(y) > 0.0) phi_diff = atan(sin(dphi)/y);
     else phi_diff = M_PI/2.0;

     if (phi_diff <= -M_PI) phi_diff = phi_diff+2*M_PI;
     else if (phi_diff > M_PI) phi_diff = phi_diff-2*M_PI;

     float phiM = phi_position-phi_diff;
     if (phiM <= -M_PI) phiM = phiM+2*M_PI;
     else if (phiM > M_PI) phiM = phiM-2*M_PI;

     //std::cout <<"PhiMomentum_Xfactor: dphi "<< dphi <<" phi_position "<< phi_position <<" X "<<X <<" phi_diff "<< phi_diff <<" phiM "<< phiM << std::endl;
     return phiM;

}



float EndcapTriggerPtAssignmentHelper::PhiMomentum_Xfactor_V2(float phi_CSC, float phi_GEM, float X)
{
     if (fabs(phi_CSC) > M_PI or fabs(phi_GEM) > M_PI) return -9;
     float dphi = deltaPhi(phi_CSC,phi_GEM);
     float y = 1.0-cos(dphi)- X;

     float phi_diff = 0.0;
     if (fabs(y) > 0.0) phi_diff = atan(sin(dphi)/y);
     else phi_diff = M_PI/2.0;

     if (phi_diff <= -M_PI) phi_diff = phi_diff+2*M_PI;
     else if (phi_diff > M_PI) phi_diff = phi_diff-2*M_PI;

     float phiM = phi_GEM-phi_diff;
     if (phiM <= -M_PI) phiM = phiM+2*M_PI;
     else if (phiM > M_PI) phiM = phiM-2*M_PI;

     //std::cout <<"PhiMomentum_Xfactor: dphi "<< dphi <<" phi_poistion1 "<< phi_GEM <<" phi_position2 "<< phi_CSC <<" Xfactor "<<X <<" phi_diff "<< phi_diff <<" phiM "<< phiM << std::endl;

     return phiM;

}



void EndcapTriggerPtAssignmentHelper::calculateAlphaBeta(const std::vector<float>& v,
                        const std::vector<float>& w,
                        const std::vector<float>& ev,
                        const std::vector<float>& ew,
                        const std::vector<float>& status,
                        float& alpha, float& beta)
{
  //std::cout << "size of v: "<<v.size() << std::endl;

  if (v.size()>=3) {

  float zmin;
  float zmax;
  if (v.front() < v.back()){
    zmin = v.front();
    zmax = v.back();
  }
  else{
    zmin = v.back();
    zmax = v.front();
  }

  TF1 *fit1 = new TF1("fit1","pol1",zmin,zmax);
  //where 0 = x-axis_lowest and 48 = x_axis_highest
  TGraphErrors* gr = new TGraphErrors(v.size(),&(v[0]),&(w[0]),&(ev[0]),&(ew[0]));
  gr->SetMinimum(w[2]-5*0.002);
  gr->SetMaximum(w[2]+5*0.002);

  gr->Fit(fit1,"RQ");

  alpha = fit1->GetParameter(0); //value of 0th parameter
  beta  = fit1->GetParameter(1); //value of 1st parameter

  delete fit1;
  delete gr;
  }
  else {alpha = -99; beta= 0.0;}
}



float EndcapTriggerPtAssignmentHelper::normalizePhi(float phi)
{
    float result = phi;
    if(result > float(M_PI)) result -= float(2*M_PI);
    else if (result <= -float(M_PI)) result += float(2*M_PI);
    return result;
}

EndcapTriggerPtAssignmentHelper::EvenOdd123
EndcapTriggerPtAssignmentHelper::getParity(EvenOdd eo1, EvenOdd eo2,
                                           EvenOdd eo3, EvenOdd eo4)
{
  EndcapTriggerPtAssignmentHelper::EvenOdd123 defaultValue = EvenOdd123::Invalid;

  if (eo1 != EvenOdd::Invalid and
      eo2 != EvenOdd::Invalid and
      eo3 != EvenOdd::Invalid) {

    // go though all cases
    if (eo1 == EvenOdd::Odd and eo2 == EvenOdd::Even and eo3 == EvenOdd::Even)
      return EvenOdd123::OddEvenEven;

    if (eo1 == EvenOdd::Odd and eo2 == EvenOdd::Odd and eo3 == EvenOdd::Odd)
      return EvenOdd123::OddOddOdd;

    if (eo1 == EvenOdd::Even and eo2 == EvenOdd::Even and eo3 == EvenOdd::Even)
      return EvenOdd123::EvenEvenEven;

    if (eo1 == EvenOdd::Even and eo2 == EvenOdd::Odd and eo3 == EvenOdd::Odd)
      return EvenOdd123::EvenOddOdd;
  }
  return defaultValue;
}
