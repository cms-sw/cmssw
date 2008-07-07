#include "DQM/SiStripCommissioningSources/plugins/tracking/SiStripFineDelayTOF.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <DataFormats/TrackReco/interface/Track.h>
#include <iostream>
#include "TVector3.h"
#include "TMath.h"

//#define TIF_COSMIC_SETUP

double SiStripFineDelayTOF::timeOfFlight(bool cosmics, bool field, double* trackParameters, double* hit, double* phit, bool onDisk)
{

  // case 1: cosmics with no field.
  if(cosmics && !field) {
   return timeOfFlightCosmic(hit,phit);
  }
  // case 2: cosmics with field.
  else if(cosmics && field) {
   return timeOfFlightCosmicB(trackParameters,hit,phit,onDisk);
  }
  // case 3: beam with no field.
  else if(!cosmics && !field) {
   return timeOfFlightBeam(hit,phit);
  }
  // case 4: beam with field.
  else {
   return timeOfFlightBeamB(trackParameters,hit,phit,onDisk);
  }
}

double SiStripFineDelayTOF::timeOfFlightCosmic(double* hit, double* phit)
{
  // constants
  const double c = 30; // cm/ns
#ifndef TIF_COSMIC_SETUP
  const double Rmu = 385; // cm
  const double zmu = 560; // cm
  // estimate the time for crossing the barrel
  TVector3 r0(hit[0],hit[1],hit[2]);
  TVector3 pr(phit[0],phit[1],phit[2]);
  TVector2 r0_xy = r0.XYvector();
  TVector2 pr_xy = pr.Unit().XYvector();
  double t_barrel = ((r0_xy*pr_xy)+sqrt( (r0_xy*pr_xy)*(r0_xy*pr_xy) - r0_xy.Mod2() + Rmu*Rmu  ))/c*pr.Mag()/pr.Perp();
  // estimate the time for crossing endcaps
  double t_endcap = fabs(((phit[2]/fabs(phit[2])*zmu)+hit[2])/c*pr.Mag()/pr.Pz());
  // take the minimum
  return t_barrel<t_endcap ? t_barrel : t_endcap;
#else
  const double y_trigger = 100; //cm
  double p = sqrt(phit[0]*phit[0]+phit[1]*phit[1]+phit[2]*phit[2]);
//  LogDebug("TOF") << "momentum:" << phit[0] << " " << phit[1] << " " << phit[2];
//  LogDebug("TOF") << "p/py=" << p/phit[1] << "  Y0,Y,dY = " << y_trigger << " " << hit[1] << " " << (y_trigger-hit[1]); 
//  LogDebug("TOF") << "d, t=d/c : " << ((y_trigger-hit[1])*(p/phit[1])) << " " << ((y_trigger-hit[1])*(p/phit[1])/c); 
  return fabs((y_trigger-hit[1])*(p/phit[1])/c);
#endif
}

double SiStripFineDelayTOF::timeOfFlightCosmicB(double* trackParameters, double* hit, double* phit, bool onDisk)
{
  // constants
  const double Rmu = 385; // cm
  const double zmu = 560; // cm
  const double c = 30; // cm/ns
  // track parameters
  double& kappa = trackParameters[0];
  double& theta = trackParameters[1];
  double& phi_0 = trackParameters[2];
  double& d_0   = trackParameters[3];
  double& z_0   = trackParameters[4];
  double invkappa = 1/kappa;
  // computes the value of the track parameter that correspond to the hit, relative to phi_0
  //double phi = kappa*tan(theta)*(hit[2]-z_0);
  double phi = getPhi(trackParameters,hit,onDisk) - phi_0;
  // computes the value of the track parameter that correspond to the muon system, relative to phi_0
  // phi_mu = phi_mu0 - phi_0
  double phi_mu_b = (kappa>0 ? -1 : 1 ) * acos((Rmu*Rmu-d_0*d_0-2*invkappa*invkappa+2*invkappa*d_0)/(2*invkappa*d_0-2*invkappa*invkappa));
  double phi_mu_e = kappa*tan(theta)*((phit[2]<0 ? 1 : -1)*zmu-z_0);
  // estimate the time for crossing the barrel
  double t_barrel = fabs(invkappa/(c*sin(theta))*(phi-phi_mu_b));
  // estimate the time for crossing endcaps
  double t_endcap = fabs(invkappa/(c*sin(theta))*(phi-phi_mu_e));
  // take the minimum
  return t_barrel<t_endcap ? t_barrel : t_endcap;
}

double SiStripFineDelayTOF::timeOfFlightBeam(double* hit, double*)
{
  // constants
  const double c = 30; // cm/ns
  TVector3 r0(hit[0],hit[1],hit[2]);
  return r0.Mag()/c;
}

double SiStripFineDelayTOF::timeOfFlightBeamB(double* trackParameters, double* hit, double* phit, bool onDisk)
{
  // constants
  const double c = 30; // cm/ns
  // track parameters
  double& theta = trackParameters[1];
  // returns the time of flight from the origin
  return onDisk? fabs(hit[2]/(c*cos(theta))) : fabs(sqrt(hit[0]*hit[0]+hit[1]*hit[1])/(c*sin(theta)));
}

double SiStripFineDelayTOF::x(double* trackParameters, double phi)
{
  return trackParameters[3]*sin(trackParameters[2]) + (1/trackParameters[0])*(sin(phi)-sin(trackParameters[2]));
}

double SiStripFineDelayTOF::y(double* trackParameters, double phi)
{
  return -trackParameters[3]*cos(trackParameters[2]) - (1/trackParameters[0])*(cos(phi)-cos(trackParameters[2]));
}

double SiStripFineDelayTOF::z(double* trackParameters, double phi)
{
  return trackParameters[4] + (phi-trackParameters[2])/(trackParameters[0]*tan(trackParameters[1]));
}
  
double SiStripFineDelayTOF::getPhi(double* trackParameters, double* hit, bool onDisk)
{
 if(onDisk) //use z coordinate to find phi
   return trackParameters[2] + trackParameters[0]*tan(trackParameters[1])*(hit[2]-trackParameters[4]);
 else // use x,y coordinate to find phi
 {
   double phi =0;
   if(trackParameters[2]>-2.35 && trackParameters[2]<-0.78) {
     // first use y
     phi = acos(-trackParameters[0]*(hit[1]-(trackParameters[3]-1/trackParameters[0])*(-cos(trackParameters[2]))));
     // use x to resolve the ambiguity
     if(fabs(x(trackParameters,phi)-hit[0])>fabs(x(trackParameters,-phi)-hit[0])) phi *= -1.;
   } else {
     // first use x
     phi = asin(trackParameters[0]*(hit[0]-(trackParameters[3]-1/trackParameters[0])*sin(trackParameters[2])));
     // use y to resolve the ambiguity
     if((fabs(y(trackParameters,phi)-hit[1])>fabs(y(trackParameters,TMath::Pi()-phi)-hit[1]))) 
       phi = phi>0 ? TMath::Pi()-phi: -TMath::Pi()-phi;
   }
   return phi;
 }
 return 0.;
}

void SiStripFineDelayTOF::trackParameters(const reco::Track& tk,double* trackParameters)
{
      math::XYZPoint position = tk.outerPosition();
      math::XYZVector momentum = tk.outerMomentum();
      LogDebug("trackParameters") << "outer position: " << position.x() << " " << position.y() << " " << position.z();
      LogDebug("trackParameters") << "outer momentum: " << momentum.x() << " " << momentum.y() << " " << momentum.z();
      math::XYZVector fieldDirection(0,0,1);
      // computes the center of curvature
      math::XYZVector radius = momentum.Cross(fieldDirection).Unit();
      position -= radius / trackParameters[0];
      LogDebug("trackParameters") << "center of curvature: " << position.x() << " " << position.y() << " " << position.z();
      // the transverse IP
      trackParameters[3] = position.rho() - fabs(1/trackParameters[0]);
      if(trackParameters[0]>0) trackParameters[3] *= -1.;
      // phi_0
      double phi_out = trackParameters[2];
      double phi_0 = position.phi() - TMath::Pi()/2;
      if(trackParameters[0]<0) phi_0 -= TMath::Pi();
      if(phi_0<-2*TMath::Pi()) phi_0 += 2*TMath::Pi();
      if(phi_0>2*TMath::Pi()) phi_0 -= 2*TMath::Pi();
      if(phi_0 > 0) phi_0 -= 2*TMath::Pi();
      LogDebug("trackParameters") << "phi_0: " << phi_0;
      trackParameters[2] = phi_0;
      // z_0
      trackParameters[4] = tk.outerPosition().z() - (phi_out-phi_0)/TMath::Tan(trackParameters[1])/trackParameters[0];
      LogDebug("trackParameters") << "z_0: " << tk.outerPosition().z() << " " << trackParameters[4] << " " << tk.innerPosition().z();  
}

