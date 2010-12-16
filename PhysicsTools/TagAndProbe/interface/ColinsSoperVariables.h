#define CM_ENERGY 7000.0
#include "TLorentzVector.h" 
#include "TVector3.h"


// calculate the Colins-Soper variables;
// everything is in the lab frame
void calCSVariables(TLorentzVector mu, TLorentzVector mubar, 
		    double *res, bool swap) {

  // convention. beam direction is on the positive Z direction.
  // beam contains quark flux.
  TLorentzVector Pbeam  (0, 0,  CM_ENERGY/2.0, CM_ENERGY/2.0);
  TLorentzVector Ptarget(0, 0, -CM_ENERGY/2.0, CM_ENERGY/2.0);
  

  TLorentzVector Q(mu+mubar);
  /************************************************************************
   *
   * 1) cos(theta) = 2 Q^-1 (Q^2+Qt^2)^-1 (mu^+ mubar^- - mu^- mubar^+)
   *
   *
   ************************************************************************/
  double muplus  = 1.0/sqrt(2.0) * (mu.E() + mu.Z());
  double muminus = 1.0/sqrt(2.0) * (mu.E() - mu.Z());

  double mubarplus  = 1.0/sqrt(2.0) * (mubar.E() + mubar.Z());
  double mubarminus = 1.0/sqrt(2.0) * (mubar.E() - mubar.Z());
 
  double costheta = 2.0 / Q.Mag() / sqrt(pow(Q.Mag(), 2) + pow(Q.Pt(), 2)) * 
    (muplus * mubarminus - muminus * mubarplus);
  if (swap) costheta = -costheta;



  /************************************************************************
   *
   * 2) sin2(theta) = Q^-2 Dt^2 - Q^-2 (Q^2 + Qt^2)^-1 * (Dt dot Qt)^2
   *
   ************************************************************************/
  TLorentzVector D(mu-mubar);
  double dt_qt = D.X()*Q.X() + D.Y()*Q.Y();
  double sin2theta = pow(D.Pt()/Q.Mag(), 2)
    - 1.0/pow(Q.Mag(), 2)/(pow(Q.Mag(), 2) + pow(Q.Pt(), 2))*pow(dt_qt, 2);

 

  /************************************************************************
   *
   * 3) tanphi = (Q^2 + Qt^2)^1/2 / Q (Dt dot R unit) /(Dt dot Qt unit)
   *
   ************************************************************************/
  // unit vector on R direction
  TVector3 R = Pbeam.Vect().Cross(Q.Vect());
  TVector3 Runit = R.Unit();


  // unit vector on Qt
  TVector3 Qt = Q.Vect(); Qt.SetZ(0);
  TVector3 Qtunit = Qt.Unit();


  TVector3 Dt = D.Vect(); Dt.SetZ(0);
  double tanphi = sqrt(pow(Q.Mag(), 2) + pow(Q.Pt(), 2)) / Q.Mag() * 
    Dt.Dot(Runit) / Dt.Dot(Qtunit);
  if (swap) tanphi = -tanphi;

  res[0] = costheta;
  res[1] = sin2theta;
  res[2] = tanphi;
}

