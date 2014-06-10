//#include "UserTreeAnalysis.H"   // remove if copied to user working directory
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <iostream>
#include "MC4Vector.H"
#include "HEPParticle.H"
#include "TH1.h"
#include "Setup.H"
#include "TObjArray.h"
#include "TMath.h"

using namespace std;


// very similar to  MC_FillUserHistogram from Generate.cxx
inline void fillUserHisto(char *name,double val, double weight=1.0, 
                          double min=Setup::bin_min[0][0], 
                          double max=Setup::bin_max[0][0]){

    TH1D *h=(TH1D*)(Setup::user_histograms->FindObject(name));
    if(!h){
      h=new TH1D(name,name,Setup::nbins[0][0],min,max);
      if(!h) return;
      Setup::user_histograms->Add(h);
      //      printf("user histogram created %s\n", name);
    }
    h->Fill(val,weight);

}


void print(double * v){
  cout << "("<<v[0]<<","<<v[1]<<","<<v[2]<<",E: "<<v[3]<<")"<<endl;
}

double normalised_cross_product(double * v1, double * v2, double * result){
  result[0] = v1[1]*v2[2] - v1[2]*v2[1];
  result[1] = v1[2]*v2[0] - v1[0]*v2[2];
  result[2] = v1[0]*v2[1] - v1[1]*v2[0];

  double normalisation = sqrt(result[0]*result[0]
			      +result[1]*result[1]
			      +result[2]*result[2]);

  for(int i=0; i<3; i++)
    result[i]=result[i]/normalisation;

  return normalisation;
}

double dot_product(double *v1, double *v2){
  return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
}

double dot_product(MC4Vector v1, MC4Vector v2){
  return v1.GetX1()*v2.GetX1()+v1.GetX2()*v2.GetX2()+v1.GetX3()*v2.GetX3();
}

double magnitude(double *v){
  return sqrt(dot_product(v,v));
}


/** Main function. This does not take any parameters. It assumes
    the events are something -> tau+ tau-, then tau -> pi+/- pi0 nu */
int RhoRhoUserTreeAnalysis(HEPParticle *mother,
			   HEPParticleList *stableDaughters, 
			   int nparams, double *params)
{
    assert(mother!=0);
    assert(stableDaughters!=0);

    HEPParticleListIterator daughters(*stableDaughters);

    //make temporary 4 vectors for the pions
    double pi_plus[4]={0};
    double pi_minus[4]={0};
    double pi0_plus[4]={0};
    double pi0_minus[4]={0};

    MC4Vector d_pi0_plus;
    MC4Vector d_pi0_minus;
    MC4Vector d_pi_plus;
    MC4Vector d_pi_minus;

    //make temporary variables to store the center of mass of
    //the rho+ rho- pair.
    double cm_px=0;
    double cm_py=0;
    double cm_pz=0;
    double cm_e=0;


    //loop over all daughters and sort them by type, filling the
    //temporary variables.
    for (HEPParticle *part=daughters.first(); part!=0; part=daughters.next()){

      if(part->GetPDGId()!=16&&part->GetPDGId()!=-16){
	//Get the center of mass
	cm_px+=part->GetPx();
	cm_py+=part->GetPy();
	cm_pz+=part->GetPz();
	cm_e+=part->GetE();
      }
      MC4Vector p4=part->GetP4();
      switch(part->GetPDGId()){
         case 211:
	   d_pi_plus.Set(&p4);
	   d_pi_plus.SetM(part->GetM());   
	   break;
         case -211:
	   d_pi_minus.Set(&p4);
	   d_pi_minus.SetM(part->GetM());
	   break;
         case 111: //fill the pi0's randomly for the moment.
	   if(d_pi0_minus.GetX1()==0&&d_pi0_minus.GetX2()==0&&d_pi0_minus.GetX3()==0){
	     d_pi0_minus.Set(&p4);
	     d_pi0_minus.SetM(part->GetM());
	   }
	   else{
	     d_pi0_plus.Set(&p4);
	     d_pi0_plus.SetM(part->GetM());
	   }	
	   break;
      }
    }

    //Now check which pi0 is associated with
    //which pi+/-. Use the angle to decide.
    double costheta1 = dot_product(d_pi_plus,d_pi0_plus)/(d_pi_plus.Length()*d_pi0_plus.Length());
    double costheta2 = dot_product(d_pi_minus,d_pi0_plus)/(d_pi_minus.Length()*d_pi0_plus.Length());


    if(costheta1<costheta2){ //and if necessary, swap the pi0 vectors
      MC4Vector temp;
      temp.Set(&d_pi0_plus);
      temp.SetM(d_pi0_plus.GetM());
      d_pi0_plus.Set(&d_pi0_minus);
      d_pi0_plus.SetM(d_pi0_minus.GetM());
      d_pi0_minus.Set(&temp);
      d_pi0_minus.SetM(temp.GetM());
    }


    //Now boost everything into the rho+ rho- center of mass frame.
    double cm_mass = sqrt(cm_e*cm_e-cm_px*cm_px-cm_py*cm_py-cm_pz*cm_pz);

    d_pi0_plus.Boost(cm_px,cm_py,cm_pz,cm_e,cm_mass);
    d_pi0_minus.Boost(cm_px,cm_py,cm_pz,cm_e,cm_mass);
    d_pi_plus.Boost(cm_px,cm_py,cm_pz,cm_e,cm_mass);
    d_pi_minus.Boost(cm_px,cm_py,cm_pz,cm_e,cm_mass);
 
    pi0_plus[0]=d_pi0_plus.GetX1();
    pi0_plus[1]=d_pi0_plus.GetX2();
    pi0_plus[2]=d_pi0_plus.GetX3();
    pi0_plus[3]=d_pi0_plus.GetM();

    pi_plus[0]=d_pi_plus.GetX1();
    pi_plus[1]=d_pi_plus.GetX2();
    pi_plus[2]=d_pi_plus.GetX3();
    pi_plus[3]=d_pi_plus.GetM();

    pi0_minus[0]=d_pi0_minus.GetX1();
    pi0_minus[1]=d_pi0_minus.GetX2();
    pi0_minus[2]=d_pi0_minus.GetX3();
    pi0_minus[3]=d_pi0_minus.GetM();

    pi_minus[0]=d_pi_minus.GetX1();
    pi_minus[1]=d_pi_minus.GetX2();
    pi_minus[2]=d_pi_minus.GetX3();
    pi_minus[3]=d_pi_minus.GetM();


    /******* calculate acoplanarity (theta) *****/
    //normal to the plane spanned by pi+ pi0 
    double n_plus[3];
    normalised_cross_product(pi_plus,pi0_plus,n_plus);

    //normal to the plane spanned by pi- pi0
    double n_minus[3];
    normalised_cross_product(pi_minus,pi0_minus,n_minus);

    //get the angle
    double theta = acos(dot_product(n_plus,n_minus));
    
    //make theta go between 0 and 2 pi.
    double pi_minus_n_plus = dot_product(pi_minus,n_plus)/magnitude(pi_minus);    
    if(pi_minus_n_plus>0)
      theta=2*M_PI-theta;


    /*********** calculate C/D reco  (y1y2 in the paper) ***/

    //boost vectors back to the lab frame
    d_pi0_plus.Boost(-cm_px,-cm_py,-cm_pz,cm_e,cm_mass);
    d_pi_plus.Boost(-cm_px,-cm_py,-cm_pz,cm_e,cm_mass);
    d_pi0_minus.Boost(-cm_px,-cm_py,-cm_pz,cm_e,cm_mass);
    d_pi_minus.Boost(-cm_px,-cm_py,-cm_pz,cm_e,cm_mass);


    //construct effective tau 4 vectors (without neutrino)
    double e_tau = 120.0/2.0;
    double m_tau = 1.78;
    double p_mag_tau = sqrt(e_tau*e_tau - m_tau*m_tau);

    MC4Vector p_tau_plus = d_pi_plus + d_pi0_plus;
    MC4Vector p_tau_minus = d_pi_minus + d_pi0_minus;

    double norm_plus = p_mag_tau/p_tau_plus.Length();
    double norm_minus = p_mag_tau/p_tau_minus.Length();

    p_tau_plus.SetX0(e_tau);
    p_tau_plus.SetX1(norm_plus*p_tau_plus.GetX1());
    p_tau_plus.SetX2(norm_plus*p_tau_plus.GetX2());
    p_tau_plus.SetX3(norm_plus*p_tau_plus.GetX3());
    p_tau_plus.SetM(m_tau);

    p_tau_minus.SetX0(e_tau);
    p_tau_minus.SetX1(norm_minus*p_tau_minus.GetX1());
    p_tau_minus.SetX2(norm_minus*p_tau_minus.GetX2());
    p_tau_minus.SetX3(norm_minus*p_tau_minus.GetX3());
    p_tau_minus.SetM(m_tau);

    //boost to the (reconstructed) tau's frames
    d_pi0_plus.BoostP(p_tau_plus);
    d_pi_plus.BoostP(p_tau_plus);
    d_pi0_minus.BoostP(p_tau_minus);
    d_pi_minus.BoostP(p_tau_minus);

    //calculate y1 and y2
    double y1=(d_pi_plus.GetX0()-d_pi0_plus.GetX0())/(d_pi_plus.GetX0()+d_pi0_plus.GetX0());
    double y2=(d_pi_minus.GetX0()-d_pi0_minus.GetX0())/(d_pi_minus.GetX0()+d_pi0_minus.GetX0());
    
    //plot
    char *plotname = new char[30];
    if(y1*y2>0)
      sprintf(plotname,"acoplanarity-angle-C");
    else
      sprintf(plotname,"acoplanarity-angle-D");
    fillUserHisto(plotname,theta,1.0,0,2*M_PI);
    delete plotname;

    return 0;
};

