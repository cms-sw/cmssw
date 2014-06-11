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
  cout << "("<<v[0]<<","<<v[1]<<","<<v[2]<<")"<<endl;
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

double magnitude(double *v){
  return sqrt(dot_product(v,v));
}


/** Main macro. No paramters should be given. 
    This assumes that the event are H/Z -> tau+ tau-
    and then tau -> pi nu. **/
int PiPiUserTreeAnalysis(HEPParticle *mother,
			 HEPParticleList *stableDaughters, 
			 int nparams, double *params)
{
    assert(mother!=0);
    assert(stableDaughters!=0);

    HEPParticleListIterator daughters(*stableDaughters);

    //arrays to hold 3 vectors of tau and pi pairs.
    double tau_plus[3]={0};
    double tau_minus[3]={0};
    double pi_plus[3]={0};
    double pi_minus[3]={0};

    //loop over the daughters, boost them into the Higgs (or Z etc.) frame,
    //then fill the 3 vector arrays.
    for (HEPParticle *part=daughters.first(); part!=0; part=daughters.next()){
      MC4Vector d4(part->GetE(),part->GetPx(),
		   part->GetPy(),part->GetPz(),part->GetM());
      d4.Boost(mother->GetPx(),mother->GetPy(),mother->GetPz(),
	       mother->GetE(),mother->GetM());

      //tau's 3 vectors are calculated from their daughters
      //(daughters are pi+/- or neutrino)
      if(part->GetPDGId()==211||part->GetPDGId()==-16){
	tau_plus[0]+=d4.GetX1();
	tau_plus[1]+=d4.GetX2();
	tau_plus[2]+=d4.GetX3();
      }
      if(part->GetPDGId()==-211||part->GetPDGId()==16){
	tau_minus[0]+=d4.GetX1();
	tau_minus[1]+=d4.GetX2();
	tau_minus[2]+=d4.GetX3();
      }
      
      //fill pi+ or pi- array
      if(part->GetPDGId()==-211){
	pi_minus[0]=d4.GetX1();
	pi_minus[1]=d4.GetX2();
	pi_minus[2]=d4.GetX3();
      }
      if(part->GetPDGId()==211){
	pi_plus[0]=d4.GetX1();
	pi_plus[1]=d4.GetX2();
	pi_plus[2]=d4.GetX3();
      }
    }

    /*** Acollinarity **/
    //calculate the angle between the pi+ and pi- (in Higgs rest frame)
    double delta = acos(dot_product(pi_plus,pi_minus)/(magnitude(pi_plus)*magnitude(pi_minus)));
    char *plotname = new char[30];
	sprintf(plotname,"delta");
    fillUserHisto(plotname,delta,1.0,0,M_PI);
	sprintf(plotname,"delta2");
    fillUserHisto(plotname,delta,1.0,3,M_PI);


    /*** Acoplanarity (theta) **/
    //calculate the angle between  pi+ and tau+
    double projection_plus = dot_product(pi_plus,tau_plus)/(magnitude(tau_plus)*magnitude(tau_plus));

    //calculate the 3-vector for the pi+ transverse to the tau+
    double pi_plus_pt[3];
    pi_plus_pt[0]  = pi_plus[0]-(projection_plus*tau_plus[0]);
    pi_plus_pt[1]  = pi_plus[1]-(projection_plus*tau_plus[1]);
    pi_plus_pt[2]  = pi_plus[2]-(projection_plus*tau_plus[2]);
    
    //calculate the angle between  pi- and tau-
    double projection_minus = dot_product(pi_minus,tau_minus)/(magnitude(tau_minus)*magnitude(tau_minus));

    //calculate the 3-vector for the pi- transverse to the tau-
    double pi_minus_pt[3];
    pi_minus_pt[0]  = pi_minus[0]-(projection_minus*tau_minus[0]);
    pi_minus_pt[1]  = pi_minus[1]-(projection_minus*tau_minus[1]);
    pi_minus_pt[2]  = pi_minus[2]-(projection_minus*tau_minus[2]);

    //calculate the angle between the pi+ transverse and pi- transverse
    //but this only gives the angle from 0 to pi.
    double theta = acos(dot_product(pi_plus_pt,pi_minus_pt)/(magnitude(pi_plus_pt)*magnitude(pi_minus_pt)));
    
    //to get the angle between 0 and 2 pi, use the normal to the pi+ pi- transverse pair
    //if the normal is in the direction of the tau-, set theta = 2pi - theta.
    double n3[3];
    normalised_cross_product(pi_plus_pt,pi_minus_pt,n3);    
    double theta_sign = dot_product(n3,tau_minus)/magnitude(tau_minus);    
    if(theta_sign>0)
      theta=2*M_PI-theta;
    
    //finally, make the plot
	sprintf(plotname,"theta");
    fillUserHisto(plotname,theta,1.0,0,2*M_PI);
    delete plotname;

    return 0;
};

