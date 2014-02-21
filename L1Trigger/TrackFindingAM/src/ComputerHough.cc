#include <stdint.h>
#include <stdio.h> 
#include "../interface/ComputerHough.h"
#include <math.h>
ComputerHough::ComputerHough(HoughCut* cuts) :theCuts_(cuts)
{
  theNStub_=0;
  theX_=NULL;
  theY_=NULL;
  theZ_=NULL;
  theLayer_=NULL;

  createHoughCPU(&ph_,768,3072,768);
  for (int i=0;i<96;i++)
    createHoughCPU(&phcand_[i]);
  for (int i=0;i<64;i++)
    createHoughCPU(&phrcand_[i]);


}

ComputerHough::~ComputerHough(){
  deleteHoughCPU(&ph_);
  for (int i=0;i<96;i++)
    deleteHoughCPU(&phcand_[i]);
  for (int i=0;i<64;i++)
    deleteHoughCPU(&phrcand_[i]);
}

void ComputerHough::DefaultCuts()
{
  theCuts_->RhoMin=-0.0031;
  theCuts_->RhoMax=0.0031;
  theCuts_->NRho=6;
  theCuts_->NTheta=48;
  theCuts_->NStubLow=4;
  theCuts_->NLayerRow=4;
  theCuts_->NStubLowCandidate=5;
  theCuts_->NBins3GeV=56;
  theCuts_->NBins5GeV=128; 
  theCuts_->NBins15GeV=128;
  theCuts_->NBins30GeV=128;
  theCuts_->NBins100GeV=128;
  theCuts_->NDelBarrel=1.5;
  theCuts_->NDelInter=2.1;
  theCuts_->NDelEndcap=2.7;
  theCuts_->NStubHigh=5;
  theCuts_->NLayerHigh=5;
  theCuts_->NStubHighCandidate=5;
}
void ComputerHough::Compute(uint32_t isel,uint32_t nstub,float* x,float* y,float* z,uint32_t* layer)
{
  theNStub_=nstub;
  theX_=x;
  theY_=y;
  theZ_=z;  
  theLayer_=layer;
  theCandidateVector_.clear();
  // Initialisation depending on sector 
  bool barrel=isel>=16 && isel<40;
  bool inter=(isel>=8 &&isel<16)||(isel>=40&&isel<48);
  bool endcap=(isel<8)||(isel>=48);
  float thmin=-PI/2,thmax=PI/2;
  float rhmin=theCuts_->RhoMin,rhmax=theCuts_->RhoMax;
  //printf("On appelle le GPU %d \n",theNStub_);
  int ntheta=160;
  int nrho=theCuts_->NRho;//8//12//192;
  //initialiseHough(&ph,gpu_nstub,ntheta,nrho,-PI/2,PI/2,-0.06,0.06);
  if (barrel || inter || endcap)
    {
      ntheta=theCuts_->NTheta;//64;
      if (isel%4==0) thmin=1.32;
      if (isel%4==1) thmin=-1.04;
      if (isel%4==2) thmin=-0.24;
      if (isel%4==3) thmin=0.51;
      thmax=thmin+1.25;
    }

  if (theNStub_>400)
    {
      ntheta*=2;
      nrho*=2;
    }
  
  initialiseHoughCPU(&ph_,theNStub_,ntheta,nrho,thmin,thmax,rhmin,rhmax);
  // Rough process
  fillConformalHoughCPU(&ph_,theX_,theY_,theZ_);
  fillLayerHoughCPU(&ph_,theLayer_);
		  //clearHough(&ph);
  processHoughCPU(&ph_,theCuts_->NStubLow,theCuts_->NLayerRow,0,endcap);
  //printf("SECTOR %d gives %d candidates Max val %d STubs %d\n",isel,ph_.h_cand[0],ph_.max_val,ph_.nstub);
  // Precise HT filling
  uint32_t nc=(int)ph_.h_cand[0];
  if (nc>96) nc=96;
  for (unsigned int ic=0;ic<nc;ic++)
    {
      clearHoughCPU(&phcand_[ic]);
    }

  // Loop on candidate
  for (unsigned int ic=0;ic<nc;ic++)
    {
      phcand_[ic].h_reg[20]=0;
      int pattern=ph_.h_cand[ic+1]; // vcand[ic]
      int ith=pattern&0X3FF;
      int ir=(pattern>>10)&0x3FF;
      //ith=(vcand[ic])&0x3FF;
      //ir=(vcand[ic]>>10)&0x3FF;
      int ns=(pattern>>20)&0x3FF;
      if (ns<(int)theCuts_->NStubLowCandidate) continue;//if (ns<3) continue;
      double PT=1./2./fabs(GET_R_VALUE(ph_,ir))*0.3*3.8/100.;
      if (PT<1.5) continue;
      //printf("%f \n",fabs(GET_R_VALUE(ph,ir)));
      uint32_t nbinf=64;
      // <5,5-10,10-30,>30
      if (PT<3) nbinf=theCuts_->NBins3GeV;
      if (PT>=3 && PT<5) nbinf=theCuts_->NBins5GeV; // 128
      if (PT>=5  && PT<15) nbinf=theCuts_->NBins15GeV;//192
      if (PT>=15 && PT<=30) nbinf=theCuts_->NBins30GeV;//256
      if (PT>=30 ) nbinf=theCuts_->NBins100GeV;//256



      uint32_t nbinr=nbinf;

      if (ns>20 ) nbinf=2*nbinf;
		  
      float ndel=theCuts_->NDelBarrel;
      if (inter) 
	ndel=theCuts_->NDelInter;
      else
	if (endcap)
	  ndel=theCuts_->NDelEndcap;
      float tmi=GET_THETA_VALUE(ph_,ith)-ndel*ph_.thetabin;
      
      float tma=GET_THETA_VALUE(ph_,ith)+ndel*ph_.thetabin;
      float rmi=GET_R_VALUE(ph_,ir)-ndel*ph_.rbin;
      float rma=GET_R_VALUE(ph_,ir)+ndel*ph_.rbin;
      
      initialiseHoughCPU(&phcand_[ic],theNStub_,nbinf,nbinr,tmi,tma,rmi,rma);	    

      copyPositionHoughCPU(&ph_,pattern,&phcand_[ic],0,false,endcap);
    }
		  

		
  //Precise HT processing
		 
  for (unsigned int ic=0;ic<nc;ic++)
    {
      if (phcand_[ic].h_reg[20]>0)
	{
	  phcand_[ic].nstub=int( phcand_[ic].h_reg[20]);
	  processHoughCPU(&phcand_[ic],theCuts_->NStubHigh,theCuts_->NLayerHigh,0,endcap);
	  
	}
    }

  // Finael analysis of High precision candidate

  for (unsigned int ic=0;ic<nc;ic++)
    {
      if (phcand_[ic].h_reg[20]>0)
	{
	  uint32_t nch=(int)phcand_[ic].h_cand[0];
	  if (nch>64) nch=64;
	  for (unsigned int ici=0;ici<nch;ici++)
	    {
	      int patterni=phcand_[ic].h_cand[ici+1]; 
	      int ithi=patterni&0X3FF;
	      int iri=(patterni>>10)&0x3FF;
	      
	      if (((patterni>>20)&0x3FF)<theCuts_->NStubHighCandidate) continue;

	      mctrack_t t;
	      // RZ  & R Phi regression
	      initialiseHoughCPU(&phrcand_[ici],theNStub_,32,32,-PI/2,PI/2,-150.,150.);
	      copyPositionHoughCPU(&phcand_[ic],patterni,&phrcand_[ici],1,true,endcap);
	      phrcand_[ici].nstub=int( phrcand_[ici].h_reg[20]);
	      if (phrcand_[ici].h_reg[60+6]<1.7) continue;
	      if ( phrcand_[ici].h_reg[20]<=0) continue;
			      
	      if ( phrcand_[ici].h_reg[70+9]<1.5) continue; //at least 2 Z points
	      t.z0=-phrcand_[ici].h_reg[70+1]/phrcand_[ici].h_reg[70+0];
	      t.eta=phrcand_[ici].h_reg[70+8];
	      if ( fabs(t.z0)>30.) continue;
			  
	      
	      float theta=GET_THETA_VALUE(phcand_[ic],ithi);
	      float r=GET_R_VALUE(phcand_[ic],iri);

	      //double a=-1./tan(theta);
	      //double b=r/sin(theta);
			      
		
			  //
	      //double R=1./2./fabs(r);
	      //double xi=-a/2./b;
	      //double yi=1./2./b;
	      //double g_pt=0.3*3.8*R/100.;
			  //g_phi=atan(a);
	      double g_phi=theta-PI/2.;
	      if (g_phi<0) g_phi+=2*PI;
	      ComputerHough::Convert(theta,r,&t);
	      t.nhits=(patterni>>20)&0x3FF;
	      t.theta=theta;
	      t.r=r;

	      t.pt=phrcand_[ici].h_reg[60+6];
	      t.phi=phrcand_[ici].h_reg[60+2];
	      t.nhits=(patterni>>20)&0x3FF;
	      t.layers.clear();
	      for (int ist=0;ist<phrcand_[ici].nstub;ist++)
		t.layers.push_back(phrcand_[ici].d_layer[ist]);	      
	      theCandidateVector_.push_back(t);

			      
	    }
	}
    }
		 









		  
		
		  
  //  printf("Fin du CPU %ld \n",	theCandidateVector_.size() );



}
void ComputerHough::ComputeOneShot(uint32_t isel,uint32_t nstub,float* x,float* y,float* z,uint32_t* layer)
{
  theNStub_=nstub;
  theX_=x;
  theY_=y;
  theZ_=z;  
  theLayer_=layer;
  theCandidateVector_.clear();
  // Initialisation depending on sector 
  bool barrel=isel>=16 && isel<40;
  bool inter=(isel>=8 &&isel<16)||(isel>=40&&isel<48);
  bool endcap=(isel<8)||(isel>=48);
  float thmin=-PI/2,thmax=PI/2;
  float rhmin=theCuts_->RhoMin,rhmax=theCuts_->RhoMax;
  //printf("On appelle le GPU %d \n",theNStub_);
  int ntheta=160;
  int nrho=theCuts_->NRho;//8//12//192;
  //initialiseHough(&ph,gpu_nstub,ntheta,nrho,-PI/2,PI/2,-0.06,0.06);
  if (barrel || inter || endcap)
    {
      ntheta=theCuts_->NTheta;//64;
      if (isel%4==0) thmin=1.32;
      if (isel%4==1) thmin=-1.04;
      if (isel%4==2) thmin=-0.24;
      if (isel%4==3) thmin=0.51;
      thmax=thmin+1.25;
    }

  if (theNStub_>400)
    {
      ntheta*=2;
      nrho*=2;
    }
 
   ntheta=960;
  nrho=156;
  if (inter)
    {
      ntheta=1056;
      nrho=88;
    }
  if (endcap)
    {
      ntheta=1056;
      nrho=64;
    }
  theCuts_->NLayerRow=5;

  initialiseHoughCPU(&ph_,theNStub_,ntheta,nrho,thmin,thmax,rhmin,rhmax);
  // Rough process
  fillConformalHoughCPU(&ph_,theX_,theY_,theZ_);
  fillLayerHoughCPU(&ph_,theLayer_);
		  //clearHough(&ph);
  processHoughCPU(&ph_,theCuts_->NStubLow,theCuts_->NLayerRow,0,endcap);
  //printf("SECTOR %d gives %d candidates Max val %d STubs %d\n",isel,ph_.h_cand[0],ph_.max_val,ph_.nstub);
  // Precise HT filling
  uint32_t nc=(int)ph_.h_cand[0];
  if (nc>512) nc=512;
  clearHoughCPU(&phcand_[0]);
  
  // Loop on candidate
  for (unsigned int ic=0;ic<nc;ic++)
    {
      phcand_[0].h_reg[20]=0;
      int pattern=ph_.h_cand[ic+1]; // vcand[ic]
      int ith=pattern&0X3FF;
      int ir=(pattern>>10)&0x3FF;
      //ith=(vcand[ic])&0x3FF;
      //ir=(vcand[ic]>>10)&0x3FF;
      int ns=(pattern>>20)&0x3FF;
      if (ns<(int)theCuts_->NStubLowCandidate) continue;//if (ns<3) continue;
      double PT=1./2./fabs(GET_R_VALUE(ph_,ir))*0.3*3.8/100.;
      if (PT<1.5) continue;
      //printf("%f \n",fabs(GET_R_VALUE(ph,ir)));

      mctrack_t t;
      // RZ  & R Phi regression
      initialiseHoughCPU(&phcand_[0],theNStub_,32,32,-PI/2,PI/2,-150.,150.);
      copyPositionHoughCPU(&ph_,pattern,&phcand_[0],1,true,endcap);
      phcand_[0].nstub=int( phcand_[0].h_reg[20]);
      if (phcand_[0].h_reg[60+6]<1.7) continue;
      if ( phcand_[0].h_reg[20]<=0) continue;
      
      if ( phcand_[0].h_reg[70+9]<1.5) continue; //at least 2 Z points
      t.z0=-phcand_[0].h_reg[70+1]/phcand_[0].h_reg[70+0];
      t.eta=phcand_[0].h_reg[70+8];
      if ( fabs(t.z0)>30.) continue;
      
	      
      float theta=GET_THETA_VALUE(ph_,ith);
      float r=GET_R_VALUE(ph_,ir);
      
      //double a=-1./tan(theta);
      //double b=r/sin(theta);
      
      
      //
      //double R=1./2./fabs(r);
      //double xi=-a/2./b;
      //double yi=1./2./b;
      //double g_pt=0.3*3.8*R/100.;
      //g_phi=atan(a);
      double g_phi=theta-PI/2.;
      if (g_phi<0) g_phi+=2*PI;
      ComputerHough::Convert(theta,r,&t);
      t.nhits=(pattern>>20)&0x3FF;
      t.theta=theta;
      t.r=r;
      
      t.pt=phcand_[0].h_reg[60+6];
      t.phi=phcand_[0].h_reg[60+2];
      t.nhits=(pattern>>20)&0x3FF;
      t.layers.clear();
      for (int ist=0;ist<phcand_[0].nstub;ist++)
	t.layers.push_back(phcand_[0].d_layer[ist]);
      theCandidateVector_.push_back(t);
      
      
      
    }
}
 
void ComputerHough::Convert(double theta,double r,mctrack_t *m)
{
  double a=-1./tan(theta);
  double b=r/sin(theta);
		
		
  //
  double R=1./2./fabs(r);
  double xi=-a/2./b;
  double yi=1./2./b;
  //printf(" From r=%f theta=%f a=%f b=%f  R= %f  => Pt=%f GeV/c  Phi0=%f \n",r,theta,a,b,R,0.3*3.8*R/100.,atan(a));
  m->pt=0.3*3.8*R/100.;
  m->phi=atan(a);
  m->phi=theta-PI/2.;
  if (m->phi<0) m->phi+=2*PI;
  m->rho0=sqrt(fabs(R*R-xi*xi-yi*yi));


 
}
