/* 
 *  \class TShapeAnalysis
 *
 *  $Date: 2012/02/09 10:08:10 $
 *  original author: Patrice Verrecchia 
 *   modified by Julie Malcles - CEA/Saclay
 */

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TShapeAnalysis.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TFParams.h>

#include <iostream>
#include <math.h>
#include <time.h>
#include <cassert>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>

//ClassImp(TShapeAnalysis)

using namespace std;

// Constructor...
TShapeAnalysis::TShapeAnalysis(double alpha0, double beta0, double width0, double chi20)
{
  init(alpha0, beta0, width0, chi20);
}

TShapeAnalysis::TShapeAnalysis(TTree *tAB, double alpha0, double beta0, double width0, double chi20)
{
  init(tAB, alpha0, beta0, width0, chi20);
}

// Destructor
TShapeAnalysis::~TShapeAnalysis()
{
}

void TShapeAnalysis::init(double alpha0, double beta0, double width0, double chi20)
{
  tABinit=NULL;
  nchsel=fNchsel;
  for(int cris=0;cris<fNchsel;cris++){

    index[cris]=-1;
    npass[cris]=0;
    npassok[cris]=0.;
    
    alpha_val[cris]=alpha0;
    beta_val[cris]=beta0;
    width_val[cris]=width0;
    chi2_val[cris]=chi20;
    flag_val[cris]=0;
    
    alpha_init[cris]=alpha0;
    beta_init[cris]=beta0;
    width_init[cris]=width0;
    chi2_init[cris]=chi20;
    flag_init[cris]=0;

    phi_init[cris]=0;
    eta_init[cris]=0;
    side_init[cris]=0;
    dcc_init[cris]=0;
    tower_init[cris]=0;
    ch_init[cris]=0;
    assignChannel(cris,cris);
    
  }
}

void TShapeAnalysis::init(TTree *tAB, double alpha0, double beta0, double width0, double chi20)
{
  init( alpha0, beta0, width0, chi20 );
  
  tABinit=tAB->CloneTree();

  // Declaration of leaf types
  Int_t           sidei;
  Int_t           iphii;
  Int_t           ietai;
  Int_t           dccIDi;
  Int_t           towerIDi;
  Int_t           channelIDi;
  Double_t        alphai;
  Double_t        betai;
  Double_t        widthi;
  Double_t        chi2i;
  Int_t           flagi;
  
  // List of branches
  TBranch        *b_iphi;   //!
  TBranch        *b_ieta;   //!
  TBranch        *b_side;   //!
  TBranch        *b_dccID;   //!
  TBranch        *b_towerID;   //!
  TBranch        *b_channelID;   //!
  TBranch        *b_alpha;   //!
  TBranch        *b_beta;   //!
  TBranch        *b_width;   //!
  TBranch        *b_chi2;   //!
  TBranch        *b_flag;   //!
  
  if(tABinit){

    tABinit->SetBranchAddress("iphi", &iphii, &b_iphi);
    tABinit->SetBranchAddress("ieta", &ietai, &b_ieta);
    tABinit->SetBranchAddress("side", &sidei, &b_side);
    tABinit->SetBranchAddress("dccID", &dccIDi, &b_dccID);
    tABinit->SetBranchAddress("towerID", &towerIDi, &b_towerID);
    tABinit->SetBranchAddress("channelID", &channelIDi, &b_channelID);
    tABinit->SetBranchAddress("alpha", &alphai, &b_alpha);
    tABinit->SetBranchAddress("beta", &betai, &b_beta);
    tABinit->SetBranchAddress("width", &widthi, &b_width);
    tABinit->SetBranchAddress("chi2", &chi2i, &b_chi2);
    tABinit->SetBranchAddress("flag", &flagi, &b_flag);
    
    nchsel=tABinit->GetEntries();
    assert(nchsel<=fNchsel);
    
    for(int cris=0;cris<nchsel;cris++){
      
      tABinit->GetEntry(cris);
      
      //      std::cout<< "Loop 1 "<< cris<<" "<<alphai<< std::endl;

      putalphaVal(cris,alphai);
      putchi2Val(cris,chi2i);
      putbetaVal(cris,betai);
      putwidthVal(cris,widthi);
      putflagVal(cris,flagi);
      
      putalphaInit(cris,alphai);
      putchi2Init(cris,chi2i);
      putbetaInit(cris,betai);
      putwidthInit(cris,widthi);
      putflagInit(cris,flagi);
      putetaInit(cris,ietai);
      putphiInit(cris,iphii);
 
    }
  }
}

void TShapeAnalysis::set_const(int ns, int ns1, int ns2, int ps, int nevtmax, double noise_val, double chi2_cut)
{
  nsamplecristal= ns;
  presample=ps;
  sampbmax= ns1;
  sampamax= ns2;
  nevt= nevtmax;
  noise= noise_val;
  chi2cut=chi2_cut;
}


void TShapeAnalysis::set_presample(int ps)
{
  presample=ps;
}
void TShapeAnalysis::set_nch(int nch){

  assert (nch<=fNchsel);
  if(tABinit) assert(nch==nchsel);
  nchsel=nch;

}
void TShapeAnalysis::assignChannel(int n, int ch)
{
    if(n >= nchsel)
         printf(" number of channels exceed maximum allowed\n");

    index[n]=ch;
}

void TShapeAnalysis::putDateStart(long int timecur)
{
    timestart=timecur;
}

void TShapeAnalysis::putDateStop(long int timecur)
{
    timestop=timecur;
}

void TShapeAnalysis::getDateStart()
{
    time_t t,timecur;
    timecur= time(&t);
    timestart= ((long int) timecur);
}

void TShapeAnalysis::getDateStop()
{
    time_t t,timecur;
    timecur= time(&t);
    timestop= ((long int) timecur);
}

void TShapeAnalysis::putAllVals(int ch, double* sampl, int ieta, int iphi, int dcc, int side, int tower, int chid)
{
  dcc_init[ch]=dcc;
  tower_init[ch]=side;
  ch_init[ch]=chid;
  side_init[ch]=side;
  eta_init[ch]=ieta;
  phi_init[ch]=iphi;
  putAllVals(ch, sampl, ieta, iphi);
  
}

void TShapeAnalysis::putAllVals(int ch, double* sampl, int ieta, int iphi)
{

  int i,k;
  int n=-1;
  for(i=0;i<nchsel;i++)
    if(index[i] == ch) n=i;
  
  if(n >= 0) {
    if(npass[n] < nevt) {
      
      for(k=0;k<nsamplecristal;k++) {
	rawsglu[n][npass[n]][k] = sampl[k];
      }
      
      npass[n]++;
    }
  } else {
    printf("no index found for ch=%d\n",ch);
  }
}

void TShapeAnalysis::computeShape(string namefile, TTree *tAB)
{

  double tm_atmax[200];
  double parout[3];

  double chi2_all=0.;

  double **dbi ;
  dbi = new double *[200];
  for(int k=0;k<200;k++) dbi[k] = new double[2];

  double **signalu ;
  signalu = new double *[200];
  for(int k=0 ;k<200;k++) signalu[k] = new double[10];        

  TFParams *pjf = new TFParams() ;

  for(int i=0;i<nchsel;i++) {

    if(index[i] >= 0) {
      
      if(npass[i] <= 10) {

	putalphaVal(i, alpha_init[i]);
	putbetaVal(i, beta_init[i]);
	putwidthVal(i,width_init[i]);
	putchi2Val(i,chi2_init[i]);
	putflagVal(i,0);
	
      } else {
	
	pjf->set_const(nsamplecristal, sampbmax, sampamax, alpha_init[i], beta_init[i], npass[i]);
	
	for(int pass=0;pass<npass[i];pass++){
	  
	  double ped=0;
	  for(int k=0;k<presample;k++) {
	    ped+= rawsglu[i][pass][k];
	  }
	  ped/=double(presample);
	  
	  for(int k=0;k<nsamplecristal;k++) {
	    signalu[pass][k]= rawsglu[i][pass][k]-ped;
	  }
	}
	
	int debug=0;
	chi2_all= pjf->fitpj(signalu,&parout[0],dbi,noise, debug);
	
	if(parout[0]>=0.0 && parout[1]>=0.0 && chi2_all<=chi2cut && chi2_all>0.0){
	  
	  putalphaVal(i,parout[0]);
	  putbetaVal(i,parout[1]);
	  putchi2Val(i,chi2_all);
	  putflagVal(i,1);
	  
	}else{
	  
	  putalphaVal(i,alpha_init[i]);
	  putbetaVal(i,beta_init[i]);
	  putwidthVal(i,width_init[i]);
	  putchi2Val(i,chi2_init[i]);
	  putflagVal(i,0);
	  
	} 
	
	for(int kj=0;kj<npass[i];kj++) { // last event wrong here
	  tm_atmax[kj]= dbi[kj][1];
	}
	computetmaxVal(i,&tm_atmax[0]);
	
      }
    }
  }
  
  if(tAB) tABinit=tAB->CloneTree();
  
  // Declaration of leaf types
  Int_t           sidei;
  Int_t           iphii;
  Int_t           ietai;
  Int_t           dccIDi;
  Int_t           towerIDi;
  Int_t           channelIDi;
  Double_t        alphai;
  Double_t        betai;
  Double_t        widthi;
  Double_t        chi2i;
  Int_t           flagi;
  
  // List of branches
  TBranch        *b_iphi;   //!
  TBranch        *b_ieta;   //!
  TBranch        *b_side;   //!
  TBranch        *b_dccID;   //!
  TBranch        *b_towerID;   //!
  TBranch        *b_channelID;   //!
  TBranch        *b_alpha;   //!
  TBranch        *b_beta;   //!
  TBranch        *b_width;   //!
  TBranch        *b_chi2;   //!
  TBranch        *b_flag;   //!
  

  if(tABinit){
    tABinit->SetBranchAddress("iphi", &iphii, &b_iphi);
    tABinit->SetBranchAddress("ieta", &ietai, &b_ieta);
    tABinit->SetBranchAddress("side", &sidei, &b_side);
    tABinit->SetBranchAddress("dccID", &dccIDi, &b_dccID);
    tABinit->SetBranchAddress("towerID", &towerIDi, &b_towerID);
    tABinit->SetBranchAddress("channelID", &channelIDi, &b_channelID);
    tABinit->SetBranchAddress("alpha", &alphai, &b_alpha);
    tABinit->SetBranchAddress("beta", &betai, &b_beta);
    tABinit->SetBranchAddress("width", &widthi, &b_width);
    tABinit->SetBranchAddress("chi2", &chi2i, &b_chi2);
    tABinit->SetBranchAddress("flag", &flagi, &b_flag);
  }

  TFile *fABout = new TFile(namefile.c_str(), "RECREATE");
  tABout=new TTree("ABCol0","ABCol0");
  
  // Declaration of leaf types
  Int_t           side;
  Int_t           iphi;
  Int_t           ieta;
  Int_t           dccID;
  Int_t           towerID;
  Int_t           channelID;
  Double_t        alpha;
  Double_t        beta;
  Double_t        width;
  Double_t        chi2;
  Int_t           flag;
  
  tABout->Branch( "iphi",      &iphi,       "iphi/I"      );
  tABout->Branch( "ieta",      &ieta,       "ieta/I"      );
  tABout->Branch( "side",      &side,       "side/I"      );
  tABout->Branch( "dccID",     &dccID,      "dccID/I"     );
  tABout->Branch( "towerID",   &towerID,    "towerID/I"   );
  tABout->Branch( "channelID", &channelID,  "channelID/I" );
  tABout->Branch( "alpha",     &alpha,     "alpha/D"    );
  tABout->Branch( "beta",      &beta,      "beta/D"     );
  tABout->Branch( "width",     &width,     "width/D"    );
  tABout->Branch( "chi2",      &chi2,      "chi2/D"     );
  tABout->Branch( "flag",      &flag,      "flag/I"     );
  
  tABout->SetBranchAddress( "ieta",      &ieta       );  
  tABout->SetBranchAddress( "iphi",      &iphi       ); 
  tABout->SetBranchAddress( "side",      &side       );
  tABout->SetBranchAddress( "dccID",     &dccID      );
  tABout->SetBranchAddress( "towerID",   &towerID    );
  tABout->SetBranchAddress( "channelID", &channelID  );
  tABout->SetBranchAddress( "alpha",     &alpha     );
  tABout->SetBranchAddress( "beta",      &beta      );
  tABout->SetBranchAddress( "width",     &width     );
  tABout->SetBranchAddress( "chi2",      &chi2      );
  tABout->SetBranchAddress( "flag",      &flag      );
  
  for(int i=0;i<nchsel;i++) {
    
    if(tABinit){
      
      tABinit->GetEntry(i);
      iphi=iphii;
      ieta=ietai;
      side=sidei;
      dccID=dccIDi;
      towerID=towerIDi;
      channelID=channelIDi;
   
    }else{
      
      iphi=phi_init[i];
      ieta=eta_init[i];
      side=side_init[i];
      dccID=dcc_init[i];
      towerID=tower_init[i];
      channelID=ch_init[i];
      
    }
    
    alpha=alpha_val[i];
    beta=beta_val[i];
    width=width_val[i];
    chi2=chi2_val[i];
    flag=flag_val[i];
    
    tABout->Fill();
  }
  
  
  tABout->Write();
  fABout->Close();

  delete pjf;
}

void TShapeAnalysis::computetmaxVal(int i, double* tm_val)
{
  double tm_mean=0; //double tm_sig=0;

        double tm=0.; double sigtm=0.;
	for(int k=0;k<npass[i]-1;k++) {
		if(1. < tm_val[k] && tm_val[k] < 10.) {
		   npassok[i]++;
                   tm+= tm_val[k];
                   sigtm+= tm_val[k]*tm_val[k];
		}
	}
	if(npassok[i] <= 0) {
	      tm_mean=0.; //tm_sig=0.;
	} else {
	      for(int k=0;k<npass[i]-1;k++) {
		    if(1. < tm_val[k] && tm_val[k] < 10.) {
                         double ss= (sigtm/npassok[i]-tm/npassok[i]*tm/npassok[i]);
                         if(ss < 0.) ss=0.;
                         //tm_sig=sqrt(ss);
                         tm_mean= tm/npassok[i];
		    }
	      }
	}
        //printf("npassok[%d]=%f tm_mean=%f tm_sig=%f\n",i,npassok[i],tm_mean,tm_sig);
        putwidthVal(i,tm_mean);
	
}

void TShapeAnalysis::putalphaVal(int n, double val)
{
    alpha_val[n]= val;
}

void TShapeAnalysis::putchi2Val(int n, double val)
{
    chi2_val[n]= val;
}
void TShapeAnalysis::putbetaVal(int n, double val)
{
    beta_val[n]= val;
}

void TShapeAnalysis::putwidthVal(int n, double val)
{
    width_val[n]= val;
}

void TShapeAnalysis::putflagVal(int n, int val)
{
    flag_val[n]= val;
}

void TShapeAnalysis::putalphaInit(int n, double val)
{
    alpha_init[n]= val;
}

void TShapeAnalysis::putchi2Init(int n, double val)
{
    chi2_init[n]= val;
}
void TShapeAnalysis::putbetaInit(int n, double val)
{
    beta_init[n]= val;
}

void TShapeAnalysis::putwidthInit(int n, double val)
{
    width_init[n]= val;
}

void TShapeAnalysis::putetaInit(int n, int val )
{
    eta_init[n]= val;
}

void TShapeAnalysis::putphiInit(int n, int val)
{
    phi_init[n]= val;
}

void TShapeAnalysis::putflagInit(int n, int val)
{
    flag_init[n]= val;
}
std::vector<double> TShapeAnalysis::getVals(int n)
{

  std::vector<double> v;
  
  v.push_back(alpha_val[n]);
  v.push_back(beta_val[n]);
  v.push_back(width_val[n]);
  v.push_back(chi2_val[n]);
  v.push_back(flag_val[n]);
  
  return v;
}
std::vector<double> TShapeAnalysis::getInitVals(int n)
{

  std::vector<double> v;
  
  v.push_back(alpha_init[n]);
  v.push_back(beta_init[n]);
  v.push_back(width_init[n]);
  v.push_back(chi2_init[n]);
  v.push_back(flag_init[n]);
  
  return v;
}

void TShapeAnalysis::printshapeData(int gRunNumber)
{
     FILE *fd;
     int nev;
     sprintf(filename,"runABW%d.pedestal",gRunNumber); 
     fd = fopen(filename, "w");                                
     if(fd == NULL) printf("Error while opening file : %s\n",filename);

     for(int i=0; i<nchsel;i++) {
        if(index[i] >= 0) {
          nev= (int) npassok[i];
          double trise= alpha_val[i]*beta_val[i];
          fprintf( fd, "%d %d 1 %ld %ld %f %f %f %f\n",
                index[i],nev,timestart,timestop,alpha_val[i],beta_val[i],trise,width_val[i]);
	}
     }
     int iret=fclose(fd);
     printf(" Closing file : %d\n",iret);

}

