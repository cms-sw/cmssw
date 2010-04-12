/* 
 *  \class TCalibData
 *
 *  $Date: 2010/04/12 08:38:39 $
 *  \author: Patrice Verrecchia - CEA/Saclay
 */


#include <iostream>
#include <sstream>
#include <cassert>
#include <TFile.h>
#include <TH2D.h>
#include <TTree.h>
#include <TBranch.h>
#include <TMath.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/MEEBGeom.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/MEEEGeom.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TCalibData.h>


//ClassImp(TCalibData)

// Default Constructor...
TCalibData::TCalibData( )
{ 
  init(610,"/nfshome0/ecallaser/calibdata/140110");
}

// Constructor...
TCalibData::TCalibData(int fed, string path )
{ 
  init(fed, path);

}

// Constructor...
TCalibData::TCalibData(int fed, string path , string alphapath )
{ 
  init(fed, path);

  stringstream ABFile;
  ABFile<<alphapath<<"/AB"<<_fed<<"_LASER.root";
  _ABfile=ABFile.str();
  _ABFileSet=setABFile(_ABfile);

}
// Destructor
TCalibData::~TCalibData()
{ 
  if(!_isBarrel){
    delete EENum;
    EENumFile->Close();
  }
  if(_ABFileSet && ABFile ){
    ABFile->Close();
  }


}

// Initialize...
void TCalibData::init(int fed, string path )
{ 
  _debug=false;
  _newLinType=true;
  _isBarrel=false;

  _fed=fed;
  _path=path;

  if(_fed<600) _fed+=600;

  stringstream sprAPDfile, sprPNfile, ABfile, linPNfile, EENumfile;
  stringstream nameEENum;
  
  if(_fed!=608 && _fed!=653) _nmemEE=2;
  else _nmemEE=NMEMEE;
  
  sprAPDfile<<path<<"/shapes/"<<_fed<<"/SPR_tau_440.data";
  sprPNfile<<path<<"/shapes/SPR_tau_PN_EE.data";
  EENumfile<<path<<"/numbering/EE"<<_fed<<".root";

  nameEENum<<"FED"<<_fed;
  

  if(_debug) cout<< "TCalibData::init EENUM NAMES:"<<EENumfile.str()<<" "<<nameEENum.str()<< endl;

  if(_fed>609 && _fed <646){
    _isBarrel=true;
    _sprPNfile=_sprAPDfile;
  }else{ 
   
    EENumFile = new TFile(EENumfile.str().c_str());
    EENum = (TH2D*) EENumFile->Get(nameEENum.str().c_str());
    assert(EENum);
  }

  ABfile<<path<<"/alphabeta/AB"<<_fed<<"_LASER.root";
  if(_isBarrel){
    if(!_newLinType) linPNfile<<path<<"/pnlincor/"<<_fed<<"/corlin_pn.data";
    else linPNfile<<path<<"/pnlincor2/"<<_fed<<"/corlin_pn.p5.data";
  }else{
    if(!_newLinType) linPNfile<<path<<"/pnlincor/corlin_pn.EE.data";
    else linPNfile<<path<<"/pnlincor2/corlin_pn.EE.p5.data";
  }
  _sprAPDfile=sprAPDfile.str();
  _sprPNfile=sprPNfile.str();
  _ABfile=ABfile.str();
  _linPNfile=linPNfile.str();

  if(_debug){
    cout<< "TCalibData::init Files: "<<endl;
    cout<< "                    AB: "<<_ABfile<<endl;
    cout<< "                    PN: "<<_sprPNfile<<endl;
    cout<< "                   APD: "<<_sprAPDfile<<endl;
    cout<< "                 PNLin: "<<_linPNfile<<endl;
  }

  if(_fed>609 && _fed <646){
    _isBarrel=true;
    _sprPNfile=_sprAPDfile;
  }

  
  _tauRead=false;
  _linPNRead=false;
  _matacqRead=false;
  _ABRead[0]=false;
  _ABRead[3]=false;

  _ABFileSet=setABFile(_ABfile);

  if(_debug) cout<< "TCalibData::init setABFile result: "<<_ABFileSet<< endl;
}

bool TCalibData::setABFile(string alphafile){
  
  if(_debug) cout<< "TCalibData::setABFile "<<alphafile<< endl;
  bool ok=false; 
  _ABfile=alphafile;  
  FILE *test; 
  test = fopen(_ABfile.c_str(),"r");
  if (test){ 
    ok=true;
    fclose(test);
  }
  if(ok){
    if(_debug) cout<< "TCalibData::setABFile File Exists"<< endl;
    ABFile= new TFile(_ABfile.c_str());
    if(ABFile) ok=true;
  }
  

  return ok;

}
bool TCalibData::readTaus(){

  bool ok=false;

  int nCrys=NCRYS;
  
  int doesFileExist=0;
  int doesFilePNExist=0;
  FILE *sprf; 
  sprf = fopen(_sprAPDfile.c_str(),"r");
  if (sprf) doesFileExist=1; 
  else cout<< "SPR file not found"<< endl;

  int nch=0;
  int nchapd=0;
  
  if (doesFileExist==1){
    
    int nread=0;

    if(_isBarrel){
      while(nread!=EOF)
	{
	  int ich=0;
	  float tau1=50., tau2=20.0;
	  nread=fscanf(sprf,"%d %f %f",&ich,&tau1,&tau2);
	  if(nread!=3)break;
	  if(ich>=0 && ich<nCrys) {_tausAPD[0][ich]=tau1; _tausAPD[1][ich]=tau2;}
	  if(ich>=nCrys && ich<nCrys+NPN) {
	    _tausPNEB[0][ich-nCrys]=tau1;
	    _tausPNEB[1][ich-nCrys]=tau2;
	    _qmaxPNEB[ich-nCrys]=getqmax(tau1,tau2);


	  }
	  nch++;  
	}
      
      nchapd=nch-NPN;
      nch=nCrys;
      assert(nchapd==nch);
      fclose(sprf);

    }else{

      while(nread!=EOF)
	{
	  int ich=0;
	  float tau1=50., tau2=20.0;
	  nread=fscanf(sprf,"%d %f %f",&ich,&tau1,&tau2);
	  if(nread!=3)break;
	  if(ich>=0 && ich<nCrys) {_tausAPD[0][ich]=tau1; _tausAPD[1][ich]=tau2;}
	  nch++;  
	}
      
      fclose(sprf);

      // first read apd:
      int nread=0;

      int pnOffset[NMEMEE];
      for(unsigned int imm=0;imm<_nmemEE;imm++){
	pnOffset[imm]=0;
      }
      
      vector<int> mems= ME::memFromDcc(_fed);
      assert(mems.size()==_nmemEE);
      
      if(_debug) cout<<"TCalibData::readTaus MEMS CHECK: "<<mems.size()<< endl ;

      for(unsigned int jmem=0;jmem<mems.size();jmem++){
	pnOffset[jmem]=PNOffset(mems[jmem]);
	if(_debug)cout<<"TCalibData::readTaus MEM "<<jmem<<" = "<< mems[jmem] << "  "<<pnOffset[jmem]<<endl ;
	
      }
      
      // then read pn:
      FILE *sprpnf; 
      sprpnf = fopen(_sprPNfile.c_str(),"r");
      if (sprpnf) doesFilePNExist=1; 
      else cout<< "SPR PN file not found"<< endl;
      
      if (doesFilePNExist==1){
	
	  nread=0;
	  if(_debug) cout<< "TCalibData::readTaus Before Loop "<< nread << endl;
	  while(nread!=EOF)
	    {
	      int ich=0;
	      float tau1=50., tau2=20.0;
	      nread=fscanf(sprpnf,"%d %f %f",&ich,&tau1,&tau2);
	      if(nread!=3) break;
	      for(unsigned int jmem=0;jmem<mems.size();jmem++){
		int nOffset=nCrys+pnOffset[jmem];
		
		if(ich>=nOffset && ich<nOffset+NPN) {
		  _tausPNEE[0][ich-nOffset][jmem]=tau1; 
		  _tausPNEE[1][ich-nOffset][jmem]=tau2;
		  _qmaxPNEE[ich-nOffset][jmem]=getqmax(tau1,tau2);
		  //if(_debug) cout<<"TCalibData::readTaus passing ich "<<ich<<" "<<nOffset<<" "<<jmem<<" "<<mems[jmem]<<" "<<tau1<<" "<<tau2<<" "<<_qmaxPNEE[ich-nOffset][jmem]<< endl;
		}
	      }
	    }
      
	fclose(sprpnf);

      }
    }
    
    ok=true;
  }




  if(_debug) cout <<"TCalibData::readTaus FINISHED" << endl;

  return ok;

}

bool TCalibData::readLinPN( ){
  
  bool ok=false;
  
  int nlines=NPN;
  float a1, a2, a3, a4;
  
  int i,j;
  if(_debug) cout << "TCalibData::readLinPN " << _isBarrel<<"  " << _linPNfile<< endl;
  
  FILE *pncorfile;
  pncorfile = fopen(_linPNfile.c_str(), "r");
  
  if(pncorfile == NULL){
    cout<< "Error while opening CORLIN_PN file : "<<_linPNfile.c_str()<< endl;
    delete pncorfile;
    return ok;
  }else if(_isBarrel){
    int ng=NGAIN;
    if(_newLinType) ng=1;
    for(int igain=0; igain<ng; igain++){
      for(int jpn=0; jpn<nlines; jpn++){
	if(!_newLinType) i = fscanf(pncorfile, "%f %f %f", &a1, &a2, &a3 );
	else i = fscanf(pncorfile, "%d %f %f %f %f", &j, &a1, &a2, &a3, &a4 );
	if((i != 3 && !_newLinType) || (i != 5 && _newLinType)){
	  printf("Error : read only %d parameters from corlin_pn file %s\n",i,_linPNfile.c_str());
	}
	_linPNEB[igain][0][jpn]=a1;
	_linPNEB[igain][1][jpn]=a2;
	_linPNEB[igain][2][jpn]=a3;
	if(!_newLinType) _linPNEB[igain][3][jpn]=0.;
	else _linPNEB[igain][3][jpn]=a4;
	
      }
    }

    //FIXME: check this with marc...
    // if(_newLinType)
//       {
//         for(jpn=0; jpn<nlines; jpn++)
// 	  {
// 	    _linPNEB[1][0][jpn]=_linPNEB[0][0][jpn]/17.5;
// 	    _linPNEB[1][1][jpn]=_linPNEB[0][1][jpn]/17.5;
// 	    _linPNEB[1][2][jpn]=_linPNEB[0][2][jpn]/17.5;
// 	    _linPNEB[1][3][jpn]=_linPNEB[0][3][jpn]/17.5;
// 	  }
//       }
    fclose(pncorfile);

    ok=true;
  }else{
    
    int pnOffset[NMEMEE];
    
    vector<int> mems= ME::memFromDcc(_fed);
    assert(mems.size()==_nmemEE);
    
    for(unsigned int jmem=0;jmem<mems.size();jmem++){
      pnOffset[jmem]=PNOffset(mems[jmem]);
    }
    if(_debug) cout <<"TCalibData::readLinPN  PNCOR CHECK "<<mems[0]<<" "<<mems[1]<<" "<<pnOffset[0]<<" "<<pnOffset[1]<<" "<< nlines<< " "<<NGAIN<< endl;

    nlines=80;
    int ng=NGAIN;if(_newLinType) ng=1;
    for(int igain=0; igain<ng; igain++){
      for(int jpn=0; jpn<nlines; jpn++){

	if(!_newLinType) i = fscanf(pncorfile, "%f %f %f", &a1, &a2, &a3 );
	else i = fscanf(pncorfile, "%d %f %f %f %f",  &j, &a1, &a2, &a3, &a4 );
	if((i != 3 && !_newLinType) || (i != 5 && _newLinType)){
	  cout<< "Error : read only "<< i<< " parameters from corlin_pn file "<<_linPNfile.c_str()<< endl;
	}
	
	for(unsigned int jmem=0;jmem<mems.size();jmem++){
	  int jjpn=jpn-pnOffset[jmem];
	  
	  if(jjpn>=0 && jjpn<NPN){
	    if(_debug) cout <<"TCalibData::readLinPN  passed "<<jjpn<<" "<<pnOffset[jmem]<<" "<<mems[jmem]<<" "<< jmem<<" "<<igain<<" "<<a1<<" "<<a2<<" "<<a3<< endl;
	    _linPNEE[igain][0][jjpn][jmem]=a1;
	    _linPNEE[igain][1][jjpn][jmem]=a2;
	    _linPNEE[igain][2][jjpn][jmem]=a3;
	    if(!_newLinType)_linPNEE[igain][3][jjpn][jmem]=0.;
	    else _linPNEE[igain][3][jjpn][jmem]=a4;	    
	  }
	}
      }
    }  

    // FIXME: do this...
    // if(_newLinType)
//       {
//         for(jpn=0; jpn<NPN; jpn++)
// 	  {
// 	    _linPNEE[1][0][jpn]=_linPNEE[0][0][jpn]/17.5;
// 	    _linPNEE[1][1][jpn]=_linPNEE[0][1][jpn]/17.5;
// 	    _linPNEE[1][2][jpn]=_linPNEE[0][2][jpn]/17.5;
// 	    _linPNEE[1][3][jpn]=_linPNEE[0][3][jpn]/17.5;
// 	  }
//       }
    fclose(pncorfile);
    
    ok=true;
  }
  return ok;
}

// From Global Coordinates:
pair<double,double> TCalibData::tauAPD(int ieta, int iphi){

  if(!_tauRead)_tauRead =readTaus();
  
  pair<double,double> taus;
  
  if(_tauRead){  
    
    if(_isBarrel){
      
      pair<int, int> LocalCoord=MEEBGeom::localCoord( ieta , iphi );
      
      int etaL=LocalCoord.first ; // local
      int phiL=LocalCoord.second ;// local

      int chan=MEEBGeom::electronic_channel( etaL, phiL );
      taus.first=_tausAPD[0][chan];
      taus.second=_tausAPD[1][chan];

    }else{
      
      if(_debug) cout<< "TCalibData::tauAPD etaphi:"<<ieta<<" "<<iphi<< endl;
      double iChPlusOne=EEchannel(ieta,iphi);
      if(_debug) cout<< "TCalibData::tauAPD iChPlusOne:"<<iChPlusOne<< endl;
      assert(iChPlusOne>0);
      int ich=int(iChPlusOne)-1;
      taus=tauAPD(ich);
      
    }
  }
  
  return taus;  
}


int TCalibData::EEchannel(int ix, int iy){

  if(_debug) cout << "TCalibData::EEchannel " <<ix<<" "<<iy<< endl;
  assert(ix>0 && iy>0);

  if(_debug) cout << "TCalibData::EEchannel " << endl;
  double chd=EENum->GetBinContent(ix,iy);
  int ch=int(chd);
  if(_debug) cout << "TCalibData::EEchannel " <<ch<< endl;
  return ch;

}
// From Global Coordinates:
pair<double,double> TCalibData::tauAPD(int ich){
  
  if(!_tauRead)_tauRead =readTaus();
  
  pair<double,double> taus;
  
  taus.first=_tausAPD[0][ich];
  taus.second=_tausAPD[1][ich];
  
  return taus;  
} 
pair<double,double> TCalibData::tauPN( int ipn ){

  assert(_isBarrel);

  pair<double,double> taus;

  if(!_tauRead)_tauRead =readTaus();
  
  assert(_tauRead);
  assert (ipn<NPN);
  
  taus.first=_tausPNEB[0][ipn];
  taus.second=_tausPNEB[1][ipn];
  
  return taus;
  
  
} 

pair<double,double> TCalibData::tauPN( int ipn , int imem ){
  
  
  pair<double,double> taus;
  
  if(_isBarrel){
    taus=tauPN( ipn ); 

  }else{
    
    if(!_tauRead)_tauRead =readTaus();    
    
    assert(_tauRead);
    assert (ipn<NPN);
    
    vector<int> mems= ME::memFromDcc(_fed);
    assert(mems.size()==_nmemEE);



    int jmem=0; int found=0;
    
    for(unsigned int jmm=0;jmm<_nmemEE;jmm++){
      if(mems[jmm]==imem){
	jmem=jmm;
	found=1;
	jmm=_nmemEE;
      }
    }
    

    if(found==0){
      cout<< "Unknown MEM:"<<imem<<" for FED:"<<_fed<<" => abort"<<endl;
      abort();
    }
    
    if(_debug)cout<< "TCalibData::tauPN "<<ipn<<" "<<imem<<" "<< _tausPNEE[0][ipn][jmem]<<" "<< _tausPNEE[1][ipn][jmem]<<endl;

    taus.first=_tausPNEE[0][ipn][jmem];
    taus.second=_tausPNEE[1][ipn][jmem];  
  }
  
  return taus;
  
}

double TCalibData::qmaxPN( int iPN ){

  assert(_isBarrel);
  if(!_tauRead)_tauRead =readTaus(); 
  return _qmaxPNEB[iPN];
}

double TCalibData::qmaxPN( int iPN, int imem ){
  
  double qmax; 
  if(!_tauRead)_tauRead =readTaus(); 
  
  if(_isBarrel){
    qmax=_qmaxPNEB[iPN];
  }else{    
    
    vector<int> mems= ME::memFromDcc(_fed);
    assert(mems.size()==_nmemEE);

    
    int jmem=0; int found=0;
    
    for(unsigned int jmm=0;jmm<_nmemEE;jmm++){
      if(mems[jmm]==imem){
	jmem=jmm;
	found=1;
	jmm=_nmemEE;
      }
    }

    if(found==0){
      cout<< "Unknown MEM:"<<imem<<" for FED:"<<_fed<<" => abort"<<endl;
      abort();
    }

    if(_debug) cout<< " TCalibData::qmaxPN  "<<jmem<<" "<< _qmaxPNEE[iPN][jmem]<< endl;
    qmax=_qmaxPNEE[iPN][jmem];
  }
  return qmax;
  
}
      

double TCalibData::getqmax( double tau1, double tau2 )
{

  //  if(_debug) cout<<  "TCalibData::getqmax taus "<< tau1<<" "<<tau2<< endl;
  double qmax=0.;
  double a=tau2/(tau2-tau1);
  //  if(_debug) cout<<  "TCalibData::getqmax  a "<< a << endl;
  double t, y;
  for(int i=0; i<1250; i++)
    {
      t=(double)i/25.; 
      y= a*(1.-a)*(TMath::Exp(-t/tau1)-TMath::Exp(-t/tau2))+t/tau1*(1.-a-t/2./tau1)*TMath::Exp(-t/tau1);      
      
      //if(i%50) cout<<  "getqmax y["<<i<<"]="<< y <<" "<< qmax<< endl;
      if(y>qmax)qmax=y;
    }
  
  if(_debug) cout<<"TCalibData::getqmax qmax "<<qmax<< endl;
  return qmax;
}

vector<double> TCalibData::linPN( int iPN, int gain ){
  
  vector<double> vecLinPN;
  assert ( _isBarrel);  
  if(! _linPNRead ) _linPNRead=readLinPN();
  assert(  _linPNRead );
  
  vecLinPN.push_back(_linPNEB[gain][0][iPN]);
  vecLinPN.push_back(_linPNEB[gain][1][iPN]);
  vecLinPN.push_back(_linPNEB[gain][2][iPN]);
  if(_newLinType) vecLinPN.push_back(_linPNEB[gain][3][iPN]);
  
  return vecLinPN;
}

vector<double> TCalibData::linPN( int iPN, int gain , int imem ){
  
  vector<double> vecLinPN;
  
  if(! _linPNRead ) _linPNRead=readLinPN();
  assert(_linPNRead);

  if(_isBarrel){
    vecLinPN=linPN( iPN, gain ); 
  }else{ 

    vector<int> mems= ME::memFromDcc(_fed);
    assert(mems.size()==_nmemEE);


    int jmem=0; int found=0;
    
    for(unsigned int jmm=0;jmm<_nmemEE;jmm++){
      if(mems[jmm]==imem){
	jmem=jmm;
	found=1;
	jmm=_nmemEE;
      }
    }

    if(found==0){
      cout<< "Unknown MEM:"<<imem<<" for FED:"<<_fed<<" => abort"<<endl;
      abort();
    }

    vecLinPN.push_back(_linPNEE[gain][0][iPN][jmem]);
    vecLinPN.push_back(_linPNEE[gain][1][iPN][jmem]);
    vecLinPN.push_back(_linPNEE[gain][2][iPN][jmem]);
    if(_newLinType) vecLinPN.push_back(_linPNEE[gain][3][iPN][jmem]);
  }
  return vecLinPN;
}

double
TCalibData::getPNCorrected( double val0 , int iPN, int gain , int imem)
{
  
  vector<double> par;

  if(gain!=0 && _newLinType ){
    par=linPN( iPN,  0 ,  imem );    
    for(int jj=0;jj<4;jj++){
      par[jj]=par[jj]/17.5;
    }
  }else{
    par=linPN( iPN,  gain ,  imem );
  }
  
  double cor=0.0;  

  double pn=val0;
  double xpn=pn/1000.0;
  
  if(!_newLinType){
    assert(par.size()==3);
    cor=xpn*(par[0] +xpn*(par[1]+xpn*par[2]));
    return val0-cor;
  }else{
    assert(par.size()==4); 
    cor = par[0]+ xpn*(par[1]+xpn*(par[2]+xpn*par[3]));
    return val0*(1. - cor);
  }
    
}

int TCalibData::PNOffset( int imem ){

  int pnOffset=0;
  
  if(imem%600==2) pnOffset=10;
  else if(imem%600==5) pnOffset=20;
  else if(imem%600==6) pnOffset=30;
  else if(imem%600==46) pnOffset=40;
  else if(imem%600==47) pnOffset=50;
  else if(imem%600==50) pnOffset=60;
  else if(imem%600==51) pnOffset=70;

  return pnOffset;
      
}

bool TCalibData::readAB(int color){

  bool ok=true;

  // FIXME: take color=0 until ABred trees done
  int icol=0;
  stringstream abname;
  abname<<"ABCol"<<icol;
  if(! ABFile) ok=false;


  if(ok){ 
    ABTree[color]=(TTree*) ABFile->Get(abname.str().c_str());
    if(!ABTree[color]) ok=false;
  }
  if(_debug) cout<< "TCalibData::readAB tree found? "<<ok<< endl;

  if(ok){  
    ABTree[color]->SetBranchAddress("channel", &channelAB, &b_channel );
    ABTree[color]->SetBranchAddress("iphi",    &iphiAB,    &b_iphi    );
    ABTree[color]->SetBranchAddress("ieta",    &ietaAB,    &b_ieta    );
    ABTree[color]->SetBranchAddress("alpha",   &alpha,     &b_alpha   );
    ABTree[color]->SetBranchAddress("beta",    &beta,      &b_beta    );

    ABTree[color]->BuildIndex("channel");
    if(_debug) cout<< "TCalibData::readAB Index Build"<< endl;
  }  
  
  return ok;  
}

pair < double, double> TCalibData::AB( int ieta, int iphi, int color) {

  pair < double, double> ab;   
  int chan;

  if(_isBarrel){
    pair<int, int> LocalCoord=MEEBGeom::localCoord( ieta , iphi );
    int etaL=LocalCoord.first ; // local
    int phiL=LocalCoord.second ;// local
    chan=MEEBGeom::electronic_channel( etaL, phiL );
  }else{ 
    double iChPlusOne=EEchannel(ieta,iphi);
    assert(iChPlusOne>0);
    chan=int(iChPlusOne)-1;
  }
  int ietaFromFile= 0;
  int iphiFromFile= 0;

  ab=AB( chan , color, ietaFromFile, iphiFromFile );
  
  if(_debug) cout<< "TCalibData::AB1 "<< ieta<<" "<<iphi<<" "<<color<< " "<<chan<<" "<<ietaFromFile<<" "<<iphiFromFile<<endl;
  
  assert( ietaFromFile==ieta && iphiFromFile==iphi);
  return ab;
  
}

pair < double, double> TCalibData::AB( int channel , int color, int& ieta, int& iphi) {
  
  pair < double, double> ab;
  ab.first=0.0;
  ab.second=0.0;
  ieta=0;
  iphi=0;
  
  if(!_ABRead[color]) _ABRead[color]=readAB(color);
  if(_ABRead[color]){
    
    ABTree[color]->GetEntryWithIndex(channel);
    
    if(_debug) cout<< "TCalibData::AB2 coord "<< ietaAB<<" "<<iphiAB<<" "<<color<< " "<<channel<<" "<< channelAB<< endl;

    assert( channelAB == channel );

    ieta=ietaAB;
    iphi=iphiAB;
    ab.first=alpha;
    ab.second=beta;

    if(_debug) cout<< "TCalibData::AB2 AB:"<< alpha<<" "<<beta<<" "<< ieta<<" "<< iphi<<endl;
    
  }    
  return ab;
}
