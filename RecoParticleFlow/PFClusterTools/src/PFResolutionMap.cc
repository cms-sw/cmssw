#include "RecoParticleFlow/PFClusterTools/interface/PFResolutionMap.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <TFile.h>
#include <TMath.h>
#include <vector>
using namespace std;


const unsigned PFResolutionMap::lineSize_ = 10000;


PFResolutionMap::PFResolutionMap(const char* name, unsigned nbinseta,  
				 double mineta, double maxeta,
				 unsigned nbinse, 
				 double mine, double maxe, double value) 
  : TH2D(name, name, nbinseta, mineta, maxeta, nbinse, mine, maxe) {
  // fNbinsEta(nbinseta), fMinEta(mineta), fMaxEta(maxeta),
  //    fNbinsE(nbinse), fMinE(mine), fMaxE(maxe) {

  GetXaxis()->SetTitle("#eta");
  GetYaxis()->SetTitle("E");

  if(value>0) {
    for(int ie=1; ie<=GetNbinsY(); ie++) {
      for(int ieta=1; ieta<=GetNbinsX(); ieta++) {
	SetBinContent(ieta,ie, value);
      }
    }  
  }
  // cout<<"creating resolution map "<<endl;
  // Print("all");
  
}

// void PFResolutionMap::Init(unsigned nbinseta,  
// 			   double mineta, double maxeta,
// 			   unsigned nbinse, 
// 			   double mine, double maxe) {
//   assert(mineta<maxeta);
//   assert(mine<maxe);
  
//   unsigned nbins =  nbinseta*nbinse;
//   fData.reserve( nbins );
//   for(unsigned i=0; i<nbins; i++) {
//     fData.push_back(-1);
//   } 
  
//   // calculate lower bound of eta and e bins
//   double binsize =  (fMaxEta - fMinEta)/fNbinsEta;
//   double lb = fMinEta;
//   // cout<<"eta bins lower bounds : "<<endl;
//   for(unsigned i=0; i<nbinseta; i++) {
//     fEtaBinsLowerBounds[lb] = i; 
//     // cout<<i<<" "<<lb<<endl;
//     lb += binsize;
//   }
  
//   binsize =  (fMaxE - fMinE)/fNbinsE;
//   lb = fMinE;
//   // cout<<"E bins lower bounds : "<<endl;
//   for(unsigned i=0; i<nbinse; i++) {
//     fEBinsLowerBounds[lb] = i; 
//     // cout<<i<<" "<<lb<<endl;
//     lb += binsize;
//   } 
// }

PFResolutionMap::PFResolutionMap(const char* name, const char* mapfile) { 
  SetTitle(name);
  GetXaxis()->SetTitle("#eta");
  GetYaxis()->SetTitle("E");
  if( ! ReadMapFile(mapfile) ) {
    string err = "PFResolutionMap::PFResolutionMap : cannot read file ";
    err += mapfile;
    throw invalid_argument(err);
  }
}


bool PFResolutionMap::WriteMapFile(const char* mapfile) {

  //  assert(fData.size() == fNbinsEta*fNbinsE);


  // open the file

  ofstream outf(mapfile);
  if( !outf.good() ) {
    cout<<"PFResolutionMap::Write : cannot open file "<<mapfile<<endl;
    return false;
  }
  
  outf<<(*this)<<endl;
  if(!outf.good() ) {
    cerr<<"PFResolutionMap::Write : corrupted file "<<mapfile<<endl;
    return false;
  }
  else {
    mapFile_ = mapfile;
    return true;
  }
}



bool PFResolutionMap::ReadMapFile(const char* mapfile) {
  
  // open the file

  ifstream inf(mapfile);
  if( !inf.good() ) {
    // cout<<"PFResolutionMap::Read : cannot open file "<<mapfile<<endl;
    return false;
  }
  
  // first data describes the map

  int nbinseta=0;
  double mineta=0;
  double maxeta=0;
  
  int nbinse=0;
  double mine=0;
  double maxe=0;
  
  inf>>nbinseta;
  inf>>mineta;
  inf>>maxeta;

  inf>>nbinse;
  inf>>mine;
  inf>>maxe;

  SetBins(nbinseta, mineta, maxeta, nbinse, mine, maxe);

  // Init(fNbinsEta,fMinEta,fMaxEta,fNbinsE,fMinE,fMaxE);

  // char data[lineSize_];
  char s[lineSize_];
  int pos=inf.tellg();

  // parse map data
  int i=1;
  do { 
    inf.seekg(pos);
    inf.getline(s,lineSize_);
    
    pos = inf.tellg();     
 
    if(string(s).empty()) {
      continue; // remove empty lines
    }

    istringstream lin(s);  

    double dataw;
    int j=1;
    do {
      lin>>dataw;
      SetBinContent(j, i, dataw);
      j++;
    } while (lin.good() );
    i++;
  } while(inf.good());
  
  if(inf.eof()) {
    mapFile_ = mapfile;  
    return true;
  }
  else {
    // cout<<"error"<<endl;
    return false;
  }
  
  mapFile_ = mapfile;  
  return true;
}



// int PFResolutionMap::FindBin(double eta, double e) const {
  
//   if(eta<fMinEta || eta>=fMaxEta) {
//     cout<<"PFResolutionMap::FindBin "<<eta<<" out of eta bounds "<<fMinEta<<" "<<fMaxEta<<endl;
//     return -1;
//   }
//   if(e<fMinE || e>=fMaxE) {
//     cout<<"PFResolutionMap::FindBin "<<e<<" out of e bounds "<<fMinE<<" "<<fMaxE<<endl;
//     return -1;
//   }
  
//   map<double,unsigned >::const_iterator iteta = 
//     fEtaBinsLowerBounds.upper_bound( eta );
//   iteta--;
		 
// //   if( iteta != fEtaBinsLowerBounds.end() ) {
// //     cout<<"eta lower bound "<<iteta->first<<" "<<iteta->second<<endl;
// //   }
// //   else return -1;

//   map<double,unsigned>::const_iterator ite = 
//     fEBinsLowerBounds.upper_bound( e );
//   ite--;
// //   if( ite != fEBinsLowerBounds.end() ) {
// //     cout<<"e lower bound "<<ite->first<<" "<<ite->second<<endl;
// //   }
// //   else return -1;

// //   cout<<"returning "<<ite->second * fNbinsEta + iteta->second<<endl;
// //   cout<<"returning "<<ite->second<<" "<<iteta->second<<endl;

//   return ite->second * fNbinsEta + iteta->second;
// }

// void PFResolutionMap::Fill(double eta, double e, double res) {

//   unsigned bin = FindBin(eta, e);
//   if( bin<0 || bin>fData.size() ) {
//     // cout<<"PFResolutionMap::Fill : out of range " <<bin<<endl;
//     return;
//   }
  
//   fData[bin] = res;
// }


double PFResolutionMap::getRes(double eta, double phi, double e, int MapEta){
  static double fMinEta = -2.95;
  static double fMaxEta = 2.95;
  static double fMinE=0;
  static double fMaxE=100;

  if( eta<fMinEta ) eta = fMinEta+0.001;
  if( eta>fMaxEta ) eta = fMaxEta-0.001;
 
  if( e<fMinE ) e = fMinE+0.001;
  if( e>fMaxE ) e = fMaxE-0.001;

  unsigned bin = FindBin(TMath::Abs(eta),e);

  double res= GetBinContent(bin);
  if(MapEta>-1){
    if((eta<1.48) && IsInAPhiCrack(phi,eta)) {
      if(MapEta==1) res *= 1.88;
      else res *= 1.65;
    }
  }
  return res;
}



int PFResolutionMap::FindBin(double eta, double e) {
  if(e >= GetYaxis()->GetXmax() )
    e = GetYaxis()->GetXmax() - 0.001;
  
  return TH2D::FindBin(eta,e);
}



ostream& operator<<(ostream& outf, const PFResolutionMap& rm) {

  if(!outf.good() ) return outf;

  // first data describes the map
  outf<<rm.GetNbinsX()<<endl;
  outf<<rm.GetXaxis()->GetXmin()<<endl;
  outf<<rm.GetXaxis()->GetXmax()<<endl;

  outf<<rm.GetNbinsY()<<endl;
  outf<<rm.GetYaxis()->GetXmin()<<endl;
  outf<<rm.GetYaxis()->GetXmax()<<endl;

  for(int ie=1; ie<=rm.GetNbinsY(); ie++) {
    for(int ieta=1; ieta<=rm.GetNbinsX(); ieta++) {
      outf<<rm.GetBinContent(ieta,ie)<<"\t";
    }
    outf<<endl;
  }
  
  return outf;
}

bool PFResolutionMap::IsInAPhiCrack(double phi, double eta){
  double dminPhi = dCrackPhi(phi,eta);
  bool Is = (TMath::Abs(dminPhi)<0.005);
  return Is;
}

//useful to compute the signed distance to the closest crack in the barrel
double PFResolutionMap::minimum(double a,double b){
  if(TMath::Abs(b)<TMath::Abs(a)) a=b;
  return a;
}


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//compute the unsigned distance to the closest phi-crack in the barrel
double PFResolutionMap::dCrackPhi(double phi, double eta){

  static double pi= M_PI;// 3.14159265358979323846;
  
  //Location of the 18 phi-cracks
  static std::vector<double> cPhi;
  cPhi.resize(18,0);
  cPhi[0]=2.97025;
  for(unsigned i=1;i<=17;i++) cPhi[i]=cPhi[0]-2*i*pi/18;

  //Shift of this location if eta<0
  static double delta_cPhi=0.00638;

  double m; //the result

  //the location is shifted
  if(eta<0){ 
    phi +=delta_cPhi;
    if(phi>pi) phi-=2*pi;
  }
  if (phi>=-pi && phi<=pi){

    //the problem of the extrema
    if (phi<cPhi[17] || phi>=cPhi[0]){
      if (phi<0) phi+= 2*pi;
      m = minimum(phi -cPhi[0],phi-cPhi[17]-2*pi);        	
    }

    //between these extrema...
    else{
      bool OK = false;
      unsigned i=16;
      while(!OK){
	if (phi<cPhi[i]){
	  m=minimum(phi-cPhi[i+1],phi-cPhi[i]);
	  OK=true;
	}
	else i-=1;
      }
    }
  }
  else{
    m=0.;        //if there is a problem, we assum that we are in a crack
    std::cout<<"Problem in dminphi"<<std::endl;
  }
  if(eta<0) m=-m;   //because of the disymetry
  return m;
}
