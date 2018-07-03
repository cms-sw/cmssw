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
#include "FWCore/MessageLogger/interface/MessageLogger.h"


const unsigned PFResolutionMap::lineSize_ = 10000;


PFResolutionMap::PFResolutionMap(const char* name, unsigned nbinseta,  
				 double mineta, double maxeta,
				 unsigned nbinse, 
				 double mine, double maxe, double value) 
  : TH2D(name, name, nbinseta, mineta, maxeta, nbinse, mine, maxe) {

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

  std::ofstream outf(mapfile);
  if( !outf.good() ) {
    edm::LogWarning("PFResolutionMap::Write")<<" : cannot open file "<<mapfile;
    return false;
  }
  
  outf<<(*this)<<endl;
  if(!outf.good() ) {
    edm::LogError("PFResolutionMap::Write")<<" : corrupted file "<<mapfile;
    return false;
  }
  else {
    mapFile_ = mapfile;
    return true;
  }
}



bool PFResolutionMap::ReadMapFile(const char* mapfile) {
  
  // open the file

  std::ifstream inf(mapfile);
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




double PFResolutionMap::getRes(double eta, double phi, double e, int MapEta){
  constexpr double fMinEta = -2.95;
  constexpr double fMaxEta = 2.95;
  constexpr double fMinE=0;
  constexpr double fMaxE=100;

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



int PFResolutionMap::FindBin(double eta, double e, double z) {
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

  constexpr double pi= M_PI;// 3.14159265358979323846;
  constexpr double twopi= 2.*pi;
  constexpr double twopiO18= pi/9;
  
  //Location of the 18 phi-cracks
  constexpr double c0 = 2.97025;
  constexpr std::array<double,18> cPhi {{c0, 
					c0-twopiO18, c0-2*twopiO18, c0-3*twopiO18, c0-4*twopiO18,
					c0-5*twopiO18, c0-6*twopiO18, c0-7*twopiO18, c0-8*twopiO18,
					c0-9*twopiO18, c0-10*twopiO18, c0-11*twopiO18, c0-12*twopiO18,
					c0-13*twopiO18, c0-14*twopiO18, c0-15*twopiO18, c0-16*twopiO18,
					c0-17*twopiO18}};

  //Shift of this location if eta<0
  constexpr double delta_cPhi=0.00638;

  double m; //the result

  //the location is shifted
  if(eta<0){ 
    phi +=delta_cPhi;
    if(phi>pi) phi-=twopi;
  }
  if (phi>=-pi && phi<=pi){

    //the problem of the extrema
    if (phi<cPhi[17] || phi>=cPhi[0]){
      if (phi<0) phi+= twopi;
      m = minimum(phi -cPhi[0],phi-cPhi[17]-twopi);        	
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
    edm::LogWarning("PFResolutionMap:Problem")<<"Problem in dminphi";
  }
  if(eta<0) m=-m;   //because of the disymetry
  return m;
}
