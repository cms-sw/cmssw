#include "RecoParticleFlow/PFRootEvent/interface/Utils.h"

#include <stdio.h>
#include <math.h>
#include <regex.h>
#include <glob.h>

#include "TCanvas.h"
#include "TVector3.h"

using namespace std;

bool Utils::StringMatch(const char* str, const char* pattern) {
  int    status;
  regex_t    re;
  
  if( regcomp(&re, pattern, REG_EXTENDED|REG_NOSUB) != 0 )
    return false;
  
  status = regexec(&re, str, (size_t)0, NULL, 0);
  regfree(&re);
  
  if (status != 0)
    return false;
  
  return true;
}


TCanvas* Utils::DivideCanvas( TCanvas *cv, int nPads ) {  
  if( !cv ) return 0;
  
  if( nPads<=1 ) return cv;
  int sqnP = (unsigned int) (sqrt( nPads ));
  int nC = sqnP;
  int nL = sqnP;
  
  while( nC*nL < nPads ) 
    if( nC < nL ) nC++;
    else nL++;
  
  cv->Divide( nC, nL, 0.005, 0.005, 0 );
  return cv;
}

vector<string>  Utils::Glob(const char* pattern) {

  glob_t globbuf;

  globbuf.gl_offs = 2;
  glob(pattern, GLOB_TILDE|GLOB_BRACE|GLOB_MARK, NULL, &globbuf);

  vector<string> results;
  for(unsigned i=0; i<globbuf.gl_pathc; i++) {
    results.push_back(globbuf.gl_pathv[i]);
  }
  
  globfree(&globbuf);
  return results;
}

string   Utils::Date() {
  string date("date +%d%b%Y_%H%M%S");
  FILE *in = popen(date.c_str(), "r");
  char result[100];
  if(fscanf(in,"%s",result) != EOF) {
    return string(result);
  }
  else return string("");
}


TVector3 Utils::VectorEPRtoXYZ( const TVector3& posepr ) {
  TVector3 posxyz(1,1,1); // to be called before a SetMag
  posxyz.SetMag( posepr.Z() );
  double theta = 2*atan( exp( - posepr.X() ) );
  posxyz.SetTheta( theta );
  posxyz.SetPhi( posepr.Y() );  

  return posxyz;
}


