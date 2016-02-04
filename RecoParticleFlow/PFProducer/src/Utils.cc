#include "RecoParticleFlow/PFProducer/interface/Utils.h"

#include <stdio.h>
#include <regex.h>


using namespace std;

bool Utils::stringMatch(const char* str, const char* pattern) {
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

vector<string>  Utils::myGlob(const char* pattern) {

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

string   Utils::date() {
  string date("date +%d%b%Y_%H%M%S");
  FILE *in = popen(date.c_str(), "r");
  char result[100];
  if(fscanf(in,"%s",result) != EOF) {
    return string(result);
  }
  else return string("");
}


double   Utils::mpi_pi(double angle) {

  const double pi = 3.14159265358979323;
  const double pi2 = pi*2.;
  while(angle>pi) angle -= pi2;
  while(angle<-pi) angle += pi2;
  return angle;
}
