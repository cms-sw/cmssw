#include "stdio.h"
#include <vector>
#include <list>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

//#include "RecoLocalMuon/DTSegment/src/DTLPPatternRecoAlgorithm.h"

#include "RecoLocalMuon/DTSegment/src/DTLPPatternRecoAlgorithm.cc"


int main(int argc, char** argv)
{
  double pzel, pxel, pexel;
  std::list<double> pz, px, pex;
  // FILE * dati;
  // dati = fopen ("dati.txt", "r");
  // int N_LINES = 8;//number of lines in the input file
  
  std::ifstream dati("dati.txt");
  std::string line;
  if (!dati.is_open()){ printf ("File could not be opened \n"); return 1;}

  while (std::getline(dati,line)) {
    if( line == "" || line[0] == '#' ) continue; // Skip comments and empty lines
    std::stringstream linestr;
    linestr << line;
    linestr >> pzel >> pxel >> pexel;
  pz.push_back(pzel);
    px.push_back(pxel);
    pex.push_back(pexel);
  }

 
//  if (dati == NULL ){ printf ("File could not be opened\n"); return 1;}
//  for (int i =0; i < N_LINES; i++) {
//   fscanf(dati, "%lf %lf %lf", &pzel, &pxel, &pexel);
//   pz.push_back(pzel);
//   px.push_back(pxel);
//   pex.push_back(pexel);
//  }   
//  fclose(dati);
  
 int  deltaFactor = 4;
  DTLPPatternReco::ResultLPAlgo theAlgoResult;
  lpAlgorithm(theAlgoResult, pz, px, pex, -1.6, 1.6, -300., 300., 100. , 4.);//1.e9 means one billion 
  printf("m = %f  q = %f\n", theAlgoResult.mVar,theAlgoResult.qVar);
  
  FILE * gplt;
  gplt = fopen("macro.gpt", "w");//print a gnuplot macro to visualize the data
  fprintf(gplt,"m = %f ; q = %f ; f(x) =  x/m - q/m; set yrange [-3:3]; plot \"dati.txt\" using 2:1:($3 * %d) with xerrorbars, f(x)", theAlgoResult.mVar,theAlgoResult.qVar, deltaFactor);
  fclose(gplt);
    
  return 0;
}

