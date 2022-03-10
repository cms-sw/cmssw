#pragma once
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

//Each individual box in the eta and phi dimension.
//  Also used to store final cluster data for each zbin.
struct EtaPhiBin {
  float pTtot;
  int numtracks;
  int numttrks;
  int numtdtrks;
  int numttdtrks;
  bool used;
  float phi;  //average phi value (halfway b/t min and max)
  float eta;  //average eta value
  std::vector<unsigned int> trackidx;
};

//store important information for plots
struct MaxZBin {
  int znum;    //Numbered from 0 to nzbins (16, 32, or 64) in order
  int nclust;  //number of clusters in this bin
  float zbincenter;
  EtaPhiBin *clusters;  //list of all the clusters in this bin
  float ht;             //sum of all cluster pTs--only the zbin with the maximum ht is stored
};
