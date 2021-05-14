#pragma once
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

//Each individual box in the eta and phi dimension.
//  Also used to store final cluster data for each zbin.
struct EtaPhiBin {
  float pTtot = 0;
  int numtracks = 0;
  int numttrks = 0;
  int numtdtrks = 0;
  int numttdtrks = 0;
  bool used = true;
  float phi = 0;  //average phi value (halfway b/t min and max)
  float eta = 0;  //average eta value
};

//store important information for plots
struct MaxZBin {
  int znum = 0;    //Numbered from 0 to nzbins (16, 32, or 64) in order
  int nclust = 0;  //number of clusters in this bin
  float zbincenter = 0;
  EtaPhiBin *clusters = nullptr;  //list of all the clusters in this bin
  float ht = 0;                   //sum of all cluster pTs--only the zbin with the maximum ht is stored
};
