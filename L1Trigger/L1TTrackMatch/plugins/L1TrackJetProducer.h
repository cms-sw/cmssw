#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <cstdlib>
using namespace std;

//Each individual box in the eta and phi dimension.
//  Also used to store final cluster data for each zbin.
struct etaphibin {
  float pTtot;
  int numtracks;
  int numttrks;
  int numtdtrks;
  int numttdtrks;
  bool used;
  float phi; //average phi value (halfway b/t min and max)
  float eta; //average eta value
};

//store important information for plots
struct maxzbin {
  int znum; //Numbered from 0 to nzbins (16, 32, or 64) in order
  int nclust; //number of clusters in this bin
  float zbincenter;
  etaphibin *clusters; //list of all the clusters in this bin
  float ht;   //sum of all cluster pTs--only the zbin with the maximum ht is stored
};
