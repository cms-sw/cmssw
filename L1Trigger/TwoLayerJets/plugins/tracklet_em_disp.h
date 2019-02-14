#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <cstdlib>
using namespace std;

//#~~~~~~~~~~ 2 layer structs added here ~~~~~~~~~~~~~~
/*
  struct mc_data {
        float ogeta;
        float ogphi;
        float ogpt;
      float ogz;
        int ntracks;
//  int ntightrks;
//  int ntdisptrks;
};
*/
//Holds data for each input track of the event.
/*
struct track_data {

  float pT;
  float eta;
  float z;
  float phi;
  int ttrk;
  int tdtrk;
  int ttdtrk;
  int bincount;  //How many zbins it's gone into (to make sure it doesn't go into more than 2): start at 0

};
*/
//Each individual box in the eta and phi dimension.
//  Also used to store final cluster data for each zbin.
struct etaphibin {

  float pTtot;
  int numtracks;
  int numttrks;
  int numtdtrks;
  int numttdtrks;
  bool used;
   //average phi value (halfway b/t min and max)
    float phi;
   //average eta value
    float eta;

};

//store important information for plots
struct maxzbin {
  int znum;        //Numbered from 0 to nzbins (16, 32, or 64) in order.-
  //mc_data * mcd;   //array of jets from input file.
  int nclust;      //number of clusters in this bin.
  float zbincenter;
//  int nttrks;   //number of tigh tracks.
//  int ntdtrks;  //number of tigh displaced tracks.
  etaphibin * clusters;     //list of all the clusters in this bin.
  float ht;   //sum of all cluster pTs--only the zbin with the maximum ht is stored.
};

const int netabins = 24;
const float maxz = 15.0;
const float maxeta = 2.4;
//etastep is the width of an etabin
const float etastep = 2.0 * maxeta / netabins;
//Any tracks with pT > 200 GeV should be capped at 200
const float pTmax = 200.0;
//Upper bound on number of tracks per event.
const int numtracks = 500;
const int nphibins = 27;       //*******************Changed from 28***********************
//phistep is the width of a phibin.
const float phistep = 2*M_PI / nphibins;
//function to find all clusters, find zbin with max ht. In file find_clusters.cpp
//maxzbin * L2_cluster(track_data*, int, int);
//etaphibin * L1_cluster(etaphibin*);


//#~~~~~~~~~~ 2 layer structs added here ~~~~~~~~~~~~~~
