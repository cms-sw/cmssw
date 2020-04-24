#include "RecoMuon/MuonIsolation/src/NominalEfficiencyThresholds.h"

#include <cmath>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
using namespace muonisolation;
using namespace std;

NominalEfficiencyThresholds::EtaBounds::EtaBounds()
{
  float BaseEtaBin = 0.087;
  theBounds[0]=0.0;
  for (int it=1; it <= 20; it++) theBounds[it] = it*BaseEtaBin;
  theBounds[21]=1.83;
  theBounds[22]=1.93;
  theBounds[23]=2.043;
  theBounds[24]=2.172;
  theBounds[25]=2.322;
  theBounds[26]=2.5;
  theBounds[27]=2.65;
  theBounds[28]=3.0;
  theBounds[29]=3.13;
  theBounds[30]=3.305;
  theBounds[31]=3.48;
  theBounds[32]=3.655;
}

int NominalEfficiencyThresholds::EtaBounds::towerFromEta(double eta) const
{
  int number = 0;
  for (int num = 1; num <= NumberOfTowers; num++) {
    if ( fabs(eta) > theBounds[num-1]) {
      number = num;
      if (eta < 0) number = -num;
    }
  }
  if (fabs(eta) >= theBounds[NumberOfTowers-1]) number = 0;
  return number;
}

vector<double> NominalEfficiencyThresholds::bins() const
{
  vector<double> result;
  for (unsigned int i=1; i <=EtaBounds::NumberOfTowers; i++) result.push_back(etabounds(i));
  return result;  
}

bool NominalEfficiencyThresholds::EfficiencyBin::operator() (const EfficiencyBin & e1,
                                     const EfficiencyBin & e2) const
{
  return e1.eff<= e2.eff_previous;
}

bool NominalEfficiencyThresholds::locless::operator()(const ThresholdLocation & l1,
                                const ThresholdLocation & l2) const
{
  int itow1 = abs(etabounds.towerFromEta(l1.eta));
  int itow2 = abs(etabounds.towerFromEta(l2.eta));
  if (itow1 < itow2) return true;
  if (itow1 == itow2 && l1.cone< l2.cone) return true;
  return false;
}

NominalEfficiencyThresholds::NominalEfficiencyThresholds(const string & infile)
{
  FILE *INFILE;
  char buffer[81];
  char tag[5];
  if ( (INFILE=fopen(infile.c_str(),"r")) == nullptr) {
    cout << "Cannot open input file " << infile <<endl;
    return;
  }
  ThresholdLocation location;
  EfficiencyBin eb;
  float thr_val;
  while (fgets(buffer,80,INFILE)) {
    sscanf(buffer,"%4s",tag);
    if (strcmp(tag,"ver:") == 0){
      cout <<" NominalEfficiencyThresholds: "<< infile <<" comment: "<<buffer<<endl;
      thresholds.clear();
    }
    if (strcmp(tag,"loc:") == 0) {
      sscanf(buffer,"%*s %f %*s %d", &location.eta, &location.cone);
      eb.eff = 0.;
    }
    if (strcmp(tag,"thr:") == 0) {
      eb.eff_previous = eb.eff;
      sscanf(buffer,"%*s %f %f", &eb.eff, &thr_val);
      add(location,ThresholdConstituent(eb,thr_val));
    }
  }
  fclose(INFILE);
  cout << "... done"<<endl;
  //dump();
}


void NominalEfficiencyThresholds::add(ThresholdLocation location,
                           ThresholdConstituent threshold)
{
  MapType::iterator ploc = thresholds.find(location);
  if ( ploc == thresholds.end() ) {
    ThresholdConstituents mt;
    mt.insert(threshold);
    thresholds[location] = mt;
  } else {
//    cout << "insert element ("<<threshold.first.eff<<","
//                            <<threshold.first.eff_previous<<") to map "<<endl;
    (*ploc).second.insert(threshold);
//    cout << "new size is:"<< (*ploc).second.size() <<endl;
  }
}

void NominalEfficiencyThresholds::dump()
{
  MapType::iterator ploc;
  for (ploc = thresholds.begin(); ploc != thresholds.end(); ploc++) {
    cout << "eta: "<< (*ploc).first.eta
         << " icone: "<< (*ploc).first.cone<<endl;
    ThresholdConstituents::iterator it;
    for (it = (*ploc).second.begin(); it != (*ploc).second.end(); it++) {
      cout << " eff: "         << (*it).first.eff
           << " eff_previous: "<< (*it).first.eff_previous
           << " cut value: "   << (*it).second <<endl;
    }
  }
}

float NominalEfficiencyThresholds::thresholdValueForEfficiency( 
   ThresholdLocation location, float eff_thr) const 
{
  MapType::const_iterator ploc = thresholds.find(location);
  if ( ploc == thresholds.end() ) {
    cout << "NominalEfficiencyThresholds: Problem:can't find location in the map :( "
       << location.eta << " " << location.cone << " " << eff_thr
       <<endl;
    return -1;
  }

  const float epsilon=1.e-6;
  EfficiencyBin eb = {eff_thr,eff_thr-epsilon};
  ThresholdConstituents::const_iterator it = (*ploc).second.find(eb);
  if (it == (*ploc).second.end()) {
    cout << "NominalEfficiencyThresholds: Problem:can't find threshold in the map :("<<endl;
    return -1;
  }

  return (*it).second;
}

