#include <iostream>
#include <vector>
#include "string"
#include "TROOT.h"
#include "TString.h"
#include "plotter.C"

using namespace std;

void splitOption(string rawoption, string& wish, string& value, char delimiter){
  size_t posEq = rawoption.find(delimiter);
  if (posEq!=string::npos){
    wish=rawoption;
    value=rawoption.substr(posEq+1);
    wish.erase(wish.begin()+posEq, wish.end());
  }
  else{
    wish="";
    value=rawoption;
  }
}
void splitOptionRecursive(string rawoption, vector<string>& splitoptions, char delimiter){
  string suboption=rawoption, result=rawoption;
  string remnant;
  while (result!=""){
    splitOption(suboption, result, remnant, delimiter);
    if (result!="") splitoptions.push_back(result);
    suboption = remnant;
  }
  if (remnant!="") splitoptions.push_back(remnant);
}

void runplot(string full_path, int iov, int iter, string plotreq="", bool only_plotting=false){
  vector<string> pathseg;
  splitOptionRecursive(full_path, pathseg, '/');
  splitOptionRecursive(pathseg.at(pathseg.size()-1), pathseg, '/');
  //for (unsigned int f=0; f<pathseg.size(); f++) cout << pathseg.at(f) << " " << f << endl;
  string obj = pathseg.at(pathseg.size()-1);
  string path = full_path; path.erase(path.find(obj.c_str()), obj.length());

  cout << "Analyzing " << obj << " in path " << path << endl;

  // choose plot type from : "cov","shift","chi2","param","hitmap"
  vector<string> plottype;
  if (plotreq==""){
    plottype.push_back("cov");
    plottype.push_back("shift");
    plottype.push_back("chi2");
    plottype.push_back("param");
    plottype.push_back("hitmap");
  }
  else{
    if (plotreq=="cov") plottype.push_back("cov");
    else if (plotreq=="shift") plottype.push_back("shift");
    else if (plotreq=="chi2") plottype.push_back("chi2");
    else if (plotreq=="param") plottype.push_back("param");
    else if (plotreq=="hitmap") plottype.push_back("hitmap");
    else return;
  }

  // plot all detectors together or individually
  bool MergeAllDet = 1;

  //*******************************************

  const unsigned int Nplots = plottype.size();

  //plotting all detectors together

  if (MergeAllDet == 1){
    for (unsigned int i = 0; i < Nplots; i++) plotter(path.c_str(), obj.c_str(), iov, plottype.at(i).c_str(), iter, 0, only_plotting);
  }

  // plotting each detector separately, don't use this for hit-map.
  else{
    // det: 0=all,1=PXB, 2=PXF, 3=TIB, 4=TID, 5=TOB, 6=TEC
    for (unsigned int i = 0; i < Nplots; i++) {
      for (unsigned int det = 0; det < 6; det++) plotter(path.c_str(), obj.c_str(), iov, plottype.at(i).c_str(), iter, det, only_plotting);
    }
  }
}

