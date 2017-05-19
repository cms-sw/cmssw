#include "MultiHistoOverlapAll_Base.C"

void MultiHistoOverlapAll_Y1S(string files, string labels, string colors = "", string linestyles = "", string markerstyles = "", TString directory = ".", bool switchONfit = false){
  MultiHistoOverlapAll_Base(files, labels, colors, linestyles, markerstyles, directory, "Y1S", switchONfit);
}
