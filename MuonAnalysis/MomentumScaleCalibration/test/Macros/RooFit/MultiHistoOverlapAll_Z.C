#include "MultiHistoOverlapAll_Base.C"

void MultiHistoOverlapAll_Z(string files, string labels, string colors = "", string linestyles = "", string markerstyles = "", TString directory = ".", bool switchONfit = false, bool AutoSetRange=false, float CustomMinY=90.85, float CustomMaxY=91.4){
  MultiHistoOverlapAll_Base(files, labels, colors, linestyles, markerstyles, directory, "Z", switchONfit, AutoSetRange, CustomMinY, CustomMaxY);
}
