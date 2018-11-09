#include "MultiHistoOverlapAll_Base.C"

void MultiHistoOverlapAll_Y1S(string files, string labels, string colors = "", string linestyles = "", string markerstyles = "", TString directory = ".", bool switchONfit = false, bool AutoSetRange=false, float CustomMinY=9.3, float CustomMaxY=9.6){//DEFAULT RANGE TO BE FIXED
  MultiHistoOverlapAll_Base(files, labels, colors, linestyles, markerstyles, directory, "Y1S", switchONfit, AutoSetRange, CustomMinY, CustomMaxY);
}
