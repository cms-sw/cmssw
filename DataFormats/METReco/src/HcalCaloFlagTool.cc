#include "DataFormats/METReco/interface/HcalCaloFlagTool.h"
#include <string.h>
#include <cstdio>

HcalCaloFlagTool::HcalCaloFlagTool(const std::string& releaseName) : 
  releaseName_(releaseName),
  standardFormat_(false),
  major_(0),
  minor_(0),
  patch_(0),
  subpatch_(0)
{
  int fields=sscanf(releaseName.c_str(),"CMSSW_%d_%d_%d_%d",&major_,&minor_,&patch_,&subpatch_);
  if (fields>=3) standardFormat_=true;
}

std::string HcalCaloFlagTool::getFieldName(HcalSubdetector sd, int bit) const {
  return "";
}

int HcalCaloFlagTool::getFieldWidth(HcalSubdetector sd, int bit) const {
  return 0;
}

int HcalCaloFlagTool::getFieldStart(HcalSubdetector sd, const std::string& field) const {
  return 0;
}

int HcalCaloFlagTool::getFieldWidth(HcalSubdetector sd, const std::string& field) const {
  return 0;
}

bool HcalCaloFlagTool::hasField(HcalSubdetector sd, const std::string& field) const {
  if (standardFormat_) {
    if (major_<3) return false;
  } 
  return getFieldWidth(sd,field)>0;
}

bool HcalCaloFlagTool::hasField(HcalSubdetector sd, int bit) const {
  if (standardFormat_) {
    if (major_<3) return false;
  } 
  return getFieldWidth(sd,bit)>0;
}
