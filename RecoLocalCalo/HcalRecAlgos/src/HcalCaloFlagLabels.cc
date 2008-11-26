#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include <string.h>

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
}

int HcalCaloFlagTool::getFieldWidth(HcalSubdetector sd, const std::string& field) const {
}

bool HcalCaloFlagTool::hasField(HcalSubdetector sd, const std::string& field) const {
  if (standardFormat_) {
    if (major_<3) return false;
  } 
  return false;
}

bool HcalCaloFlagTool::hasField(HcalSubdetector sd, int bit) const {
  if (standardFormat_) {
    if (major_<3) return false;
  } 
  return false;
}
