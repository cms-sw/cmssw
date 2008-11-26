#ifndef GUARD_HCALCALOFLAGLABELS_H
#define GUARD_HCALCALOFLAGLABELS_H

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include <string>

// Create alias names for all status bits
// These aliases are valid for only the _current release_
// Use the HcalCaloFlagTool (below) for full interpretation
namespace HcalCaloFlagLabels
{
  enum HBHEStatusFlag{HBHEBit=0};
  enum HOStatusFlag{HOBit=0};
  enum HFStatusFlag{HFDigiTime=0,
		    HFLongShort=1};
  enum ZDCStatusFlag{ZDCBit=0};
  enum CalibrationFlag{CalibrationBit=0};
}

/** \brief Provides interpretation of flag bits with understanding of 
    CMSSW version dependence.
*/
class HcalCaloFlagTool {
public:
  HcalCaloFlagTool(const std::string& releaseName);
  std::string getFieldName(HcalSubdetector sd, int bit) const;
  int getFieldWidth(HcalSubdetector sd, int bit) const;
  int getFieldStart(HcalSubdetector sd, const std::string& field) const;
  int getFieldWidth(HcalSubdetector sd, const std::string& field) const;
  bool hasField(HcalSubdetector sd, const std::string& field) const;
  bool hasField(HcalSubdetector sd, int bit) const;
private:
  std::string releaseName_;
  bool standardFormat_;
  int major_, minor_, patch_, subpatch_;
};

#endif
