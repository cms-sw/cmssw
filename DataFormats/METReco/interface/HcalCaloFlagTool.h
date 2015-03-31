#ifndef DataFormats_METReco_HcalCaloFlagTool_h
#define DataFormats_METReco_HcalCaloFlagTool_h

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
#include <string>

// Use the HcalCaloFlagTool (below) for full interpretation of HcalCaloFlagLabels
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

#endif //DataFormats_METReco_HcalCaloFlagTool_h
