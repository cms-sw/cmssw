#ifndef HtrXmlPatternTool_h_included
#define HtrXmlPatternTool_h_included 1

#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "HtrXmlPatternSet.h"
#include "HtrXmlPatternToolParameters.h"
#include "HtrXmlPatternWriter.h"

class HtrXmlPatternTool {
public:
  HtrXmlPatternTool(HtrXmlPatternToolParameters* m_params);
  ~HtrXmlPatternTool();
  void Fill(const HcalElectronicsId HEID, HBHEDigiCollection::const_iterator data);
  void Fill(const HcalElectronicsId HEID, HFDigiCollection::const_iterator data);
  void Fill(const HcalElectronicsId HEID, HODigiCollection::const_iterator data);
  void prepareDirs();
  void createHists();
  void writeXML();
  HtrXmlPatternSet* GetPatternSet() { return m_patternSet; }

private:
  HtrXmlPatternSet* m_patternSet;
  HtrXmlPatternToolParameters* m_params;
  HtrXmlPatternWriter m_xmlWriter;
};

#endif
