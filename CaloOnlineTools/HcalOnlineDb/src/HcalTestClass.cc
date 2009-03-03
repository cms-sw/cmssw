// for testing includes

#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LutXml.h"

class HcalTestClass{
public:
  HcalTestClass(){};
  ~HcalTestClass(){};
  int test(void);
};

int HcalTestClass::test(void){
  std::string in_="file";
  uint32_t det_id;
  
  LutXml * _xml = new LutXml(in_);
  _xml->create_lut_map();
  std::vector<unsigned int> * l = _xml->getLutFast(det_id);
  delete _xml;
  XMLProcessor::getInstance()->terminate();

  return 0;
}
