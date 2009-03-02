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
  
  LutXml * _xml = new LutXml(in_);
  _xml->create_lut_map();
  _xml->test_access("noname");
  delete _xml;
  XMLProcessor::getInstance()->terminate();

  return 0;
}
