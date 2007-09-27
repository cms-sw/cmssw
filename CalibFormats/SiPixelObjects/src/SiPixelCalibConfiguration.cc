
#include "CalibFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"

short SiPixelCalibConfiguration::vcalIndexForEvent(uint32_t eventnumber){
  uint32_t relative_event = eventnumber%patternSize();
  short relative_pattern = relative_event/NTriggers();
  return relative_pattern; 
}
short SiPixelCalibConfiguration::vcalForEvent(uint32_t eventnumber){
  short result = fVCalValues[vcalIndexForEvent(eventnumber)];
  return result;
}
SiPixelCalibConfiguration::ColPatternStruct SiPixelCalibConfiguration::columnPatternForEvent(uint32_t eventnumber){
  uint32_t relative_event = eventnumber%patternSize(); // TO BE CHECKED: ASSUMES LOOP OF ROWS OUTSIDE LOOP OF COLUMNS
  relative_event/=fRowPattern.size();
  return fColumnPattern[relative_event];
}

SiPixelCalibConfiguration::RowPatternStruct SiPixelCalibConfiguration::rowPatternForEvent(uint32_t eventnumber){
  uint32_t relative_event = eventnumber/patternSize();// TO BE CHECKED: ASSUMES LOOP OF ROWS OUTSIDE LOOP OF COLUMNS
  return fRowPattern[relative_event];
}
uint32_t SiPixelCalibConfiguration::nextPatternChangeForEvent(uint32_t eventnumber){
  uint32_t relative_event = eventnumber/patternSize();
  relative_event+=1;
  return relative_event*patternSize();
}
uint32_t SiPixelCalibConfiguration::expectedTotalEvents(){
  return patternSize()*nPatterns();
}

SiPixelCalibConfiguration::SiPixelCalibConfiguration(const PixelCalibConfiguration &fancyConfig):
  fNTriggers(0),
  fVCalValues(std::vector<short>(0)),
  fRowPattern(std::vector<RowPatternStruct>(0)),
  fColumnPattern(std::vector<ColPatternStruct>(0)),
  fROCIds(std::vector<std::string>(0))
{ // copy constructor that uses the complex object
  fNTriggers = fancyConfig.nTriggersPerPattern();
  std::vector<int> vcalpoints(0);
  for(int vcalvalue = fancyConfig.scanValueMin(0); vcalvalue<=fancyConfig.scanValueMax(0); vcalvalue+=fancyConfig.scanValueStep(0)){
    int vcalinput = vcalvalue;
    fVCalValues.push_back(vcalinput);
  }
  // copy row and column patterns

  std::vector<std::vector<uint32_t> > cols=fancyConfig.columnList();
  std::vector<std::vector<uint32_t> > rows= fancyConfig.rowList();
  for(std::vector<std::vector<uint32_t> >::iterator icolpat=cols.begin(); icolpat!=cols.end(); ++icolpat){
    if(icolpat->size()>2)
      std::cout << "size of column pattern is: " << icolpat->size() << ", which is not compatible!!!" << std::endl;
    ColPatternStruct tempstruct;
    tempstruct.first=icolpat->at(0);
    tempstruct.second=icolpat->at(1);
    fColumnPattern.push_back(tempstruct);
  } 
  for(std::vector<std::vector<uint32_t> >::iterator irowpat=rows.begin(); irowpat!=rows.end(); ++irowpat){
    if(irowpat->size()!=2)
      std::cout << "size of row pattern is: " << irowpat->size() << ", which is not compatible!!!" << std::endl;
    RowPatternStruct tempstruct;
    tempstruct.first=irowpat->at(0);
    tempstruct.second=irowpat->at(1);
    fRowPattern.push_back(tempstruct);
  }
  // copy roc names
  std::vector<PixelROCName> rocnames = fancyConfig.rocList();
  for(std::vector<PixelROCName>::iterator iname = rocnames.begin(); iname!=rocnames.end(); ++iname){
    std::string tempname=iname->rocname();
    fROCIds.push_back(tempname);
  }
}

