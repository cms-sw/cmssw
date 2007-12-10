
#include "CalibFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"

short SiPixelCalibConfiguration::vcalIndexForEvent(const uint32_t & eventnumber) const{
  uint32_t relative_event = eventnumber%patternSize();
  short relative_pattern = relative_event/getNTriggers();
  return relative_pattern; 
}
short SiPixelCalibConfiguration::vcalForEvent(const uint32_t & eventnumber) const{
  short result = fVCalValues[vcalIndexForEvent(eventnumber)];
  return result;
}
std::vector<short> SiPixelCalibConfiguration::columnPatternForEvent(const uint32_t & eventnumber) const{
  uint32_t relative_event = eventnumber/patternSize(); //  ASSUMES LOOP OF ROWS OUTSIDE LOOP OF COLUMNS

  std::cout <<"columnPatternForEvent relative event is: " << relative_event << std::endl;
  uint32_t patterncounter=0;
  std::vector<short> result(0);
  for(uint32_t i=0; i<fColumnPattern.size()-1; ++i){
    short val = fColumnPattern[i];
    if(val==-1){
      patterncounter++;
      continue;
    }
    if(patterncounter!=relative_event)
      continue;
    result.push_back(val);
  }
  return result;
}

std::vector<short> SiPixelCalibConfiguration::rowPatternForEvent(const uint32_t & eventnumber) const {
  uint32_t relative_event = eventnumber%patternSize();// ASSUMES LOOP OF ROWS OUTSIDE LOOP OF COLUMNS  
  relative_event/=nColumnPatterns();
  uint32_t patterncounter=0;
  std::cout <<"rowPatternForEvent relative event is: " << relative_event << std::endl;
  std::vector<short> result(0);
  for(uint32_t i=0;i<fRowPattern.size();++i){
    short val = fRowPattern[i];
    if(val==-1){
      patterncounter++;
      continue;
    }
    if(patterncounter!=relative_event)
      continue;
    result.push_back(val);
  }
  return result;
}
uint32_t SiPixelCalibConfiguration::nextPatternChangeForEvent(const uint32_t & eventnumber) const {
  uint32_t relative_event = eventnumber/patternSize();
  relative_event+=1;
  return relative_event*patternSize();
}
uint32_t SiPixelCalibConfiguration::expectedTotalEvents() const {
  return patternSize()*nPatterns();
}

SiPixelCalibConfiguration::SiPixelCalibConfiguration(const pos::PixelCalibConfiguration &fancyConfig):
  fNTriggers(0),
  fROCIds(std::vector<std::string>(0)),
  fRowPattern(std::vector<short>(0)),
  fColumnPattern(std::vector<short>(0)),
  fVCalValues(std::vector<short>(0)),
  fMode("unknown")

{ // copy constructor that uses the complex object
  fNTriggers = fancyConfig.nTriggersPerPattern();
  std::vector<int> vcalpoints(0);
  for(short vcalvalue = (short) fancyConfig.scanValueMin(0); vcalvalue<=fancyConfig.scanValueMax(0); vcalvalue+=(short)fancyConfig.scanValueStep(0)){
    short vcalinput = vcalvalue;
    fVCalValues.push_back(vcalinput);
  }
  // copy row and column patterns

  std::vector<std::vector<uint32_t> > cols=fancyConfig.columnList();
  std::vector<std::vector<uint32_t> > rows= fancyConfig.rowList();
  for(uint32_t i=0; i<cols.size(); ++i){
    for(uint32_t j=0; j<cols[i].size(); ++j){
      short colval = cols[i][j];
      fColumnPattern.push_back(colval);
    }
    fColumnPattern.push_back(-1);
  }
  for(uint32_t i=0; i<rows.size(); ++i){
    for(uint32_t j=0; j<rows[i].size(); ++j){
      short rowval = rows[i][j];
      fRowPattern.push_back(rowval);
    }
    fRowPattern.push_back(-1);
  }
  // copy roc names, disabled as it does not work with POS243.

//     std::vector<pos::PixelROCName> rocnames = fancyConfig.rocList();
//     for(std::vector<pos::PixelROCName>::iterator iname = rocnames.begin(); iname!=rocnames.end(); ++iname){
//       std::string tempname=iname->rocname();
//       fROCIds.push_back(tempname);
//     }

}

uint32_t SiPixelCalibConfiguration::nRowPatterns() const{
 uint32_t nrows = 0;
 for(std::vector<short>::const_iterator i=fRowPattern.begin();i!=fRowPattern.end();++i){
   if(*i == -1)
    nrows++;
  }
  return nrows;
}
uint32_t SiPixelCalibConfiguration::nColumnPatterns() const{
  uint32_t ncols = 0;
  
  for(std::vector<short>::const_iterator i=fColumnPattern.begin();i!=fColumnPattern.end();++i){
    if(*i == -1)
    ncols++;
  }
  return ncols;
}

