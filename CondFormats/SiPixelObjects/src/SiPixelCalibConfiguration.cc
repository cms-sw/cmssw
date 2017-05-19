#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h"
#include <cstdlib>

short SiPixelCalibConfiguration::vcalIndexForEvent(const uint32_t & eventnumber) const{
  uint32_t relative_event = std::abs((int32_t)eventnumber-1)%patternSize();
  short relative_pattern = relative_event/getNTriggers();
  return relative_pattern; 
}
short SiPixelCalibConfiguration::vcalForEvent(const uint32_t & eventnumber) const{
  short result = fVCalValues[vcalIndexForEvent(eventnumber)];
  return result;
}
std::vector<short> SiPixelCalibConfiguration::columnPatternForEvent(const uint32_t & eventnumber) const{
  std::vector<short> result;
  uint32_t patternnumber = eventnumber/patternSize();
  uint32_t colpatternnumber = patternnumber%nColumnPatterns();

  uint32_t nminuscol=0;
  for(size_t icol=0; icol<fColumnPattern.size(); icol++){
    if(fColumnPattern[icol]==-1)
      nminuscol++;
    else if(nminuscol>colpatternnumber)
      break;
    else if(nminuscol==colpatternnumber){
      short val = fColumnPattern[icol];
      result.push_back(val);
    }
  }
  return result;
}

std::vector<short> SiPixelCalibConfiguration::rowPatternForEvent(const uint32_t & eventnumber) const {
  std::vector<short> result;
  uint32_t patternnumber = eventnumber/patternSize();
  uint32_t rowpatternnumber = patternnumber/nColumnPatterns();

  uint32_t nminusrow=0;
  for(size_t irow=0; irow<fRowPattern.size(); irow++){
    if(fRowPattern[irow]==-1)
      nminusrow++;
    else if(nminusrow>rowpatternnumber)
      break;
    else if(nminusrow==rowpatternnumber){
      short val = fRowPattern[irow];
      result.push_back(val);
    }
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
  fRowPattern(std::vector<short>(0)),
  fColumnPattern(std::vector<short>(0)),
  fVCalValues(std::vector<short>(0)),
  fMode("unknown")

{ // copy constructor that uses the complex object
  fNTriggers = fancyConfig.nTriggersPerPattern();
  std::vector<int> vcalpoints(0);
  std::cout << "scan name = " << fancyConfig.scanName(0) << std::endl;
  std::vector<uint32_t> vcalpointsuint32 = fancyConfig.scanValues(fancyConfig.scanName(0));
  for(size_t ical=0; ical<vcalpointsuint32.size(); ++ical){
    short vcalinput = vcalpointsuint32[ical];
  
    std::cout << "Vcal value " << ical << " = " << vcalinput << std::endl;
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
  fMode = fancyConfig.mode();
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

