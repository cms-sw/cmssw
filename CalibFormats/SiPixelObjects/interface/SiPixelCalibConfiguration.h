//---------------------------------------------------

// Author : Freya.Blekman@cern.ch
// Name   : SiPixelCalibConfiguration

//---------------------------------------------------

#ifndef SiPixelCalibConfiguration_H
#define SiPixelCalibConfiguration_H

#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include "CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h"

class SiPixelCalibConfiguration
{
  
 public :

  struct RowPatternStruct{
     short first;
     short second;
  };

  struct ColPatternStruct{
     short first;
     short second;
  };

  SiPixelCalibConfiguration() {;}
  SiPixelCalibConfiguration(const PixelCalibConfiguration &fancyConfig);

  virtual ~SiPixelCalibConfiguration(){;}

 //-- Setter/Getter
  
  short  NTriggers() const { return fNTriggers;}
  void  setNTriggers(const  short & in) { fNTriggers = in; }

  std::vector<std::string> ROCIds() const { return fROCIds;}
  void  setROCIds(const std::vector<std::string> & in) { fROCIds = in; }

  std::vector<RowPatternStruct > RowPattern() const { return fRowPattern;}
  void  setRowPattern(const std::vector<RowPatternStruct> & in) { fRowPattern = in; }

  std::vector<ColPatternStruct> ColumnPattern() const { return fColumnPattern;}
  void  setColumnPattern(const std::vector<ColPatternStruct> & in) { fColumnPattern = in; }

  const std::vector<short> VCalValues() const { return fVCalValues;}
  void  setVCalValues(const std::vector< short> & in) { fVCalValues = in; }
  
  // interface with calibration analyzers:
  
  short vcalForEvent(uint32_t eventnumber);
  short vcalIndexForEvent(uint32_t eventnumber);
  ColPatternStruct columnPatternForEvent(uint32_t eventnumber);
  RowPatternStruct rowPatternForEvent(uint32_t eventnumber);
  uint32_t nextPatternChangeForEvent(uint32_t eventnumber);
  uint32_t expectedTotalEvents();
  uint32_t patternSize() {return fNTriggers*fVCalValues.size();}
  uint32_t nPatterns() {return fColumnPattern.size()*fRowPattern.size();}

 private :

  short                             fNTriggers;//
  std::vector<std::string>             fROCIds;// can be replaced by  shorts I suppose....
  std::vector<RowPatternStruct>        fRowPattern;//
  std::vector<ColPatternStruct>        fColumnPattern;//
  std::vector<short>                fVCalValues;//
};

#endif

