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

  SiPixelCalibConfiguration() {;}
  SiPixelCalibConfiguration(const pos::PixelCalibConfiguration &fancyConfig);

  virtual ~SiPixelCalibConfiguration(){;}

 //-- Setter/Getter
  
  short  getNTriggers() const { return fNTriggers;}
  void  setNTriggers(const  short & in) { fNTriggers = in; }

  std::vector<short> getRowPattern() const { return fRowPattern;}
  void  setRowPattern(const std::vector<short> & in) { fRowPattern = in; }

  std::vector<short> getColumnPattern() const { return fColumnPattern;}
  void  setColumnPattern(const std::vector<short> & in) { fColumnPattern = in; }

  void setCalibrationMode(const std::string & in) { fMode = in; }
  std::string getCalibrationMode() const {return fMode;}

  const std::vector<short> getVCalValues() const { return fVCalValues;}
  void  setVCalValues(const std::vector< short> & in) { fVCalValues = in; }
  
  // interface with calibration analyzers:
  short vcalForEvent(const uint32_t & eventnumber) const;
  short vcalIndexForEvent(const uint32_t & eventnumber) const;
  std::vector<short> columnPatternForEvent(const uint32_t & eventnumber) const;
  std::vector<short> rowPatternForEvent(const uint32_t & eventnumber) const;
  uint32_t nextPatternChangeForEvent(const uint32_t & eventnumber) const;
  uint32_t expectedTotalEvents () const;
  uint32_t  patternSize() const {return fNTriggers*fVCalValues.size();}
  uint32_t nPatterns() const {return nRowPatterns()*nColumnPatterns();}
  uint32_t nColumnPatterns() const ;
  uint32_t nRowPatterns() const ;
  uint32_t nVCal() const { return fVCalValues.size();}
    
 private :

  short                     fNTriggers;//
  std::vector<short>        fRowPattern;//
  std::vector<short>        fColumnPattern;//
  std::vector<short>        fVCalValues;//
  std::string               fMode;
};

#endif

