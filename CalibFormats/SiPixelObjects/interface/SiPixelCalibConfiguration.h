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
  
  short  NTriggers() const { return fNTriggers;}
  void  setNTriggers(const  short & in) { fNTriggers = in; }

  std::vector<std::string> ROCIds() const { return fROCIds;}
  void  setROCIds(const std::vector<std::string> & in) { fROCIds = in; }

  std::vector<short> RowPattern() const { return fRowPattern;}
  void  setRowPattern(const std::vector<short> & in) { fRowPattern = in; }

  std::vector<short> ColumnPattern() const { return fColumnPattern;}
  void  setColumnPattern(const std::vector<short> & in) { fColumnPattern = in; }

  const std::vector<short> VCalValues() const { return fVCalValues;}
  void  setVCalValues(const std::vector< short> & in) { fVCalValues = in; }
  
  // interface with calibration analyzers:
  // old access methods
  uint32_t nVcal() const {return fVCalValues.size();}
  uint32_t vcal_first() const {return (uint32_t) fVCalValues[0];}
  uint32_t vcal_last() const {return (uint32_t) fVCalValues[fVCalValues.size()-1];}
  uint32_t vcal_step() const {return (uint32_t) (vcal_last()- vcal_first())/nVcal();}
  uint32_t nTriggers() const {return (uint32_t) NTriggers();}
  // new access methods
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

 private :

  short                             fNTriggers;//
  std::vector<std::string>             fROCIds;// can be replaced by  shorts I suppose....
  std::vector<short>        fRowPattern;//
  std::vector<short>        fColumnPattern;//
  std::vector<short>                fVCalValues;//
};

#endif

