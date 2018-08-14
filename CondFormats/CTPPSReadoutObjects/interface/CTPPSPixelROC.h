#ifndef CTPPSReadoutObjects_PixelROC_H
#define CTPPSReadoutObjects_PixelROC_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelIndices.h"

#include <cstdint>
#include <string>

/** \class PixelROC
 * Represents ReadOut Chip of DetUnit. 
 * Converts pixel coordinates from Local (in ROC) to Global (in DetUnit).
 * The Local coordinates are double column (dcol) and pixel index in dcol.
 * The Global coordinates are row and column in DetUnit.
 */


class CTPPSPixelROC {
public:

CTPPSPixelROC() : theDetUnit(0), theIdDU(0), theIdLk(0) {
  }
  
  ~CTPPSPixelROC(){
  }
  
  /// ctor with DetUnit id, 
  /// ROC number in DU (given by token passage), 
  /// ROC number in Link (given by token passage),
  CTPPSPixelROC( uint32_t du, int idInDU, int idLk);

  /// return the DetUnit to which this ROC belongs to.
  uint32_t rawId() const { return theDetUnit; }
  
  /// id of this ROC in DetUnit etermined by token path 
  unsigned int idInDetUnit() const { return theIdDU; }

  /// id of this ROC in parent Link.
  unsigned int idInLink() const { return theIdLk; }

  std::pair<int,int>  toLocal(const std::pair<int,int> &modulePixel ) const {
 
    int rocPixelRow, rocPixelColumn, idDU;
    int modulePixelRow = modulePixel.first; 
    int modulePixelColumn = modulePixel.second;

    theIndices.transformToROC(modulePixelColumn, modulePixelRow, idDU, rocPixelColumn, rocPixelRow);

    std::pair<int,int> rocPixel;
    rocPixel = std::make_pair(rocPixelRow, rocPixelColumn);

    return rocPixel;

  }

  std::pair<int,int>  toGlobal(const std::pair<int,int> &rocPixel ) const {
 
    int modulePixelRow=-1;
    int modulePixelColumn=-1;
    int rocPixelRow = rocPixel.first; 
    int rocPixelColumn = rocPixel.second;

    theIndices.transformToModule(rocPixelColumn, rocPixelRow, theIdDU, modulePixelColumn, modulePixelRow);

    std::pair<int,int> modulePixel;
    modulePixel = std::make_pair(modulePixelRow, modulePixelColumn);
    
    return modulePixel;

  }

  std::pair<int,int>  toGlobalfromDcol(const std::pair<int,int> &rocPixel ) const {
 
    int modulePixelRow=-1;
    int modulePixelColumn=-1;
    int rocDcol = rocPixel.first; 
    int rocPxl = rocPixel.second;

    int rocPixelRow ; 
    int rocPixelColumn ;

    theIndices.convertDcolToCol(rocDcol, rocPxl, rocPixelColumn, rocPixelRow);

    theIndices.transformToModule(rocPixelColumn, rocPixelRow, theIdDU, modulePixelColumn, modulePixelRow);

    std::pair<int,int> modulePixel;
    modulePixel = std::make_pair(modulePixelRow, modulePixelColumn);

    return modulePixel;

  }





  /// printout for debug
  std::string print(int depth = 0) const;


private:
  uint32_t theDetUnit;
  unsigned int theIdDU, theIdLk;

  CTPPSPixelIndices theIndices;

  COND_SERIALIZABLE;
};



#endif
