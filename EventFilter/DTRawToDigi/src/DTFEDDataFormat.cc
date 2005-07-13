/**  
 *  See header file for a description of this class.
 * 
 *
 *
 *  $Date: 2005/07/06 15:52:01 $
 *  $Revision: 1.1 $
 *  \author G. Bruno  - CERN, EP Division
 */
#include "DTFEDDataFormat.h"
#include <cmath>

const int DTFEDDataFormat::NFields=7;
const int DTFEDDataFormat::BitMap[7]={6,9,11,14,31,51,63};

int DTFEDDataFormat::getNumberOfFields(){
  return NFields;
}  

int DTFEDDataFormat::getFieldLastBit(int i){

  return BitMap[i];

}

int DTFEDDataFormat::getSizeInBytes(int nobj){

  return (nobj) ? (int) ceil((double)(nobj*(DTFEDDataFormat::getFieldLastBit(DTFEDDataFormat::getNumberOfFields()-1)+1))/8) : 0;

}



