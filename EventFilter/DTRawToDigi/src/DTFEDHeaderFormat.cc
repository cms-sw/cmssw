/**  
 *  See header file for a description of this class.
 * 
 *
 *
 *  $Date: 2005/07/06 15:52:01 $
 *  $Revision: 1.1 $
 *  \author G. Bruno  - CERN, EP Division
 */
#include "DTFEDHeaderFormat.h"
#include <cmath>

const int DTFEDHeaderFormat::NFields=2;
const int DTFEDHeaderFormat::BitMap[2]={31,63};

int DTFEDHeaderFormat::getNumberOfFields(){
  return NFields;
}  

int DTFEDHeaderFormat::getFieldLastBit(int i){

  return BitMap[i];

}

int DTFEDHeaderFormat::getSizeInBytes(int nobj){

  return (nobj) ? (int) ceil((double)(nobj*(DTFEDHeaderFormat::getFieldLastBit(DTFEDHeaderFormat::getNumberOfFields()-1)+1))/8) : 0;

}

