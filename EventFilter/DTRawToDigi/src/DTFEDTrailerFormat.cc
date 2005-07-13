/**  
 *  See header file for a description of this class.
 * 
 *
 *
 *  $Date: 2005/07/06 15:52:01 $
 *  $Revision: 1.1 $
 *  \author G. Bruno  - CERN, EP Division
 */
#include "DTFEDTrailerFormat.h"
#include <cmath>

const int DTFEDTrailerFormat::NFields=1;
const int DTFEDTrailerFormat::BitMap[1]={31};

int DTFEDTrailerFormat::getNumberOfFields(){
  return NFields;
}  

int DTFEDTrailerFormat::getFieldLastBit(int i){

  return BitMap[i];

}

int DTFEDTrailerFormat::getSizeInBytes(int nobj){

  return (nobj) ? (int) ceil((double)(nobj*(DTFEDTrailerFormat::getFieldLastBit(DTFEDTrailerFormat::getNumberOfFields()-1)+1))/8) : 0;

}



