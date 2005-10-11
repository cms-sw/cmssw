/** \file
 *
 *  $Date: 2005/07/13 09:06:50 $
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



