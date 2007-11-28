/** \file MEtoROOTFormat.cc
 *  
 *  See header file for description of classe
 *  
 *
 *  $Date: 2007/10/04 19:26:49 $
 *  $Revision: 1.1 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "DataFormats/Histograms/interface/MEtoROOTFormat.h"

void MEtoROOT::putRootObject(std::string dirpath)
{
  test.dirpath = dirpath;
}
