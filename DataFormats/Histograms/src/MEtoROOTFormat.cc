/** \file MEtoROOTFormat.cc
 *  
 *  See header file for description of class
 *  
 *
 *  $Date: 2007/12/02 03:46:24 $
 *  $Revision: 1.2 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "DataFormats/Histograms/interface/MEtoROOTFormat.h"
#include <iostream>
#include <stdlib.h>

void MEtoROOT::putMERootObject(std::vector<uint64_t> version,
			       std::vector<std::string> name,
			       std::vector<TagList> tags,
			       std::vector<TH1F> object,
			       std::vector<TH1F> refobj,
			       std::vector<QReports> qreports,
			       std::vector<uint32_t> flags)
{

  //std::cout << "Filling MERootObject" << std::endl;

  MERootObject.resize(name.size());
  for (unsigned int i = 0; i < name.size(); ++i) {
    MERootObject[i].version = version[i];
    MERootObject[i].name = name[i];
    MERootObject[i].tags = tags[i];
    MERootObject[i].object = object[i];
    MERootObject[i].refobj = refobj[i];
    MERootObject[i].qreports = qreports[i];
    MERootObject[i].flags = flags[i];
  }

  return;
}
