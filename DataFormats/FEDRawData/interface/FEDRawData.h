#ifndef _raw_FEDRawData_H
#define _raw_FEDRawData_H
/** \class DaqFEDRawData
 *
 *  Here is where the actual detector raw data  
 *  corresponding to a given FED is stored. 
 *
 *
 *  $Date: 2004/07/05 08:37:05 $
 *  $Revision: 1.4 $
 *  \author G. Bruno - CERN, EP Division
 *  \author S. Argiro - CERN and INFN - 
 *                      Refactoring and Modifications to fit into CMSSW
 */   

#include <vector>

namespace raw{

  struct FEDRawData {

    typedef std::vector<unsigned char> Data;
    typedef Data::iterator iterator;

    FEDRawData();
    FEDRawData(size_t size);

    const unsigned char * data() const;

    Data data_;

  };
}
#endif
