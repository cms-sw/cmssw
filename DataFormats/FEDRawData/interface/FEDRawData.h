#ifndef _raw_FEDRawData_H
#define _raw_FEDRawData_H
/** \class DaqFEDRawData
 *
 *  Here is where the actual detector raw data  
 *  corresponding to a given FED is stored. 
 *
 *
 *  $Date: 2005/09/30 12:35:36 $
 *  $Revision: 1.3 $
 *  \author G. Bruno - CERN, EP Division
 *  \author S. Argiro - CERN and INFN - 
 *                      Refactoring and Modifications to fit into CMSSW
 */   

#include <vector>

class FEDRawData {

 public:
  typedef std::vector<unsigned char> Data;
  typedef Data::iterator iterator;

  FEDRawData();
  FEDRawData(size_t size);

  const unsigned char * data() const;
  unsigned char * data();

  size_t size() const {return data_.size();}
    
  /// Resize to the specified size in bytes. It is required that 
  /// the size is a multiple of the size of a FED word (8 bytes)
  void resize(size_t size);

 private:
  Data data_;

};

#endif
