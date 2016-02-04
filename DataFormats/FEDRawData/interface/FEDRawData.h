#ifndef FEDRawData_FEDRawData_h
#define FEDRawData_FEDRawData_h

/** \class FEDRawData
 *
 *  Class representing the raw data for one FED.
 *  The raw data is owned as a binary buffer. It is required that the 
 *  lenght of the data is a multiple of the S-Link64 word lenght (8 byte).
 *  The FED data should include the standard FED header and trailer.
 *
 *  $Date: 2011/03/12 14:22:47 $
 *  $Revision: 1.7 $
 *  \author G. Bruno - CERN, EP Division
 *  \author S. Argiro - CERN and INFN - 
 *                      Refactoring and Modifications to fit into CMSSW
 */   


#include <vector>
#include <cstddef>

class FEDRawData {

 public:
  typedef std::vector<unsigned char> Data;
  typedef Data::iterator iterator;

  /// Default ctor
  FEDRawData();

  /// Ctor specifying the size to be preallocated, in bytes. 
  /// It is required that the size is a multiple of the size of a FED
  /// word (8 bytes)
  FEDRawData(size_t newsize);

  /// Copy constructor
  FEDRawData(const FEDRawData &);

  /// Dtor
  ~FEDRawData();

  /// Return a const pointer to the beginning of the data buffer
  const unsigned char * data() const;

  /// Return a pointer to the beginning of the data buffer
  unsigned char * data();

  /// Lenght of the data buffer in bytes
  size_t size() const {return data_.size();}
    
  /// Resize to the specified size in bytes. It is required that 
  /// the size is a multiple of the size of a FED word (8 bytes)
  void resize(size_t newsize);

 private:


  Data data_;

};

#endif
