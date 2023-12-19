#ifndef FEDRawData_FEDRawData_h
#define FEDRawData_FEDRawData_h

/** \class FEDRawData
 *
 *  Class representing the raw data for one FED.
 *  The raw data is owned as a binary buffer. It is required that the 
 *  lenght of the data is a multiple of the S-Link64 word lenght (8 byte).
 *  The FED data should include the standard FED header and trailer.
 *
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
  /// word (8 bytes default)
  FEDRawData(size_t newsize, size_t wordsize = 8);

  /// Copy constructor
  FEDRawData(const FEDRawData &);

  /// Assignment operator
  FEDRawData &operator=(const FEDRawData &) = default;

  /// Dtor
  ~FEDRawData();

  /// Return a const pointer to the beginning of the data buffer
  const unsigned char *data() const;

  /// Return a pointer to the beginning of the data buffer
  unsigned char *data();

  /// Lenght of the data buffer in bytes
  size_t size() const { return data_.size(); }

  /// Resize to the specified size in bytes. It is required that
  /// the size is a multiple of the size of a FED word (8 bytes default)
  void resize(size_t newsize, size_t wordsize = 8);

private:
  Data data_;
};

#endif
