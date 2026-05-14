#ifndef FEDRawData_FEDRawData_h
#define FEDRawData_FEDRawData_h

/** \class FEDRawData
 *
 *  Class representing the raw data for one FED.
 *  The raw data is owned as a binary buffer. It is required that the 
 *  length of the data is a multiple of the S-Link64 word length (8 byte).
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
  using Data = std::vector<unsigned char>;
  using value_type = unsigned char;
  using size_type = size_t;
  using reference = Data::reference;
  using const_reference = Data::const_reference;
  using pointer = Data::pointer;
  using const_pointer = Data::const_pointer;
  using iterator = Data::iterator;
  using const_iterator = Data::const_iterator;

  /// Default ctor
  FEDRawData() = default;

  /// Ctor specifying the size to be preallocated, in bytes.
  /// It is required that the size is a multiple of the size of a FED
  /// word (8 bytes default)
  FEDRawData(size_t newsize, size_t wordsize = 8);

  /// Copy constructor
  FEDRawData(const FEDRawData &) = default;

  /// Assignment operator
  FEDRawData &operator=(const FEDRawData &) = default;

  /// Move constructor
  FEDRawData(FEDRawData &&) = default;

  /// Move assignment operator
  FEDRawData &operator=(FEDRawData &&) = default;

  /// Dtor
  ~FEDRawData() = default;

  /// Return a const pointer to the beginning of the data buffer
  const unsigned char *data() const noexcept { return data_.data(); }

  /// Return a pointer to the beginning of the data buffer
  unsigned char *data() noexcept { return data_.data(); }

  /// Length of the data buffer in bytes
  size_t size() const noexcept { return data_.size(); }

  auto begin() const noexcept { return data_.begin(); }
  auto end() const noexcept { return data_.end(); }

  auto begin() noexcept { return data_.begin(); }
  auto end() noexcept { return data_.end(); }

  auto cbegin() noexcept { return data_.cbegin(); }
  auto cend() noexcept { return data_.cend(); }

  /// Resize to the specified size in bytes. It is required that
  /// the size is a multiple of the size of a FED word (8 bytes default)
  void resize(size_t newsize, size_t wordsize = 8);

private:
  Data data_;
};

#endif
