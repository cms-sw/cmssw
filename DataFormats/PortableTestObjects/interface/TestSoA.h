#ifndef DataFormats_PortableTestObjects_interface_TestSoA_h
#define DataFormats_PortableTestObjects_interface_TestSoA_h

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/typedefs.h"

namespace portabletest {

  // SoA layout with x, y, z, id fields
  class TestSoA {
  public:
    static constexpr size_t alignment = 128;  // align all fields to 128 bytes

    // constructor
    TestSoA() : size_(0), buffer_(nullptr), x_(nullptr), y_(nullptr), z_(nullptr), id_(nullptr) {
      edm::LogVerbatim("TestSoA") << "TestSoA default constructor";
    }

    TestSoA(int32_t size, void *buffer)
        : size_(size),
          buffer_(buffer),
          x_(reinterpret_cast<double *>(reinterpret_cast<intptr_t>(buffer_))),
          y_(reinterpret_cast<double *>(reinterpret_cast<intptr_t>(x_) + pad(size * sizeof(double)))),
          z_(reinterpret_cast<double *>(reinterpret_cast<intptr_t>(y_) + pad(size * sizeof(double)))),
          id_(reinterpret_cast<int32_t *>(reinterpret_cast<intptr_t>(z_) + pad(size * sizeof(double)))) {
      assert(size == 0 or (size > 0 and buffer != nullptr));
      edm::LogVerbatim("TestSoA") << "TestSoA constructor with " << size_ << " elements at 0x" << buffer_;
    }

    ~TestSoA() {
      // the default implementation would work correctly, but we want to add a call to the MessageLogger
      if (buffer_) {
        edm::LogVerbatim("TestSoA") << "TestSoA destructor with " << size_ << " elements at 0x" << buffer_;
      } else {
        edm::LogVerbatim("TestSoA") << "TestSoA destructor wihout data";
      }
    }

    // non-copyable
    TestSoA(TestSoA const &) = delete;
    TestSoA &operator=(TestSoA const &) = delete;

    // movable
    TestSoA(TestSoA &&other)
        : size_(other.size_), buffer_(other.buffer_), x_(other.x_), y_(other.y_), z_(other.z_), id_(other.id_) {
      // the default implementation would work correctly, but we want to add a call to the MessageLogger
      edm::LogVerbatim("TestSoA") << "TestSoA move constructor with " << size_ << " elements at 0x" << buffer_;
      other.buffer_ = nullptr;
    }

    TestSoA &operator=(TestSoA &&other) {
      // the default implementation would work correctly, but we want to add a call to the MessageLogger
      size_ = other.size_;
      buffer_ = other.buffer_;
      x_ = other.x_;
      y_ = other.y_;
      z_ = other.z_;
      id_ = other.id_;
      edm::LogVerbatim("TestSoA") << "TestSoA move assignment with " << size_ << " elements at 0x" << buffer_;
      other.buffer_ = nullptr;
      return *this;
    }

    // global accessors
    int32_t size() const { return size_; }

    uint32_t extent() const { return compute_size(size_); }

    void *data() { return buffer_; }
    void const *data() const { return buffer_; }

    // element-wise accessors are not implemented for simplicity

    // field-wise accessors
    double const &x(int32_t i) const {
      assert(i >= 0);
      assert(i < size_);
      return x_[i];
    }

    double &x(int32_t i) {
      assert(i >= 0);
      assert(i < size_);
      return x_[i];
    }

    double const &y(int32_t i) const {
      assert(i >= 0);
      assert(i < size_);
      return y_[i];
    }

    double &y(int32_t i) {
      assert(i >= 0);
      assert(i < size_);
      return y_[i];
    }

    double const &z(int32_t i) const {
      assert(i >= 0);
      assert(i < size_);
      return z_[i];
    }

    double &z(int32_t i) {
      assert(i >= 0);
      assert(i < size_);
      return z_[i];
    }

    int32_t const &id(int32_t i) const {
      assert(i >= 0);
      assert(i < size_);
      return id_[i];
    }

    int32_t &id(int32_t i) {
      assert(i >= 0);
      assert(i < size_);
      return id_[i];
    }

    // pad a size (in bytes) to the next multiple of the alignment
    static constexpr uint32_t pad(size_t size) { return ((size + alignment - 1) / alignment * alignment); }

    // takes the size in elements, returns the size in bytes
    static constexpr uint32_t compute_size(int32_t elements) {
      assert(elements >= 0);
      return pad(elements * sizeof(double)) +  // x
             pad(elements * sizeof(double)) +  // y
             pad(elements * sizeof(double)) +  // z
             elements * sizeof(int32_t);       // id - no need to pad the last field
    }

    void ROOTReadStreamer(TestSoA const &onfile) {
      auto size = onfile.size();
      memcpy(x_, &onfile.x(0), size * sizeof(*x_));
      memcpy(y_, &onfile.y(0), size * sizeof(*y_));
      memcpy(z_, &onfile.z(0), size * sizeof(*z_));
      memcpy(id_, &onfile.id(0), size * sizeof(*id_));
    }

  private:
    // non-owned memory
    cms_int32_t size_;  // must be the same as ROOT's Int_t
    void *buffer_;      //!

    // layout
    double *x_;    //[size_]
    double *y_;    //[size_]
    double *z_;    //[size_]
    int32_t *id_;  //[size_]
  };

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_TestSoA_h
