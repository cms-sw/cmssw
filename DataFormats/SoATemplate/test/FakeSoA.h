#ifndef CUDADataFormatsCommonFakeSoA_H
#define CUDADataFormatsCommonFakeSoA_H
// A SoA-like class (with fake alignment (padding))
#include <cstddef>
#include <iostream>

#define myassert(A)                                                                            \
  if (!(A)) {                                                                                  \
    std::cerr << "Failed assertion: " #A " at  " __FILE__ "(" << __LINE__ << ")" << std::endl; \
    abort();                                                                                   \
  }

class FakeSoA {
public:
  static constexpr size_t padding_ = 128;
  // A fake SoA with 2 columns of uint16_t and uin32_t, plus fake padding.
  static size_t computeBufferSize(size_t nElements) {
    return nElements * (sizeof(uint16_t) + sizeof(uint32_t)) + padding_ + 0x400;
  }
  FakeSoA(std::byte *buffer, size_t nElements) { ConstFromBufferImpl(buffer, nElements); }

private:
  void ConstFromBufferImpl(std::byte *buffer, size_t nElements) {
    buffer_ = buffer;
    size_ = nElements;
    a16_ = reinterpret_cast<uint16_t *>(buffer_);
    buffer += nElements * sizeof(uint16_t) + padding_;
    b32_ = reinterpret_cast<uint32_t *>(buffer);
    buffer += nElements * sizeof(uint32_t);
    std::cout << "Buffer first byte after (const) =" << buffer << std::endl;
    std::cout << "At end of FakeSoA::ConstFromBufferImpl(std::byte * buffer, size_t nElements): ";
    Dump();
  }

public:
  FakeSoA() : size_(0) { std::cout << "At end of FakeSoA::FakeSoA()" << std::endl; }

  template <typename T>
  void AllocateAndIoRead(T &onfile) {
    std::cout << "AllocateAndIoRead begin" << std::endl;
    auto buffSize = FakeSoA::computeBufferSize(onfile.size_);
    auto buffer = new std::byte[buffSize];
    std::cout << "Buffer first byte after (alloc) =" << buffer + buffSize << std::endl;
    ConstFromBufferImpl(buffer, onfile.size_);
    memcpy(a16_, onfile.a16_, sizeof(uint16_t) * onfile.size_);
    memcpy(b32_, onfile.b32_, sizeof(uint32_t) * onfile.size_);
    std::cout << "AllocateAndIoRead end" << std::endl;
  }

  void Dump() {
    std::cout << "size=" << size_ << " buffer=" << buffer_ << " a16=" << a16_ << " b32=" << b32_ << " (b32 - a16)="
              << reinterpret_cast<std::byte *>(reinterpret_cast<intptr_t>(b32_) - reinterpret_cast<intptr_t>(a16_))
              << " buffer size=" << computeBufferSize(size_) << "(" << std::hex << computeBufferSize(size_) << ")"
              << std::endl;
  }

  void DumpData() { std::cout << "a16_[0]=" << a16_[0] << " b32_[0]=" << b32_[0] << std::endl; }

  void Fill() {
    for (Int_t i = 0; i < size_; i++) {
      a16_[i] = 42 + i;
      b32_[i] = 24 + i;
    }
  }

  void Check() {
    for (Int_t i = 0; i < size_; i++) {
      if (a16_[i] != 42 + i) {
        std::cout << "a16 mismatch at i=" << i << "(" << a16_[i] << "/" << 42 + i << ")" << std::endl;
      }
      if (b32_[i] != 24 + (uint32_t)i) {
        std::cout << "b32 mismatch at i=" << i << "(" << b32_[i] << "/" << 24 + i << ")" << std::endl;
      }
    }
  }

  Int_t size_;
  uint16_t *a16_ = nullptr;      //[size_]
  uint32_t *b32_ = nullptr;      //[size_]
  std::byte *buffer_ = nullptr;  //!
};

#endif  //ndef CUDADataFormatsCommonFakeSoA_H