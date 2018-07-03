#ifndef BlobComplex_h
#define BlobComplex_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <utility>

struct BlobComplexData {
  BlobComplexData(){
    a = 0;
    b = 0;
  }


  void fill(unsigned int &serial);
  void print() const;
  bool operator == (const BlobComplexData &rhs) const;
  bool operator != (const BlobComplexData &rhs) const
  { return !(*this == rhs); }

  unsigned int a, b;
  std::vector<unsigned int> values;

  COND_SERIALIZABLE;
};

struct BlobComplexContent {
  BlobComplexContent() {}

  void fill(unsigned int &serial);
  void print() const;
  bool operator == (const BlobComplexContent &rhs) const;
  bool operator != (const BlobComplexContent &rhs) const
  { return !(*this == rhs); }

  typedef std::pair<BlobComplexData, unsigned int> Data;

  Data data1;
  Data data2;
  Data data3;

  COND_SERIALIZABLE;
};

struct BlobComplexObjects {
  BlobComplexObjects() {
    a = 0;
    b = 0;
  }

  void fill(unsigned int &serial);
  void print() const;
  bool operator == (const BlobComplexObjects &rhs) const;
  bool operator != (const BlobComplexObjects &rhs) const
  { return !(*this == rhs); }

  unsigned int a, b;
  std::vector<BlobComplexContent> content;

  COND_SERIALIZABLE;
};

struct BlobComplex {
  BlobComplex() {}

  void fill(unsigned int &serial);
  void print() const;
  bool operator == (const BlobComplex &rhs) const;
  bool operator != (const BlobComplex &rhs) const
  { return !(*this == rhs); }

  std::vector<BlobComplexObjects> objects;

  COND_SERIALIZABLE;
};

#endif
