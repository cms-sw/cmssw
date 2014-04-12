#ifndef BlobComplex_h
#define BlobComplex_h

#include <vector>
#include <utility>

struct BlobComplexData {
  BlobComplexData() {}

  void fill(unsigned int &serial);
  void print() const;
  bool operator == (const BlobComplexData &rhs) const;
  bool operator != (const BlobComplexData &rhs) const
  { return !(*this == rhs); }

  unsigned int a, b;
  std::vector<unsigned int> values;
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
};

struct BlobComplexObjects {
  BlobComplexObjects() {}

  void fill(unsigned int &serial);
  void print() const;
  bool operator == (const BlobComplexObjects &rhs) const;
  bool operator != (const BlobComplexObjects &rhs) const
  { return !(*this == rhs); }

  unsigned int a, b;
  std::vector<BlobComplexContent> content;
};

struct BlobComplex {
  BlobComplex() {}

  void fill(unsigned int &serial);
  void print() const;
  bool operator == (const BlobComplex &rhs) const;
  bool operator != (const BlobComplex &rhs) const
  { return !(*this == rhs); }

  std::vector<BlobComplexObjects> objects;
};

#endif
