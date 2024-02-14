#ifndef DataFormats_TestObjects_SchemaEvolutionTestObjects_h
#define DataFormats_TestObjects_SchemaEvolutionTestObjects_h

#include <vector>

// Don't delete the following comment line.
// This #define is required when generating data files
// using the old formats. These data files are saved and used
// as input in unit tests to verify that ROOT can use schema
// evolution to read the old formats with a release that has
// modified formats. When reading this #define should be commented
// out and it should be commented out in the code repository.
// Note that the data files are generated manually and this
// line and classes_def.xml must be manually modified and built
// when generating new data files. The data files are saved
// in this repository: https://github.com/cms-data/IOPool-Input.
//#define DataFormats_TestObjects_USE_OLD
#if defined DataFormats_TestObjects_USE_OLD

#include <map>

#else

#include <array>
#include <list>
#include <memory>
#include <unordered_map>

#endif

namespace edmtest {

  class SchemaEvolutionChangeOrder {
  public:
#if defined DataFormats_TestObjects_USE_OLD
    SchemaEvolutionChangeOrder() : a_(0), b_(0) {}
    SchemaEvolutionChangeOrder(int a, int b) : a_(a), b_(b) {}
    int a_;
    int b_;
#else
    SchemaEvolutionChangeOrder() : b_(0), a_(0) {}
    SchemaEvolutionChangeOrder(int a, int b) : b_(b), a_(a) {}
    int b_;
    int a_;
#endif
  };

  class SchemaEvolutionAddMember {
  public:
#if defined DataFormats_TestObjects_USE_OLD
    SchemaEvolutionAddMember() : a_(0), b_(0) {}
    SchemaEvolutionAddMember(int a, int b, int) : a_(a), b_(b) {}
    int a_;
    int b_;
#else
    SchemaEvolutionAddMember() : a_(0), b_(0), c_(0) {}
    SchemaEvolutionAddMember(int a, int b, int c) : a_(a), b_(b), c_(c) {}
    int a_;
    int b_;
    int c_;
#endif
  };

  class SchemaEvolutionRemoveMember {
  public:
#if defined DataFormats_TestObjects_USE_OLD
    SchemaEvolutionRemoveMember() : a_(0), b_(0) {}
    SchemaEvolutionRemoveMember(int a, int b) : a_(a), b_(b) {}
    int a_;
    int b_;
#else
    SchemaEvolutionRemoveMember() : a_(0) {}
    SchemaEvolutionRemoveMember(int a, int) : a_(a) {}
    int a_;
#endif
  };

#if defined DataFormats_TestObjects_USE_OLD
  class SchemaEvolutionBase {
  public:
    SchemaEvolutionBase() : d_(0) {}
    SchemaEvolutionBase(int d) : d_(d) {}
    int d_;
  };

  class SchemaEvolutionMoveToBase : public SchemaEvolutionBase {
  public:
    SchemaEvolutionMoveToBase() : a_(0), b_(0), c_(0) {}
    SchemaEvolutionMoveToBase(int a, int b, int c, int d) : SchemaEvolutionBase(d), a_(a), b_(b), c_(c) {}
    int a_;
    int b_;
    int c_;
  };
#else
  class SchemaEvolutionBase {
  public:
    SchemaEvolutionBase() : c_(0), d_(0) {}
    SchemaEvolutionBase(int c, int d) : c_(c), d_(d) {}
    int c_;
    int d_;
  };

  class SchemaEvolutionMoveToBase : public SchemaEvolutionBase {
  public:
    SchemaEvolutionMoveToBase() : a_(0), b_(0) {}
    SchemaEvolutionMoveToBase(int a, int b, int c, int d) : SchemaEvolutionBase(c, d), a_(a), b_(b) {}
    int a_;
    int b_;
  };
#endif

  class SchemaEvolutionChangeType {
  public:
#if defined DataFormats_TestObjects_USE_OLD
    SchemaEvolutionChangeType() : a_(0), b_(0) {}
    SchemaEvolutionChangeType(int a, int b) : a_(a), b_(b) {}
    int a_;
    int b_;
#else
    SchemaEvolutionChangeType() : a_(0.0), b_(0LL) {}
    SchemaEvolutionChangeType(int a, int b) : a_(a), b_(b) {}
    double a_;
    long long b_;
#endif
  };

  class SchemaEvolutionBaseA {
  public:
    SchemaEvolutionBaseA() : c_(0) {}
    SchemaEvolutionBaseA(int c) : c_(c) {}
    int c_;
  };

#if defined DataFormats_TestObjects_USE_OLD
  class SchemaEvolutionAddBase {
  public:
    SchemaEvolutionAddBase() : a_(0), b_(0) {}
    SchemaEvolutionAddBase(int a, int b, int) : a_(a), b_(b) {}
#else
  class SchemaEvolutionAddBase : public SchemaEvolutionBaseA {
  public:
    SchemaEvolutionAddBase() : a_(0), b_(0) {}
    SchemaEvolutionAddBase(int a, int b, int c) : SchemaEvolutionBaseA(c), a_(a), b_(b) {}
#endif
    int a_;
    int b_;
  };

  class SchemaEvolutionContained {
  public:
    SchemaEvolutionContained() : c_(0) {}
    SchemaEvolutionContained(int c) : c_(c) {}
    int c_;
  };

  class SchemaEvolutionPointerToMember {
  public:
#if defined DataFormats_TestObjects_USE_OLD
    SchemaEvolutionPointerToMember() : a_(0), b_(0), contained_(nullptr) {}
    SchemaEvolutionPointerToMember(SchemaEvolutionPointerToMember const& other)
        : a_(other.a_), b_(other.b_), contained_(new SchemaEvolutionContained(other.contained_->c_)) {}
    SchemaEvolutionPointerToMember(SchemaEvolutionPointerToMember&&) = delete;
    SchemaEvolutionPointerToMember& operator=(SchemaEvolutionPointerToMember const&) = delete;
    SchemaEvolutionPointerToMember& operator=(SchemaEvolutionPointerToMember&&) = delete;

    SchemaEvolutionPointerToMember(int a, int b, int c) : a_(a), b_(b), contained_(new SchemaEvolutionContained(c)) {}
    ~SchemaEvolutionPointerToMember() { delete contained_; }
    int c() const { return contained_->c_; }
    int a_;
    int b_;
    SchemaEvolutionContained* contained_;
#else
    SchemaEvolutionPointerToMember() : a_(0), b_(0) {}
    SchemaEvolutionPointerToMember(int a, int b, int c) : a_(a), b_(b), contained_(c) {}
    int c() const { return contained_.c_; }
    int a_;
    int b_;
    SchemaEvolutionContained contained_;
#endif
  };

  class SchemaEvolutionPointerToUniquePtr {
  public:
    SchemaEvolutionPointerToUniquePtr(SchemaEvolutionPointerToUniquePtr&&) = delete;
    SchemaEvolutionPointerToUniquePtr& operator=(SchemaEvolutionPointerToUniquePtr const&) = delete;
    SchemaEvolutionPointerToUniquePtr& operator=(SchemaEvolutionPointerToUniquePtr&&) = delete;

#if defined DataFormats_TestObjects_USE_OLD
    SchemaEvolutionPointerToUniquePtr() : a_(0), b_(0), contained_(nullptr) {}
    SchemaEvolutionPointerToUniquePtr(SchemaEvolutionPointerToUniquePtr const& other)
        : a_(other.a_), b_(other.b_), contained_(new SchemaEvolutionContained(other.contained_->c_)) {}

    SchemaEvolutionPointerToUniquePtr(int a, int b, int c)
        : a_(a), b_(b), contained_(new SchemaEvolutionContained(c)) {}
    ~SchemaEvolutionPointerToUniquePtr() { delete contained_; }
    int a_;
    int b_;
    SchemaEvolutionContained* contained_;
#else
    SchemaEvolutionPointerToUniquePtr() : a_(0), b_(0) {}
    SchemaEvolutionPointerToUniquePtr(int a, int b, int c)
        : a_(a), b_(b), contained_(std::make_unique<SchemaEvolutionContained>(c)) {}
    SchemaEvolutionPointerToUniquePtr(SchemaEvolutionPointerToUniquePtr const& other)
        : a_(other.a_), b_(other.b_), contained_(std::make_unique<SchemaEvolutionContained>(other.contained_->c_)) {}
    int a_;
    int b_;
    std::unique_ptr<SchemaEvolutionContained> contained_;
#endif
  };

  class SchemaEvolutionCArrayToStdArray {
  public:
#if defined DataFormats_TestObjects_USE_OLD
    SchemaEvolutionCArrayToStdArray() : a_{0, 0, 0} {}
    SchemaEvolutionCArrayToStdArray(int x, int y, int z) : a_{x, y, z} {}
    int a_[3];
#else
    SchemaEvolutionCArrayToStdArray() : a_{{0, 0, 0}} {}
    SchemaEvolutionCArrayToStdArray(int x, int y, int z) : a_{{x, y, z}} {}
    std::array<int, 3> a_;
#endif
  };

  class SchemaEvolutionCArrayToStdVector {
  public:
#if defined DataFormats_TestObjects_USE_OLD
    SchemaEvolutionCArrayToStdVector() : a_{new int[fSize_]{0, 0, 0}} {}
    SchemaEvolutionCArrayToStdVector(int x, int y, int z) : a_{new int[fSize_]{x, y, z}} {}
    SchemaEvolutionCArrayToStdVector(SchemaEvolutionCArrayToStdVector const& other)
        : a_(new int[fSize_]{other.a_[0], other.a_[1], other.a_[2]}) {}
    SchemaEvolutionCArrayToStdVector(SchemaEvolutionCArrayToStdVector&&) = delete;
    SchemaEvolutionCArrayToStdVector& operator=(SchemaEvolutionCArrayToStdVector const&) = delete;
    SchemaEvolutionCArrayToStdVector& operator=(SchemaEvolutionCArrayToStdVector&&) = delete;
    ~SchemaEvolutionCArrayToStdVector() { delete[] a_; }

    int fSize_ = 3;
    int* a_;  //[fSize_]
#else
    SchemaEvolutionCArrayToStdVector() : a_{0, 0, 0} {}
    SchemaEvolutionCArrayToStdVector(int x, int y, int z) : a_{x, y, z} {}
    std::vector<int> a_;
#endif
  };

  class SchemaEvolutionVectorToList {
  public:
    SchemaEvolutionVectorToList() : a_{0, 0, 0} {}
    SchemaEvolutionVectorToList(int x, int y, int z) : a_{x, y, z} {}
#if defined DataFormats_TestObjects_USE_OLD
    std::vector<int> a_;
#else
    std::list<int> a_;
#endif
  };

  class SchemaEvolutionMapToUnorderedMap {
  public:
    SchemaEvolutionMapToUnorderedMap() {
      a_.insert({0, 0});
      a_.insert({1, 0});
      a_.insert({2, 0});
    }
    SchemaEvolutionMapToUnorderedMap(int keyX, int x, int keyY, int y, int keyZ, int z) {
      a_.insert({keyX, x});
      a_.insert({keyY, y});
      a_.insert({keyZ, z});
    }
#if defined DataFormats_TestObjects_USE_OLD
    std::map<int, int> a_;
#else
    std::unordered_map<int, int> a_;
#endif
  };

  class VectorVectorElement {
  public:
    VectorVectorElement();
    VectorVectorElement(int a,
                        int b,
                        SchemaEvolutionChangeOrder const&,
                        SchemaEvolutionAddMember const&,
                        SchemaEvolutionRemoveMember const&,
                        SchemaEvolutionMoveToBase const&,
                        SchemaEvolutionChangeType const&,
                        SchemaEvolutionAddBase const&,
                        SchemaEvolutionPointerToMember const&,
                        SchemaEvolutionPointerToUniquePtr const&,
                        SchemaEvolutionCArrayToStdArray const&,
                        // SchemaEvolutionCArrayToStdVector const&,
                        SchemaEvolutionVectorToList const&,
                        SchemaEvolutionMapToUnorderedMap const&);
#if defined DataFormats_TestObjects_USE_OLD
    int a_;
    int b_;
#else
    int a_;
    int b_;
    int c_ = 0;
#endif
    SchemaEvolutionChangeOrder changeOrder_;
    SchemaEvolutionAddMember addMember_;
    SchemaEvolutionRemoveMember removeMember_;
    SchemaEvolutionMoveToBase moveToBase_;
    SchemaEvolutionChangeType changeType_;
    SchemaEvolutionAddBase addBase_;
    SchemaEvolutionPointerToMember pointerToMember_;
    SchemaEvolutionPointerToUniquePtr pointerToUniquePtr_;
    SchemaEvolutionCArrayToStdArray cArrayToStdArray_;
    // This one is commented out because it fails reading an old format
    // input file with an executable built with the modified format.
    // If the issue in ROOT is ever fixed and this is added back,
    // it also would need to be added into the constructor above.
    // SchemaEvolutionCArrayToStdVector cArrayToStdVector_;
    SchemaEvolutionVectorToList vectorToList_;
    SchemaEvolutionMapToUnorderedMap mapToUnorderedMap_;
  };

  class VectorVectorElementNonSplit {
  public:
    VectorVectorElementNonSplit();
    VectorVectorElementNonSplit(int a, int b);
#if defined DataFormats_TestObjects_USE_OLD
    // This version of the class is forced to be non-split because
    // it has only one data member. The unit test this is used by
    // was developed in response to a ROOT bug in the version of ROOT
    // associated with CMSSW_13_0_0. This bug only affected non split
    // classes and this class was necessary to reproduce it.
    int a_;
#else
    int a_;
    int b_;
#endif
  };

}  // namespace edmtest

#endif
