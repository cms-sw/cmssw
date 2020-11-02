#ifndef FWCore_Utilities_ProcessBlockIndex_h
#define FWCore_Utilities_ProcessBlockIndex_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     edm::ProcessBlockIndex
//
/**\class edm::ProcessBlockIndex

 Description: Identifies a ProcessBlock

 Usage:
*/
//
// Original Author:  W. David Dagenhart
//         Created:  18 March 2020

namespace edm {
  class ProcessBlockPrincipal;

  class ProcessBlockIndex {
  public:
    ProcessBlockIndex() = delete;
    ~ProcessBlockIndex() = default;
    ProcessBlockIndex(const ProcessBlockIndex&) = default;
    ProcessBlockIndex& operator=(const ProcessBlockIndex&) = default;

    bool operator==(const ProcessBlockIndex& iIndex) const { return value() == iIndex.value(); }
    operator unsigned int() const { return value_; }

    unsigned int value() const { return value_; }

    static ProcessBlockIndex invalidProcessBlockIndex();

  private:
    ///Only the ProcessBlockPrincipal is allowed to make one of these
    friend class ProcessBlockPrincipal;
    explicit ProcessBlockIndex(unsigned int iIndex) : value_(iIndex) {}

    unsigned int value_;

    static const unsigned int invalidValue_;
  };
}  // namespace edm

#endif
