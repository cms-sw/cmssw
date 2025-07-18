//  .
// ..: P. Chang, philip@physics.ucsd.edu

#ifndef ttreex_h
#define ttreex_h

// C/C++
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdarg.h>
#include <functional>
#include <cmath>
#include <utility>

// ROOT
#include "TBenchmark.h"
#include "TBits.h"
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TH1.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TChainElement.h"
#include "TTreeCache.h"
#include "TTreePerfStats.h"
#include "TStopwatch.h"
#include "TSystem.h"
#include "TString.h"
#include "TLorentzVector.h"
#include "Math/LorentzVector.h"
#include "Math/GenVector/PtEtaPhiM4D.h"

#include "printutil.h"

//#define MAP std::unordered_map
//#define STRING std::string
#define MAP std::map
//#define TTREEXSTRING std::string
#define TTREEXSTRING TString

///////////////////////////////////////////////////////////////////////////////////////////////
// LorentzVector typedef that we use very often
///////////////////////////////////////////////////////////////////////////////////////////////
#ifdef LorentzVectorPtEtaPhiM4D
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float>> LV;
#else
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float>> LV;
#endif

typedef std::vector<Int_t> VInt;
typedef std::vector<Float_t> VFloat;

namespace RooUtil {

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // TTreeX class
  ///////////////////////////////////////////////////////////////////////////////////////////////
  // NOTE: This class assumes accessing TTree in the SNT style which uses the following,
  // https://github.com/cmstas/Software/blob/master/makeCMS3ClassFiles/makeCMS3ClassFiles.C
  // It is assumed that the "template" class passed to this class will have
  // 1. "Init(TTree*)"
  // 2. "GetEntry(uint)"
  // 3. "progress(nevtProc'ed, total)"
  class TTreeX {
  public:
    enum kType {
      kInt_t = 1,
      kBool_t = 2,
      kFloat_t = 3,
      kTString = 4,
      kLV = 5,
      kVecInt_t = 11,
      kVecUInt_t = 12,
      kVecBool_t = 13,
      kVecFloat_t = 14,
      kVecTString = 15,
      kVecLV = 16
    };
    typedef std::vector<LV>::const_iterator lviter;

  private:
    TTree* ttree;
    std::map<TTREEXSTRING, Int_t> mapInt_t;
    std::map<TTREEXSTRING, Bool_t> mapBool_t;
    std::map<TTREEXSTRING, Float_t> mapFloat_t;
    std::map<TTREEXSTRING, TString> mapTString;
    std::map<TTREEXSTRING, LV> mapLV;
    std::map<TTREEXSTRING, TBits> mapTBits;
    std::map<TTREEXSTRING, unsigned long long> mapULL;
    std::map<TTREEXSTRING, unsigned int> mapUI;
    std::map<TTREEXSTRING, std::vector<Int_t>> mapVecInt_t;
    std::map<TTREEXSTRING, std::vector<UInt_t>> mapVecUInt_t;
    std::map<TTREEXSTRING, std::vector<Bool_t>> mapVecBool_t;
    std::map<TTREEXSTRING, std::vector<Float_t>> mapVecFloat_t;
    std::map<TTREEXSTRING, std::vector<TString>> mapVecTString;
    std::map<TTREEXSTRING, std::vector<LV>> mapVecLV;
    std::map<TTREEXSTRING, std::vector<std::vector<Int_t>>> mapVecVInt;
    std::map<TTREEXSTRING, std::vector<std::vector<Float_t>>> mapVecVFloat;

    std::map<TTREEXSTRING, std::function<float()>> mapFloatFunc_t;

    std::map<TTREEXSTRING, Bool_t> mapIsBranchSet;

  public:
    TTreeX();
    TTreeX(TString treename, TString title);
    TTreeX(TTree* tree);
    ~TTreeX();
    TTree* getTree() { return ttree; }
    void setTree(TTree* tree) { ttree = tree; }
    void* getValPtr(TString brname);
    template <class T>
    T* get(TString brname, int entry = -1);
    void fill() { ttree->Fill(); }
    void write() { ttree->Write(); }

    template <class T>
    void createBranch(TString, bool = true);
    template <class T>
    void setBranch(TString, T, bool = false, bool = false);
    template <class T>
    const T& getBranch(TString, bool = true);
    template <class T>
    const T& getBranchLazy(TString);
    template <class T>
    bool isBranchSet(TString);
    template <class T>
    T* getBranchAddress(TString);
    template <class T>
    void createBranch(T&);
    template <class T>
    bool hasBranch(TString);
    template <class T>
    void setBranch(T&);
    template <class T>
    void pushbackToBranch(TString, T);

    void sortVecBranchesByPt(TString, std::vector<TString>, std::vector<TString>, std::vector<TString>);
    template <class T>
    std::vector<T> sortFromRef(std::vector<T> const& in, std::vector<std::pair<size_t, lviter>> const& reference);
    struct ordering {
      bool operator()(std::pair<size_t, lviter> const& a, std::pair<size_t, lviter> const& b) {
        return (*(a.second)).pt() > (*(b.second)).pt();
      }
    };
    void createFlatBranch(std::vector<TString>, std::vector<TString>, std::vector<TString>, std::vector<TString>, int);
    void setFlatBranch(std::vector<TString>, std::vector<TString>, std::vector<TString>, std::vector<TString>, int);

    void clear();
    void save(TFile*);
  };

  //_________________________________________________________________________________________________
  template <>
  void TTreeX::setBranch<Int_t>(TString bn, Int_t val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<Bool_t>(TString bn, Bool_t val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<Float_t>(TString bn, Float_t val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<TString>(TString bn, TString val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<LV>(TString bn, LV val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<TBits>(TString bn, TBits val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<unsigned long long>(TString bn, unsigned long long val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<unsigned int>(TString bn, unsigned int val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<std::vector<Int_t>>(TString bn, std::vector<Int_t> val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<std::vector<UInt_t>>(TString bn, std::vector<UInt_t> val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<std::vector<Bool_t>>(TString bn, std::vector<Bool_t> val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<std::vector<Float_t>>(TString bn, std::vector<Float_t> val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<std::vector<TString>>(TString bn, std::vector<TString> val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<std::vector<LV>>(TString bn, std::vector<LV> val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<std::vector<VInt>>(TString bn, std::vector<VInt> val, bool force, bool ignore);
  template <>
  void TTreeX::setBranch<std::vector<VFloat>>(TString bn, std::vector<VFloat> val, bool force, bool ignore);
  template <>
  void TTreeX::pushbackToBranch<Int_t>(TString bn, Int_t val);
  template <>
  void TTreeX::pushbackToBranch<UInt_t>(TString bn, UInt_t val);
  template <>
  void TTreeX::pushbackToBranch<Bool_t>(TString bn, Bool_t val);
  template <>
  void TTreeX::pushbackToBranch<Float_t>(TString bn, Float_t val);
  template <>
  void TTreeX::pushbackToBranch<TString>(TString bn, TString val);
  template <>
  void TTreeX::pushbackToBranch<LV>(TString bn, LV val);
  template <>
  void TTreeX::pushbackToBranch<VInt>(TString bn, VInt val);
  template <>
  void TTreeX::pushbackToBranch<VFloat>(TString bn, VFloat val);
  // functor
  template <>
  void TTreeX::setBranch<std::function<float()>>(TString bn, std::function<float()> val, bool force, bool ignore);

  //_________________________________________________________________________________________________
  template <>
  const Int_t& TTreeX::getBranch<Int_t>(TString bn, bool check);
  template <>
  const Bool_t& TTreeX::getBranch<Bool_t>(TString bn, bool check);
  template <>
  const Float_t& TTreeX::getBranch<Float_t>(TString bn, bool check);
  template <>
  const TString& TTreeX::getBranch<TString>(TString bn, bool check);
  template <>
  const LV& TTreeX::getBranch<LV>(TString bn, bool check);
  template <>
  const TBits& TTreeX::getBranch<TBits>(TString bn, bool check);
  template <>
  const unsigned long long& TTreeX::getBranch<unsigned long long>(TString bn, bool check);
  template <>
  const unsigned int& TTreeX::getBranch<unsigned int>(TString bn, bool check);
  template <>
  const std::vector<Int_t>& TTreeX::getBranch<std::vector<Int_t>>(TString bn, bool check);
  template <>
  const std::vector<UInt_t>& TTreeX::getBranch<std::vector<UInt_t>>(TString bn, bool check);
  template <>
  const std::vector<Bool_t>& TTreeX::getBranch<std::vector<Bool_t>>(TString bn, bool check);
  template <>
  const std::vector<Float_t>& TTreeX::getBranch<std::vector<Float_t>>(TString bn, bool check);
  template <>
  const std::vector<TString>& TTreeX::getBranch<std::vector<TString>>(TString bn, bool check);
  template <>
  const std::vector<LV>& TTreeX::getBranch<std::vector<LV>>(TString bn, bool check);
  template <>
  const std::vector<VInt>& TTreeX::getBranch<std::vector<VInt>>(TString bn, bool check);
  template <>
  const std::vector<VFloat>& TTreeX::getBranch<std::vector<VFloat>>(TString bn, bool check);
  // functor
  template <>
  const std::function<float()>& TTreeX::getBranch<std::function<float()>>(TString bn, bool check);

  //_________________________________________________________________________________________________
  template <>
  const Int_t& TTreeX::getBranchLazy<Int_t>(TString bn);
  template <>
  const Bool_t& TTreeX::getBranchLazy<Bool_t>(TString bn);
  template <>
  const Float_t& TTreeX::getBranchLazy<Float_t>(TString bn);
  template <>
  const TString& TTreeX::getBranchLazy<TString>(TString bn);
  template <>
  const LV& TTreeX::getBranchLazy<LV>(TString bn);
  template <>
  const TBits& TTreeX::getBranchLazy<TBits>(TString bn);
  template <>
  const unsigned long long& TTreeX::getBranchLazy<unsigned long long>(TString bn);
  template <>
  const unsigned int& TTreeX::getBranchLazy<unsigned int>(TString bn);
  template <>
  const std::vector<Int_t>& TTreeX::getBranchLazy<std::vector<Int_t>>(TString bn);
  template <>
  const std::vector<UInt_t>& TTreeX::getBranchLazy<std::vector<UInt_t>>(TString bn);
  template <>
  const std::vector<Bool_t>& TTreeX::getBranchLazy<std::vector<Bool_t>>(TString bn);
  template <>
  const std::vector<Float_t>& TTreeX::getBranchLazy<std::vector<Float_t>>(TString bn);
  template <>
  const std::vector<TString>& TTreeX::getBranchLazy<std::vector<TString>>(TString bn);
  template <>
  const std::vector<LV>& TTreeX::getBranchLazy<std::vector<LV>>(TString bn);
  template <>
  const std::vector<VInt>& TTreeX::getBranchLazy<std::vector<VInt>>(TString bn);
  template <>
  const std::vector<VFloat>& TTreeX::getBranchLazy<std::vector<VFloat>>(TString bn);
  // functor
  template <>
  const std::function<float()>& TTreeX::getBranchLazy<std::function<float()>>(TString bn);

  //_________________________________________________________________________________________________
  template <>
  bool TTreeX::isBranchSet<Int_t>(TString bn);
  template <>
  bool TTreeX::isBranchSet<Bool_t>(TString bn);
  template <>
  bool TTreeX::isBranchSet<Float_t>(TString bn);
  template <>
  bool TTreeX::isBranchSet<TString>(TString bn);
  template <>
  bool TTreeX::isBranchSet<LV>(TString bn);
  template <>
  bool TTreeX::isBranchSet<TBits>(TString bn);
  template <>
  bool TTreeX::isBranchSet<unsigned long long>(TString bn);
  template <>
  bool TTreeX::isBranchSet<unsigned int>(TString bn);
  template <>
  bool TTreeX::isBranchSet<std::vector<Int_t>>(TString bn);
  template <>
  bool TTreeX::isBranchSet<std::vector<UInt_t>>(TString bn);
  template <>
  bool TTreeX::isBranchSet<std::vector<Bool_t>>(TString bn);
  template <>
  bool TTreeX::isBranchSet<std::vector<Float_t>>(TString bn);
  template <>
  bool TTreeX::isBranchSet<std::vector<TString>>(TString bn);
  template <>
  bool TTreeX::isBranchSet<std::vector<LV>>(TString bn);
  template <>
  bool TTreeX::isBranchSet<std::vector<VInt>>(TString bn);
  template <>
  bool TTreeX::isBranchSet<std::vector<VFloat>>(TString bn);
  // functors
  template <>
  bool TTreeX::isBranchSet<std::function<float()>>(TString bn);

  //_________________________________________________________________________________________________
  template <>
  Int_t* TTreeX::getBranchAddress<Int_t>(TString bn);
  template <>
  Bool_t* TTreeX::getBranchAddress<Bool_t>(TString bn);
  template <>
  Float_t* TTreeX::getBranchAddress<Float_t>(TString bn);

  //_________________________________________________________________________________________________
  template <>
  void TTreeX::createBranch<Int_t>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<Bool_t>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<Float_t>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<TString>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<LV>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<TBits>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<unsigned long long>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<unsigned int>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<std::vector<Int_t>>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<std::vector<UInt_t>>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<std::vector<Bool_t>>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<std::vector<Float_t>>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<std::vector<TString>>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<std::vector<LV>>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<std::vector<VInt>>(TString bn, bool writeToTree);
  template <>
  void TTreeX::createBranch<std::vector<VFloat>>(TString bn, bool writeToTree);
  // functors
  template <>
  void TTreeX::createBranch<std::function<float()>>(
      TString bn,
      bool writeToTree);  // writeToTree in this specification does not matter it will never write a std::function to TTree

  //_________________________________________________________________________________________________
  template <>
  bool TTreeX::hasBranch<Int_t>(TString bn);
  template <>
  bool TTreeX::hasBranch<Bool_t>(TString bn);
  template <>
  bool TTreeX::hasBranch<Float_t>(TString bn);
  template <>
  bool TTreeX::hasBranch<TString>(TString bn);
  template <>
  bool TTreeX::hasBranch<LV>(TString bn);
  template <>
  bool TTreeX::hasBranch<TBits>(TString bn);
  template <>
  bool TTreeX::hasBranch<unsigned long long>(TString bn);
  template <>
  bool TTreeX::hasBranch<unsigned int>(TString bn);
  template <>
  bool TTreeX::hasBranch<std::vector<Int_t>>(TString bn);
  template <>
  bool TTreeX::hasBranch<std::vector<UInt_t>>(TString bn);
  template <>
  bool TTreeX::hasBranch<std::vector<Bool_t>>(TString bn);
  template <>
  bool TTreeX::hasBranch<std::vector<Float_t>>(TString bn);
  template <>
  bool TTreeX::hasBranch<std::vector<TString>>(TString bn);
  template <>
  bool TTreeX::hasBranch<std::vector<LV>>(TString bn);
  template <>
  bool TTreeX::hasBranch<std::vector<VInt>>(TString bn);
  template <>
  bool TTreeX::hasBranch<std::vector<VFloat>>(TString bn);
  // functors
  template <>
  bool TTreeX::hasBranch<std::function<float()>>(TString bn);

  //_________________________________________________________________________________________________
  template <>
  void TTreeX::setBranch<std::map<TTREEXSTRING, std::vector<Int_t>>>(std::map<TTREEXSTRING, std::vector<Int_t>>& objidx);
  template <>
  void TTreeX::createBranch<std::map<TTREEXSTRING, std::vector<Int_t>>>(
      std::map<TTREEXSTRING, std::vector<Int_t>>& objidx);

  template <class T>
  std::vector<T> TTreeX::sortFromRef(std::vector<T> const& in,
                                     std::vector<std::pair<size_t, TTreeX::lviter>> const& reference) {
    std::vector<T> ret(in.size());

    size_t const size = in.size();
    for (size_t i = 0; i < size; ++i)
      ret[i] = in[reference[i].first];

    return ret;
  }

}  // namespace RooUtil

//_________________________________________________________________________________________________
template <class T>
T* RooUtil::TTreeX::get(TString brname, int entry) {
  if (entry >= 0)
    ttree->GetEntry(entry);
  return (T*)getValPtr(brname);
}

#endif
