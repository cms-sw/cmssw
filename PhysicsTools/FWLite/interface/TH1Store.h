// -*- C++ -*-

#if !defined(TH1Store_H)
#define TH1Store_H

#include <map>
#include <string>
#include <set>

#include "TH1.h"
#include "TFile.h"
#include "TString.h"
#include "TDirectory.h"

class TH1Store {
public:
  //////////////////////
  // Public Constants //
  //////////////////////

  typedef std::vector<std::string> SVec;
  typedef std::map<std::string, std::string> SSMap;
  typedef std::map<std::string, TH1 *> STH1PtrMap;
  typedef SSMap::const_iterator SSMapConstIter;
  typedef STH1PtrMap::iterator STH1PtrMapIter;
  typedef STH1PtrMap::const_iterator STH1PtrMapConstIter;

  static const SVec kEmptyVec;

  /////////////
  // friends //
  /////////////
  // tells particle data how to print itself out
  friend std::ostream &operator<<(std::ostream &o_stream, const TH1Store &rhs);

  //////////////////////////
  //            _         //
  // |\/|      |_         //
  // |  |EMBER | UNCTIONS //
  //                      //
  //////////////////////////

  /////////////////////////////////
  // Constructors and Destructor //
  /////////////////////////////////
  TH1Store();
  ~TH1Store();

  ////////////////
  // One Liners //
  ////////////////
  // Whether or not to delete histogram pointers on destruction
  void setDeleteOnDestruction(bool deleteOnDestruction = true) { m_deleteOnDestruction = deleteOnDestruction; }

  //////////////////////////////
  // Regular Member Functions //
  //////////////////////////////

  // adds a histogram pointer to the map
  void add(TH1 *histPtr, const std::string &directory = "");

  // given a string, returns corresponding histogram pointer
  TH1 *hist(const std::string &name);
  TH1 *hist(const char *name) { return hist((const std::string)name); }
  TH1 *hist(const TString &name) { return hist((const char *)name); }

  // write all histograms to a root file
  void write(const std::string &filename, const SVec &argsVec = kEmptyVec, const SVec &inputFilesVec = kEmptyVec) const;
  void write(TFile *filePtr, const SVec &argsVec = kEmptyVec, const SVec &inputFilesVec = kEmptyVec) const;

  /////////////////////////////
  // Static Member Functions //
  /////////////////////////////

  // turn on verbose messages (e.g., printing out histogram names
  // when being made)
  static void setVerbose(bool verbose = true) { sm_verbose = verbose; }

private:
  //////////////////////////////
  // Private Member Functions //
  //////////////////////////////

  // creates directory and all parent directories as needed
  // (equivalent to unix 'mkdir -p') and then changes (cd's) to
  // that directory.  Returns TDirectory of pointing to dirname.
  TDirectory *_createDir(const std::string &dirname, TFile *filePtr) const;

  /////////////////////////
  // Private Member Data //
  /////////////////////////

  bool m_deleteOnDestruction;
  STH1PtrMap m_ptrMap;
  SSMap m_nameDirMap;

  ////////////////////////////////
  // Private Static Member Data //
  ////////////////////////////////

  static bool sm_verbose;
};

#endif  // TH1Store_H
