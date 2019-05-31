// -*- C++ -*-
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>

#include "boost/regex.hpp"

#include "PhysicsTools/FWLite/interface/TH1Store.h"

using namespace std;

////////////////////////////////////
// Static Member Data Declaration //
////////////////////////////////////

const TH1Store::SVec TH1Store::kEmptyVec;
bool TH1Store::sm_verbose = false;

TH1Store::TH1Store() : m_deleteOnDestruction(false) {}

TH1Store::~TH1Store() {
  if (m_deleteOnDestruction) {
    for (STH1PtrMapIter iter = m_ptrMap.begin(); m_ptrMap.end() != iter; ++iter) {
      delete iter->second;
    }  // for iter
  }    // if destroying pointers
}

void TH1Store::add(TH1 *histPtr, const std::string &directory) {
  // Do we have a histogram with this name already?
  string name = histPtr->GetName();
  if (m_ptrMap.end() != m_ptrMap.find(name)) {
    //  D'oh
    cerr << "TH1Store::add() Error: '" << name << "' already exists.  Aborting." << endl;
    assert(0);
  }  // if already exists
  if (sm_verbose) {
    cout << "THStore::add() : Adding " << name << endl;
  }
  m_ptrMap[name] = histPtr;
  histPtr->SetDirectory(nullptr);
  if (directory.length()) {
    m_nameDirMap[name] = directory;
  }
}

TH1 *TH1Store::hist(const string &name) {
  STH1PtrMapIter iter = m_ptrMap.find(name);
  if (m_ptrMap.end() == iter) {
    //  D'oh
    cerr << "TH1Store::hist() Error: '" << name << "' does not exists.  Aborting." << endl;
    assert(0);
  }  // doesn't exist
  return iter->second;
}

void TH1Store::write(const string &filename, const SVec &argsVec, const SVec &inputFilesVec) const {
  TFile *filePtr = TFile::Open(filename.c_str(), "RECREATE");
  if (!filePtr) {
    cerr << "TH1Store::write() Error: Can not open '" << filename << "' for output.  Aborting." << endl;
    assert(0);
  }
  write(filePtr, argsVec, inputFilesVec);
  delete filePtr;
}

void TH1Store::write(TFile *filePtr, const SVec &argsVec, const SVec &inputFilesVec) const {
  filePtr->cd();
  // write out all histograms
  for (STH1PtrMapConstIter iter = m_ptrMap.begin(); m_ptrMap.end() != iter; ++iter) {
    SSMapConstIter nameDirIter = m_nameDirMap.find(iter->first);
    if (m_nameDirMap.end() != nameDirIter) {
      // we want a subdirectory for this one
      _createDir(nameDirIter->second, filePtr);
    } else {
      // we don't need a subdirectory, just save this in the main
      // directory.
      filePtr->cd();
    }
    iter->second->Write();
  }  // for iter
  // Write out command line arguments.  Save information in directory
  // called provenance.
  TDirectory *dir = _createDir("args", filePtr);
  if (!argsVec.empty()) {
    dir->WriteObject(&argsVec, "argsVec");
  }
  if (!inputFilesVec.empty()) {
    dir->WriteObject(&inputFilesVec, "inputFiles");
  }
  cout << "TH1Store::write(): Successfully written to '" << filePtr->GetName() << "'." << endl;
}

TDirectory *TH1Store::_createDir(const string &dirName, TFile *filePtr) const {
  // do we have this one already
  TDirectory *dirPtr = filePtr->GetDirectory(dirName.c_str());
  if (dirPtr) {
    dirPtr->cd();
    return dirPtr;
  }
  // if we're here, then this directory doesn't exist.  Is this
  // directory a subdirectory?
  const boost::regex subdirRE("(.+?)/([^/]+)");
  boost::smatch matches;
  TDirectory *parentDir = nullptr;
  string useName = dirName;
  if (boost::regex_match(dirName, matches, subdirRE)) {
    parentDir = _createDir(matches[1], filePtr);
    useName = matches[2];
  } else {
    // This is not a subdirectory, so we're golden
    parentDir = filePtr;
  }
  dirPtr = parentDir->mkdir(useName.c_str());
  dirPtr->cd();
  return dirPtr;
}

// friends
ostream &operator<<(ostream &o_stream, const TH1Store &rhs) { return o_stream; }
