/*
 * SoAStreamer_t.cpp
 * 
 * A test validating and the serialization of SoA Layouts to a ROOT file
 */

#include <cstdlib>
#include <memory>

#include <TFile.h>
#include <TTree.h>

#include "FakeSoA.h"

void writeSoA() {
  std::cout << "write begin" << std::endl;
  constexpr size_t nElements = 128;

  auto buffer = std::make_unique<std::byte[]>(FakeSoA::computeBufferSize(nElements));
  FakeSoA fsoa(buffer.get(), nElements);
  fsoa.dump();
  fsoa.fill();
  if (not fsoa.check()) {
    exit(EXIT_FAILURE);
  }

  std::unique_ptr<TFile> myFile(TFile::Open("serializerNoTObj.root", "RECREATE"));
  TTree tt("serializerNoTObjTree", "A SoA TTree");
  // In CMSSW, we will get a branch of objects (each row from the branched corresponding to an event)
  // So we have a branch with one element for the moment.
  [[maybe_unused]] auto Branch = tt.Branch("FakeSoA", &fsoa);
  std::cout << "In writeFile(), about to Fill()" << std::endl;
  fsoa.dump();
  auto prevGDebug = gDebug;
  gDebug = 5;
  tt.Fill();
  gDebug = prevGDebug;
  tt.Write();
  myFile->Close();
  std::cout << "write end" << std::endl;
}

void readSoA() {
  std::cout << "read begin" << std::endl;
  std::unique_ptr<TFile> myFile(TFile::Open("serializerNoTObj.root", "READ"));
  myFile->ls();
  std::unique_ptr<TTree> fakeSoATree((TTree *)myFile->Get("serializerNoTObjTree"));
  fakeSoATree->ls();
  auto prevGDebug = gDebug;
  //gDebug = 3;
  FakeSoA *fakeSoA = nullptr;
  fakeSoATree->SetBranchAddress("FakeSoA", &fakeSoA);
  fakeSoATree->GetEntry(0);
  gDebug = prevGDebug;
  std::cout << "fakeSoAAddress=" << fakeSoA << std::endl;
  fakeSoA->dump();
  fakeSoA->dumpData();
  std::cout << "Checking SoA readback...";
  if (not fakeSoA->check()) {
    exit(EXIT_FAILURE);
  }
  std::cout << " OK" << std::endl;
}

int main() {
  writeSoA();
  readSoA();
  return EXIT_SUCCESS;
}
