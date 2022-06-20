/*
 * SoAStreamer_t.cu
 * 
 * A test validating and the serialization of SoA Layouts to a ROOT file
 */

#include <TFile.h>
#include <TTree.h>
#include <memory>
#include "FakeSoA.h"

void writeSoA() {
  std::cout << "write begin" << std::endl << std::flush;
  constexpr size_t nElements = 128;

  auto buffer = std::make_unique<std::byte[]>(FakeSoA::computeBufferSize(nElements));
  FakeSoA fsoa(buffer.get(), nElements);
  fsoa.Dump();
  fsoa.Fill();
  fsoa.Check();

  std::unique_ptr<TFile> myFile(TFile::Open("serializerNoTObj.root", "RECREATE"));
  TTree tt("serializerNoTObjTree", "A SoA TTree");
  // In CMSSW, we will get a branch of objects (each row from the branched corresponding to an event)
  // So we have a branch with one element for the moment.
  [[maybe_unused]] auto Branch = tt.Branch("FakeSoA", &fsoa);
  std::cout << "In writeFile(), about to Fill()" << std::endl;
  fsoa.Dump();
  auto prevGDebug = gDebug;
  gDebug = 5;
  tt.Fill();
  gDebug = prevGDebug;
  tt.Write();
  myFile->Close();
  std::cout << "write end" << std::endl << std::flush;
}

void readSoA() {
  std::cout << "read begin" << std::endl << std::flush;
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
  fakeSoA->Dump();
  fakeSoA->DumpData();
  std::cout << "Checking SoA readback...";
  fakeSoA->Check();
  std::cout << " OK" << std::endl << std::flush;
}

int main() {
  writeSoA();
  readSoA();
  return EXIT_SUCCESS;
}