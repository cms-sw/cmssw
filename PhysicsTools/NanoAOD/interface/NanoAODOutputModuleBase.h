// -*- C++ -*-
//
// Package:     PhysicsTools/NanoAODOutput
// Class  :     NanoAODOutputModuleBase
//
// Implementation:
//     Base class for OutputModules that create
//     flat NanoAOD ntuples based on ROOT's TTree
//
// Original Author:  Christopher Jones
//         Created:  Mon, 07 Aug 2017 14:21:41 GMT
//
#ifndef PhysicsTools_NanoAOD_NanoAODOutputModuleBase_h
#define PhysicsTools_NanoAOD_NanoAODOutputModuleBase_h

#include <memory>
#include <string>
#include <vector>

#include "TFile.h"
#include "TTree.h"

#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

class NanoAODOutputModuleBase : public edm::one::OutputModule<> {
public:
  explicit NanoAODOutputModuleBase(edm::ParameterSet const& pset);
  ~NanoAODOutputModuleBase() override = default;

  static void fillDescription(edm::ParameterSetDescription& desc);

private:
  void write(edm::EventForOutput const& iEvent) override;
  void writeLuminosityBlock(edm::LuminosityBlockForOutput const& iLumi) override;
  void writeRun(edm::RunForOutput const& iRun) override;

  bool isFileOpen() const override;
  void openFile(edm::FileBlock const& iFile) override;
  void reallyCloseFile() override;

  // The three functions below are called inside
  // write(iEvent), writeLuminosityBlock(iLumi) and writeRun(iRun), respectively.
  // They must include all the calls needed to fill
  // the corresponding TTree for a given Event/Lumi/Run.
  // The write(iEvent), writeLuminosityBlock(iLumi) and writeRun(iRun) functions
  // handle the information for the framework's JobReport and ProcessHistoryRegistry,
  // as well as auto-flushing for the Event's TTree (if enabled),
  // and the three functions below do the rest.
  virtual void writeEventTree(edm::EventForOutput const& iEvent) = 0;
  virtual void writeLuminosityBlockTree(edm::LuminosityBlockForOutput const& iLumi) = 0;
  virtual void writeRunTree(edm::RunForOutput const& iRun) = 0;

  // initTables: called in openFile(iFile), and used to
  // fill the vectors of table output branches.
  virtual void initTables() = 0;

  // The three functions below are called inside openFile(iFile)
  // after the Event, Lumi, and Run TTrees have been created.
  // The argument of each function corresponds to the relevant TTree
  // (*m_eventTree for initEventTree, *m_lumiTree for initLuminosityBlockTree,
  // and *m_runTree for initRunTree).
  // These functions contain the calls needed to link branches to the relevant TTree.
  virtual void initEventTree(TTree& tree) = 0;
  virtual void initLuminosityBlockTree(TTree& tree) = 0;
  virtual void initRunTree(TTree& tree) = 0;

  std::string m_fileName;
  std::string m_logicalFileName;
  int m_compressionLevel;
  int m_eventsSinceFlush{0};
  std::string m_compressionAlgorithm;
  bool m_writeProvenance;
  bool m_fakeName;  //crab workaround, remove after crab is fixed
  int m_autoFlush;
  edm::ProcessHistoryRegistry m_processHistoryRegistry;
  edm::JobReport::Token m_jrToken;
  std::unique_ptr<TTree> m_metaDataTree;
  std::unique_ptr<TTree> m_parameterSetsTree;

  static constexpr int kFirstFlush{1000};
  static constexpr const char* kFakeOutputModuleType{"PoolOutputModule"};

protected:
  std::unique_ptr<TFile> m_file;
  std::unique_ptr<TTree> m_eventTree;
  std::unique_ptr<TTree> m_lumiTree;
  std::unique_ptr<TTree> m_runTree;
};

#endif
