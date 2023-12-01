#include "RntDumper.h"

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

#include "TFile.h"
#include "TTree.h"

#include <vector>

namespace REX = ROOT::Experimental;

std::vector<RntDumper *> RntDumper::s_instances;

RntDumper::RntDumper(const char *fname) : m_file(TFile::Open(fname, "recreate")) {
  if (!m_file || !m_file->IsOpen()) {
    printf("RntDumper::RntDumper() failed creeating file '%s'.\n", fname);
    throw std::runtime_error("Failed creating file");
  }
  printf("RntDumper::RntDumper() succesfully opened file '%s'.\n", fname);
}

RntDumper::~RntDumper() {
  printf("RntDumper::~RntDumper() destroying writers and closing file '%s'.\n", m_file->GetName());
  // Finish up trees first, ntuple-writers seem to write everything reulting
  // in two cycles of trees.
  for (auto &tp : m_trees) {
    tp->Write();
    delete tp;
  }
  m_trees.clear();
  m_writers.clear();
  if (m_file) {
    m_file->Close();
  }
}

std::unique_ptr<REX::RNTupleModel> RntDumper::CreateModel() { return RNTupleModel::Create(); }

REX::RNTupleWriter *RntDumper::WritifyModel(std::unique_ptr<REX::RNTupleModel> &model, std::string_view mname) {
  auto wup = RNTupleWriter::Append(std::move(model), mname, *m_file);
  REX::RNTupleWriter *w = wup.get();
  m_writers.insert({std::string(mname), std::move(wup)});
  return w;
}

void RntDumper::RegisterTree(TTree *t) { m_trees.push_back(t); }

// === static ===

RntDumper *RntDumper::Create(const char *fname) {
  // Should check fnames ?
  RntDumper *d = new RntDumper(fname);
  s_instances.push_back(d);
  return d;
}

void RntDumper::FinalizeAll() {
  printf("RntDumper::FinalizeAll() shutting down %d instances.\n", (int)s_instances.size());
  for (auto &d : s_instances)
    delete d;
}
