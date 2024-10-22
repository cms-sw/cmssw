#ifndef RecoTracker_MkFitCore_standalone_RntDumper_RntDumper_h
#define RecoTracker_MkFitCore_standalone_RntDumper_RntDumper_h

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class TFile;
class TTree;

namespace ROOT::Experimental {
  class RNTupleModel;
  class RNTupleWriter;
}  // namespace ROOT::Experimental

class RntDumper {
  using RNTupleModel = ROOT::Experimental::RNTupleModel;
  using RNTupleWriter = ROOT::Experimental::RNTupleWriter;

public:
  std::unique_ptr<RNTupleModel> CreateModel();
  RNTupleWriter *WritifyModel(std::unique_ptr<RNTupleModel> &model, std::string_view mname);

  void RegisterTree(TTree *t);

  static RntDumper *Create(const char *fname);
  static void FinalizeAll();

  TFile *file() { return m_file.get(); }

private:
  explicit RntDumper(const char *fname);
  ~RntDumper();

  std::unique_ptr<TFile> m_file;
  std::unordered_map<std::string, std::unique_ptr<RNTupleWriter>> m_writers;
  std::vector<TTree *> m_trees;

  static std::vector<RntDumper *> s_instances;
};

#endif
