#ifndef LSTEff_H
#define LSTEff_H

#include "TBranch.h"
#include "TTree.h"

#include <vector>
#include <unordered_map>
#include <string>
#include <variant>

using DataTypes = std::variant<int*,
                               float*,
                               std::vector<int>*,
                               std::vector<float>*,
                               std::vector<std::vector<int>>*,
                               std::vector<std::vector<float>>*>;

struct BranchData {
  TBranch* branch;
  bool isLoaded;
  DataTypes ptr;
};

class LSTEff {
private:
  TTree* tree;
  unsigned int index;
  std::unordered_map<std::string, BranchData> data;

public:
  void Init(TTree* tree);
  void GetEntry(unsigned int idx);
  static void progress(int nEventsTotal, int nEventsChain);
  void loadAllBranches();
  template <typename T>
  T const& get(std::string name);
  int const& getI(std::string name);
  float const& getF(std::string name);
  std::vector<int> const& getVI(std::string name);
  std::vector<float> const& getVF(std::string name);
  std::vector<std::vector<int>> const& getVVI(std::string name);
  std::vector<std::vector<float>> const& getVVF(std::string name);
};

#ifndef __CINT__
extern LSTEff lstEff;
#endif

#endif
