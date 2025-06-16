#ifndef treeutil_H
#define treeutil_H

#include "TBranch.h"
#include "TTree.h"

#include <vector>
#include <unordered_map>
#include <string>
#include <variant>

class TreeUtil {
private:
  using DataTypes = std::variant<short*,
                                 unsigned short*,
                                 int*,
                                 unsigned int*,
                                 float*,
                                 std::vector<short>*,
                                 std::vector<unsigned short>*,
                                 std::vector<int>*,
                                 std::vector<unsigned int>*,
                                 std::vector<float>*,
                                 std::vector<std::vector<short>>*,
                                 std::vector<std::vector<unsigned short>>*,
                                 std::vector<std::vector<int>>*,
                                 std::vector<std::vector<unsigned int>>*,
                                 std::vector<std::vector<float>>*>;

  struct BranchData {
    TBranch* branch;
    bool isLoaded;
    DataTypes ptr;
  };

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
  short const& getS(std::string name);
  unsigned short const& getUS(std::string name);
  int const& getI(std::string name);
  unsigned int const& getU(std::string name);
  float const& getF(std::string name);
  std::vector<short> const& getVS(std::string name);
  std::vector<unsigned short> const& getVUS(std::string name);
  std::vector<int> const& getVI(std::string name);
  std::vector<unsigned int> const& getVU(std::string name);
  std::vector<float> const& getVF(std::string name);
  std::vector<std::vector<short>> const& getVVS(std::string name);
  std::vector<std::vector<unsigned short>> const& getVVUS(std::string name);
  std::vector<std::vector<int>> const& getVVI(std::string name);
  std::vector<std::vector<unsigned int>> const& getVVU(std::string name);
  std::vector<std::vector<float>> const& getVVF(std::string name);
};

#endif
