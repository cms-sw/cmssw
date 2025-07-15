#include "treeutil.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <iostream>

#include "TLeaf.h"

void TreeUtil::Init(TTree* treeIn) {
  tree = treeIn;
  data.clear();
  data.reserve(1000);
}

void TreeUtil::GetEntry(unsigned int idx) {
  index = idx;
  for (auto& pair : data) {
    pair.second.isLoaded = false;
  }
}

void TreeUtil::progress(int nEventsTotal, int nEventsChain) {
  int period = 1000;
  if (nEventsTotal % 1000 == 0) {
    if (isatty(1)) {
      if ((nEventsChain - nEventsTotal) > period) {
        float frac = (float)nEventsTotal / (nEventsChain * 0.01);
        printf(
            "\015\033[32m ---> \033[1m\033[31m%4.1f%%"
            "\033[0m\033[32m <---\033[0m\015",
            frac);
        fflush(stdout);
      } else {
        printf(
            "\015\033[32m ---> \033[1m\033[31m%4.1f%%"
            "\033[0m\033[32m <---\033[0m\015\n",
            100.);
        fflush(stdout);
      }
    }
  }
}

void TreeUtil::loadAllBranches() {
  for (auto branch : *tree->GetListOfBranches()) {
    TLeaf* leaf = static_cast<TLeaf*>(static_cast<TBranch*>(branch)->GetListOfLeaves()->First());
    std::string leafTypename = leaf->GetTypeName();
    // I'm not sure if there is a better way to do this
    if (leafTypename == "Short_t")
      getS(std::string(branch->GetName()));
    else if (leafTypename == "UShort_t")
      getUS(std::string(branch->GetName()));
    else if (leafTypename == "Int_t")
      getI(std::string(branch->GetName()));
    else if (leafTypename == "UInt_t")
      getU(std::string(branch->GetName()));
    else if (leafTypename == "Float_t")
      getF(std::string(branch->GetName()));
    else if (leafTypename == "vector<short>")
      getVS(std::string(branch->GetName()));
    else if (leafTypename == "vector<unsigned short>")
      getVUS(std::string(branch->GetName()));
    else if (leafTypename == "vector<int>")
      getVI(std::string(branch->GetName()));
    else if (leafTypename == "vector<unsigned int>")
      getVU(std::string(branch->GetName()));
    else if (leafTypename == "vector<float>")
      getVF(std::string(branch->GetName()));
    else if (leafTypename == "vector<vector<short> >")
      getVVS(std::string(branch->GetName()));
    else if (leafTypename == "vector<vector<insigned short> >")
      getVVUS(std::string(branch->GetName()));
    else if (leafTypename == "vector<vector<int> >")
      getVVI(std::string(branch->GetName()));
    else if (leafTypename == "vector<vector<unsigned int> >")
      getVVU(std::string(branch->GetName()));
    else if (leafTypename == "vector<vector<float> >")
      getVVF(std::string(branch->GetName()));
    else
      std::cout << "Skipping branch " << std::string(branch->GetName()) << " with type " << leafTypename << std::endl;
  }
}

template <typename T>
T const& TreeUtil::get(std::string name) {
  auto search = data.find(name);
  if (search == data.end()) {
    tree->SetMakeClass(1);
    TBranch* branch = tree->GetBranch(name.c_str());
    if (branch == nullptr)
      throw std::out_of_range("Branch " + name + " does not exist!");
    search = data.emplace(name, std::move(BranchData{branch, false, static_cast<T*>(nullptr)})).first;
    branch->SetAddress(&search->second.ptr);
    tree->SetMakeClass(0);
  }
  if (!search->second.isLoaded) {
    search->second.branch->GetEntry(index);
    search->second.isLoaded = true;
  }
  return *std::get<T*>(search->second.ptr);
}

short const& TreeUtil::getS(std::string name) { return get<short>(name); }
unsigned short const& TreeUtil::getUS(std::string name) { return get<unsigned short>(name); }
int const& TreeUtil::getI(std::string name) { return get<int>(name); }
unsigned int const& TreeUtil::getU(std::string name) { return get<unsigned int>(name); }
float const& TreeUtil::getF(std::string name) { return get<float>(name); }
std::vector<short> const& TreeUtil::getVS(std::string name) { return get<std::vector<short>>(name); }
std::vector<unsigned short> const& TreeUtil::getVUS(std::string name) { return get<std::vector<unsigned short>>(name); }
std::vector<int> const& TreeUtil::getVI(std::string name) { return get<std::vector<int>>(name); }
std::vector<unsigned int> const& TreeUtil::getVU(std::string name) { return get<std::vector<unsigned int>>(name); }
std::vector<float> const& TreeUtil::getVF(std::string name) { return get<std::vector<float>>(name); }
std::vector<std::vector<short>> const& TreeUtil::getVVS(std::string name) {
  return get<std::vector<std::vector<short>>>(name);
}
std::vector<std::vector<unsigned short>> const& TreeUtil::getVVUS(std::string name) {
  return get<std::vector<std::vector<unsigned short>>>(name);
}
std::vector<std::vector<int>> const& TreeUtil::getVVI(std::string name) {
  return get<std::vector<std::vector<int>>>(name);
}
std::vector<std::vector<unsigned int>> const& TreeUtil::getVVU(std::string name) {
  return get<std::vector<std::vector<unsigned int>>>(name);
}
std::vector<std::vector<float>> const& TreeUtil::getVVF(std::string name) {
  return get<std::vector<std::vector<float>>>(name);
}
