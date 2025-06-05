#include "LSTEff.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <iostream>

#include "TLeaf.h"

LSTEff lstEff;

void LSTEff::Init(TTree* treeIn) {
  tree = treeIn;
  data.reserve(100);
}

void LSTEff::GetEntry(unsigned int idx) {
  index = idx;
  for (auto& pair : data) {
    pair.second.isLoaded = false;
  }
}

void LSTEff::progress(int nEventsTotal, int nEventsChain) {
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

void LSTEff::loadAllBranches() {
  for (auto branch : *tree->GetListOfBranches()) {
    TLeaf* leaf = static_cast<TLeaf*>(static_cast<TBranch*>(branch)->GetListOfLeaves()->First());
    std::string leafTypename = leaf->GetTypeName();
    // I'm not sure if there is a better way to do this
    if (leafTypename == "Int_t")
      get<int>(std::string(branch->GetName()));
    else if (leafTypename == "Float_t")
      get<float>(std::string(branch->GetName()));
    else if (leafTypename == "vector<int>")
      get<std::vector<int>>(std::string(branch->GetName()));
    else if (leafTypename == "vector<float>")
      get<std::vector<float>>(std::string(branch->GetName()));
    else if (leafTypename == "vector<vector<int> >")
      get<std::vector<std::vector<int>>>(std::string(branch->GetName()));
    else if (leafTypename == "vector<vector<float> >")
      get<std::vector<std::vector<float>>>(std::string(branch->GetName()));
    else
      std::cout << "Skipping branch " << std::string(branch->GetName()) << " with type " << leafTypename << std::endl;
  }
}

template <typename T>
T const& LSTEff::get(std::string name) {
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

int const& LSTEff::getI(std::string name) { return get<int>(name); }
float const& LSTEff::getF(std::string name) { return get<float>(name); }
std::vector<int> const& LSTEff::getVI(std::string name) { return get<std::vector<int>>(name); }
std::vector<float> const& LSTEff::getVF(std::string name) { return get<std::vector<float>>(name); }
std::vector<std::vector<int>> const& LSTEff::getVVI(std::string name) {
  return get<std::vector<std::vector<int>>>(name);
}
std::vector<std::vector<float>> const& LSTEff::getVVF(std::string name) {
  return get<std::vector<std::vector<float>>>(name);
}
