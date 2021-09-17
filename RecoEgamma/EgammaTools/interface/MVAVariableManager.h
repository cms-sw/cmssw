#ifndef RecoEgamma_EgammaTools_MVAVariableManager_H
#define RecoEgamma_EgammaTools_MVAVariableManager_H

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "CommonTools/Utils/interface/ThreadSafeFunctor.h"

#include <fstream>

template <class ParticleType>
class MVAVariableManager {
public:
  template <class IndexMap>
  MVAVariableManager(const std::string &variableDefinitionFileName, IndexMap const &indexMap) : nVars_(0) {
    edm::FileInPath variableDefinitionFileEdm(variableDefinitionFileName);
    std::ifstream file(variableDefinitionFileEdm.fullPath());

    std::string name, formula, upper, lower;
    while (true) {
      file >> name;
      if (file.eof()) {
        break;
      }
      if (name.find('#') != std::string::npos) {
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        continue;
      }
      file >> formula >> lower >> upper;
      if (file.eof()) {
        break;
      }
      addVariable(name, formula, lower, upper, indexMap);
    }
  }

  int getVarIndex(const std::string &name) {
    std::map<std::string, int>::iterator it = indexMap_.find(name);
    if (it == indexMap_.end()) {
      return -1;
    } else {
      return it->second;
    }
  }

  const std::string &getName(int index) const { return names_[index]; }

  int getNVars() const { return nVars_; }

  float getValue(int index, const ParticleType &particle, const std::vector<float> &auxVariables) const {
    float value;

    MVAVariableInfo varInfo = variableInfos_[index];

    if (varInfo.auxIndex >= 0)
      value = auxVariables[varInfo.auxIndex];
    else
      value = functions_[index](particle);

    if (varInfo.hasLowerClip && value < varInfo.lowerClipValue) {
      value = varInfo.lowerClipValue;
    }
    if (varInfo.hasUpperClip && value > varInfo.upperClipValue) {
      value = varInfo.upperClipValue;
    }
    return value;
  }

private:
  struct MVAVariableInfo {
    bool hasLowerClip;
    bool hasUpperClip;
    float lowerClipValue;
    float upperClipValue;
    int auxIndex;
  };

  template <class IndexMap>
  void addVariable(const std::string &name,
                   const std::string &formula,
                   const std::string &lowerClip,
                   const std::string &upperClip,
                   IndexMap const &indexMap) {
    bool hasLowerClip = lowerClip.find("None") == std::string::npos;
    bool hasUpperClip = upperClip.find("None") == std::string::npos;
    bool isAuxiliary = formula.find("Rho") != std::string::npos;  // *Rho* is still hardcoded...
    float lowerClipValue = hasLowerClip ? (float)::atof(lowerClip.c_str()) : 0.;
    float upperClipValue = hasUpperClip ? (float)::atof(upperClip.c_str()) : 0.;

    if (!isAuxiliary)
      functions_.emplace_back(formula);
    // Else push back a dummy function since we won't use the
    // StringObjectFunction to evaluate an auxiliary variable
    else
      functions_.emplace_back("pt");

    formulas_.push_back(formula);

    int auxIndex = isAuxiliary ? indexMap.at(formula) : -1;

    MVAVariableInfo varInfo{
        .hasLowerClip = hasLowerClip,
        .hasUpperClip = hasUpperClip,
        .lowerClipValue = lowerClipValue,
        .upperClipValue = upperClipValue,
        .auxIndex = auxIndex,
    };

    variableInfos_.push_back(varInfo);
    names_.push_back(name);
    indexMap_[name] = nVars_;
    nVars_++;
  };

  int nVars_;

  std::vector<MVAVariableInfo> variableInfos_;
  std::vector<ThreadSafeFunctor<StringObjectFunction<ParticleType>>> functions_;
  std::vector<std::string> formulas_;
  std::vector<std::string> names_;
  std::map<std::string, int> indexMap_;
};

#endif
