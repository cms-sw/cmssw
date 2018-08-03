#ifndef RecoEgamma_EgammaTools_MVAVariableManager_H
#define RecoEgamma_EgammaTools_MVAVariableManager_H

#include "CommonTools/Utils/interface/StringObjectFunction.h"                                                        
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"

#include <fstream>

template <class ParticleType>
class MVAVariableManager {

  public:
    MVAVariableManager()
      : nVars_ (0)
      , nHelperVars_ (0)
      , nGlobalVars_ (0)
    {
    };

    MVAVariableManager(const std::string &variableDefinitionFileName) {
        init(variableDefinitionFileName);
    };

    int init(const std::string &variableDefinitionFileName)
    {
        nVars_ = 0;
        nHelperVars_ = 0;
        nGlobalVars_ = 0;

        variableInfos_.clear();
        functions_.clear();
        formulas_.clear();
        names_.clear();
        helperInputTags_.clear();
        globalInputTags_.clear();

        edm::FileInPath variableDefinitionFileEdm(variableDefinitionFileName);
        std::ifstream file(variableDefinitionFileEdm.fullPath());

        std::string name, formula, upper, lower;
        while( true ) {
            file >> name;
            if (name.find("#") != std::string::npos) {
                file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                continue;
            }
            file >> formula >> lower >> upper;
            if (file.eof()) {
                break;
            }
            addVariable(name, formula, lower, upper);
        }
        return nVars_;
    };

    int getVarIndex(const std::string &name)
    {
        std::map<std::string,int>::iterator it = indexMap_.find(name);
        if (it == indexMap_.end()) {
            return -1;
        } else {
            return it->second;
        }
    }

    const std::string& getName(int index) const {
        return names_[index];
    }

    int getNVars() const {
        return nVars_;
    }

    // For edm::EventBase with getByLabel
    float getValue(int index, const edm::Ptr<ParticleType>& ptclPtr, const edm::EventBase& iEvent) const {
        float value;
        MVAVariableInfo varInfo = variableInfos_[index];
        if (varInfo.fromVariableHelper >= 0) {
            edm::Handle<edm::ValueMap<float>> vMap;
            iEvent.getByLabel(edm::InputTag(formulas_[index]), vMap);
            value = (*vMap)[ptclPtr];
        } else if (varInfo.isGlobalVariable >= 0) {
            edm::Handle<double> valueHandle;
            iEvent.getByLabel(edm::InputTag(formulas_[index]), valueHandle);
            value = *valueHandle;
        } else {
            value = functions_[index](*ptclPtr);
        }
        if (varInfo.hasLowerClip && value < varInfo.lowerClipValue) {
            value = varInfo.lowerClipValue;
        }
        if (varInfo.hasUpperClip && value > varInfo.upperClipValue) {
            value = varInfo.upperClipValue;
        }
        return value;
    }

    // For edm::Event where getByToken is possible
    float getValue(int index, const edm::Ptr<ParticleType>& ptclPtr, const edm::Event& iEvent) const {
        float value;
        MVAVariableInfo varInfo = variableInfos_[index];
        if (varInfo.fromVariableHelper >= 0) {
            edm::Handle<edm::ValueMap<float>> vMap;
            iEvent.getByToken(helperTokens_[varInfo.fromVariableHelper], vMap);
            value = (*vMap)[ptclPtr];
        } else if (varInfo.isGlobalVariable >= 0) {
            edm::Handle<double> valueHandle;
            iEvent.getByToken(globalTokens_[varInfo.isGlobalVariable], valueHandle);
            value = *valueHandle;
        } else {
            value = functions_[index](*ptclPtr);
        }
        if (varInfo.hasLowerClip && value < varInfo.lowerClipValue) {
            value = varInfo.lowerClipValue;
        }
        if (varInfo.hasUpperClip && value > varInfo.upperClipValue) {
            value = varInfo.upperClipValue;
        }
        return value;
    }

    void setConsumes(edm::ConsumesCollector&& cc) {
      // All tokens for event content needed by the MVA
      // Tags from the variable helper
      for (auto &tag : helperInputTags_) {
          helperTokens_.push_back(cc.consumes<edm::ValueMap<float>>(tag));
      }
      for (auto &tag : globalInputTags_) {
          globalTokens_.push_back(cc.consumes<double>(tag));
      }
    }

  private:

    struct MVAVariableInfo {
        bool hasLowerClip;
        bool hasUpperClip;
        float lowerClipValue;
        float upperClipValue;
        int fromVariableHelper;
        int isGlobalVariable;
    };

    void addVariable(const std::string &name,      const std::string &formula,
                     const std::string &lowerClip, const std::string &upperClip)
    {
        bool hasLowerClip = lowerClip.find("None") == std::string::npos;
        bool hasUpperClip = upperClip.find("None") == std::string::npos;
        int fromVariableHelper = formula.find("MVAVariableHelper") != std::string::npos ||
                                 formula.find("IDValueMapProducer") != std::string::npos ||
                                 formula.find("egmPhotonIsolation") != std::string::npos;
        float lowerClipValue = hasLowerClip ? (float)::atof(lowerClip.c_str()) : 0.;
        float upperClipValue = hasUpperClip ? (float)::atof(upperClip.c_str()) : 0.;

        // *Rho* is the only global variable used ever, so its hardcoded...
        int isGlobalVariable = formula.find("Rho") != std::string::npos;

        if ( !(fromVariableHelper || isGlobalVariable) ) {
            functions_.push_back(StringObjectFunction<ParticleType>(formula));
        } else {
            // Push back a dummy function since we won't use the
            // StringObjectFunction to evaluate a variable form the helper or a
            // global variable
            functions_.push_back(StringObjectFunction<ParticleType>("pt"));
        }

        formulas_.push_back(formula);
        if (fromVariableHelper) {
            helperInputTags_.push_back(edm::InputTag(formula));
        }
        if (isGlobalVariable) {
            globalInputTags_.push_back(edm::InputTag(formula));
        }

        // Switch from bool to int, corresponding to the token index
        fromVariableHelper = fromVariableHelper ? nHelperVars_++ : -1;
        isGlobalVariable   = isGlobalVariable ? nGlobalVars_++ : - 1;

        MVAVariableInfo varInfo = {
            .hasLowerClip       = hasLowerClip,
            .hasUpperClip       = hasUpperClip,
            .lowerClipValue     = lowerClipValue,
            .upperClipValue     = upperClipValue,
            .fromVariableHelper = fromVariableHelper,
            .isGlobalVariable   = isGlobalVariable
        };
        variableInfos_.push_back(varInfo);
        names_.push_back(name);
        indexMap_[name] = nVars_;
        nVars_++;
    };

    int nVars_;
    int nHelperVars_;
    int nGlobalVars_;

    std::vector<MVAVariableInfo> variableInfos_;
    std::vector<StringObjectFunction<ParticleType>> functions_;
    std::vector<std::string> formulas_;
    std::vector<std::string> names_;
    std::map<std::string, int> indexMap_;

    // To store the MVAVariableHelper input tags needed for the variables in this container
    std::vector<edm::InputTag> helperInputTags_;
    std::vector<edm::InputTag> globalInputTags_;

    // Tokens
    std::vector<edm::EDGetToken> helperTokens_;
    std::vector<edm::EDGetToken> globalTokens_;
};

#endif
