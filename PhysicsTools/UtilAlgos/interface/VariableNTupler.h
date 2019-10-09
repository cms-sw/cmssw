#ifndef VariableNtupler_NTupler_H
#define VariableNtupler_NTupler_H

#include "PhysicsTools/UtilAlgos/interface/VariableHelper.h"
//#include "PhysicsTools/UtilAlgos/interface/UpdaterService.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/ProducesCollector.h"

//#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"

#include "PhysicsTools/UtilAlgos/interface/NTupler.h"

#include <algorithm>

class VariableNTupler : public NTupler {
public:
  VariableNTupler(const edm::ParameterSet& iConfig) {
    ownTheTree_ = false;
    edm::ParameterSet variablePSet = iConfig.getParameter<edm::ParameterSet>("variablesPSet");
    if (variablePSet.getParameter<bool>("allVariables")) {
      VariableHelper::iterator v = edm::Service<VariableHelperService>()->get().begin();
      VariableHelper::iterator v_end = edm::Service<VariableHelperService>()->get().end();
      for (; v != v_end; ++v) {
        leaves_[v->second->name()] = v->second;
      }
    } else {
      std::vector<std::string> leaves = variablePSet.getParameter<std::vector<std::string> >("leaves");
      for (uint i = 0; i != leaves.size(); ++i) {
        leaves_[leaves[i]] = edm::Service<VariableHelperService>()->get().variable(leaves[i]);
      }
    }
    if (variablePSet.exists("useTFileService"))
      useTFileService_ = variablePSet.getParameter<bool>("useTFileService");
    else
      useTFileService_ = iConfig.getParameter<bool>("useTFileService");

    if (useTFileService_) {
      if (variablePSet.exists("treeName"))
        treeName_ = variablePSet.getParameter<std::string>("treeName");
      else
        treeName_ = iConfig.getParameter<std::string>("treeName");
    }
  }

  uint registerleaves(edm::ProducesCollector producesCollector) override {
    uint nLeaves = 0;
    if (useTFileService_) {
      //loop the leaves registered
      nLeaves = leaves_.size();
      // make arrays of pointer to the actual values
      dataHolder_ = new double[nLeaves];
      iterator i = leaves_.begin();
      iterator i_end = leaves_.end();
      edm::Service<TFileService> fs;
      if (ownTheTree_) {
        ownTheTree_ = true;
        tree_ = fs->make<TTree>(treeName_.c_str(), "VariableNTupler tree");
      } else {
        TObject* object = fs->file().Get(treeName_.c_str());
        if (!object) {
          ownTheTree_ = true;
          tree_ = fs->make<TTree>(treeName_.c_str(), "VariableNTupler tree");
        } else {
          tree_ = dynamic_cast<TTree*>(object);
          if (!tree_) {
            ownTheTree_ = true;
            tree_ = fs->make<TTree>(treeName_.c_str(), "VariableNTupler tree");
          } else
            ownTheTree_ = false;
        }
      }
      uint iInDataHolder = 0;
      for (; i != i_end; ++i, ++iInDataHolder) {
        tree_->Branch(i->first.c_str(), &(dataHolder_[iInDataHolder]), (i->first + "/D").c_str());
      }
    } else {
      //loop the leaves registered
      iterator i = leaves_.begin();
      iterator i_end = leaves_.end();
      for (; i != i_end; ++i) {
        nLeaves++;
        std::string lName(i->first);
        std::replace(lName.begin(), lName.end(), '_', '0');
        producesCollector.produces<double>(lName).setBranchAlias(i->first);
      }
    }
    return nLeaves;
  }

  void fill(edm::Event& iEvent) override {
    if (useTFileService_) {
      //fill the data holder
      iterator i = leaves_.begin();
      iterator i_end = leaves_.end();
      uint iInDataHolder = 0;
      for (; i != i_end; ++i, ++iInDataHolder) {
        dataHolder_[iInDataHolder] = (*i->second)(iEvent);
      }
      //fill into root;
      if (ownTheTree_) {
        tree_->Fill();
      }
    } else {
      //other leaves
      iterator i = leaves_.begin();
      iterator i_end = leaves_.end();
      for (; i != i_end; ++i) {
        auto leafValue = std::make_unique<double>((*i->second)(iEvent));
        std::string lName(i->first);
        std::replace(lName.begin(), lName.end(), '_', '0');
        iEvent.put(std::move(leafValue), lName);
      }
    }
  }
  void callBack() {}

protected:
  typedef std::map<std::string, const CachingVariable*>::iterator iterator;
  std::map<std::string, const CachingVariable*> leaves_;

  bool ownTheTree_;
  std::string treeName_;
  double* dataHolder_;
};

#endif
