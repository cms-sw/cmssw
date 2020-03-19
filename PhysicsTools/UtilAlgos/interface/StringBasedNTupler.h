#ifndef StringBasedNTupler_NTupler_H
#define StringBasedNTupler_NTupler_H

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

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/UtilAlgos/interface/InputTagDistributor.h"

#include "DataFormats/PatCandidates/interface/PFParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

//#define StringBasedNTuplerPrecision float;

#include <memory>
#include <string>
#include <sstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// LHE Event
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

class TreeBranch {
public:
  TreeBranch() : class_(""), expr_(""), order_(""), selection_(""), maxIndexName_(""), branchAlias_("") {}
  TreeBranch(
      std::string C, edm::InputTag S, std::string E, std::string O, std::string SE, std::string Mi, std::string Ba)
      : class_(C), src_(S), expr_(E), order_(O), selection_(SE), maxIndexName_(Mi), branchAlias_(Ba) {
    branchTitle_ = E + " calculated on " + C + " object from " + S.encode();
    if (!O.empty())
      branchTitle_ += " ordered according to " + O;
    if (!SE.empty())
      branchTitle_ += " selecting on " + SE;
    edm::LogInfo("TreeBranch") << "the branch with alias: " << branchAlias_ << " corresponds to: " << branchTitle_;
  }

  const std::string& className() const { return class_; }
  const edm::InputTag& src() const { return src_; }
  const std::string& expr() const { return expr_; }
  const std::string& order() const { return order_; }
  const std::string& selection() const { return selection_; }
  const std::string& maxIndexName() const { return maxIndexName_; }
  const std::string branchName() const {
    std::string name(branchAlias_);
    std::replace(name.begin(), name.end(), '_', '0');
    return std::string(name);
  }
  const std::string& branchAlias() const { return branchAlias_; }
  const std::string& branchTitle() const { return branchTitle_; }
  typedef std::unique_ptr<std::vector<float> > value;
  value branch(const edm::Event& iEvent);

  std::vector<float>** dataHolderPtrAdress() { return &dataHolderPtr_; }
  std::vector<float>* dataHolderPtr() { return dataHolderPtr_; }
  void assignDataHolderPtr(std::vector<float>* data) { dataHolderPtr_ = data; }

private:
  std::string class_;
  edm::InputTag src_;
  std::string expr_;
  std::string order_;
  std::string selection_;
  std::string maxIndexName_;
  std::string branchAlias_;
  std::string branchTitle_;

  std::vector<float>* dataHolderPtr_;
};

template <typename Object>
class StringLeaveHelper {
public:
  typedef TreeBranch::value value;
  value operator()() { return std::move(value_); }

  StringLeaveHelper(const TreeBranch& B, const edm::Event& iEvent) {
    const float defaultValue = 0.;
    //    grab the object
    edm::Handle<Object> oH;
    iEvent.getByLabel(B.src(), oH);
    //empty vector if product not found
    if (oH.failedToGet()) {
      edm::LogError("StringBranchHelper") << "cannot open: " << B.src();
      value_.reset(new std::vector<float>(0));
    } else {
      //parser for the object expression
      StringObjectFunction<Object> expr(B.expr());
      //allocate enough memory for the data holder
      value_.reset(new std::vector<float>(1));
      try {
        (*value_)[0] = (expr)(*oH);
      } catch (...) {
        LogDebug("StringLeaveHelper") << "could not evaluate expression: " << B.expr()
                                      << " on class: " << B.className();
        (*value_)[0] = defaultValue;
      }
    }
  }

private:
  value value_;
};

template <typename Object, typename Collection = std::vector<Object> >
class StringBranchHelper {
public:
  typedef TreeBranch::value value;
  value operator()() { return std::move(value_); }

  StringBranchHelper(const TreeBranch& B, const edm::Event& iEvent) {
    const float defaultValue = 0.;

    //    grab the collection
    edm::Handle<Collection> oH;
    iEvent.getByLabel(B.src(), oH);

    //empty vector if product not found
    if (oH.failedToGet()) {
      if (!(iEvent.isRealData() && B.className() == "reco::GenParticle")) {  //don't output genparticle error in data
        edm::LogError("StringBranchHelper") << "cannot open: " << B.src() << "  " << B.className();
      }
      value_.reset(new std::vector<float>());
    } else {
      //parser for the object expression
      StringObjectFunction<Object> expr(B.expr());
      //allocate enough memory for the data holder
      value_.reset(new std::vector<float>());
      value_->reserve(oH->size());

      StringCutObjectSelector<Object>* selection = nullptr;
      if (!B.selection().empty()) {
        //std::cout<<"trying to get to a selection"<<std::endl;
        selection = new StringCutObjectSelector<Object>(B.selection());
        //std::cout<<"got the objet"<<std::endl;
      }
      uint i_end = oH->size();
      //sort things first if requested
      if (!B.order().empty()) {
        StringObjectFunction<Object> order(B.order());
        // allocate a vector of pointers (we are using view) to be sorted
        std::vector<const Object*> copyToSort(oH->size());
        for (uint i = 0; i != i_end; ++i)
          copyToSort[i] = &(*oH)[i];
        std::sort(copyToSort.begin(), copyToSort.end(), sortByStringFunction<Object>(&order));
        //then loop and fill
        for (uint i = 0; i != i_end; ++i) {
          //try and catch is necessary because ...
          try {
            if (selection && !((*selection)(*(copyToSort)[i])))
              continue;
            value_->push_back((expr)(*(copyToSort)[i]));
          } catch (...) {
            LogDebug("StringBranchHelper")
                << "with sorting. could not evaluate expression: " << B.expr() << " on class: " << B.className();
            value_->push_back(defaultValue);  //push a default value to not change the indexing
          }
        }
      } else {
        //actually fill the vector of values
        for (uint i = 0; i != i_end; ++i) {
          //try and catch is necessary because ...
          try {
            if (selection && !((*selection)((*oH)[i])))
              continue;
            value_->push_back((expr)((*oH)[i]));
          } catch (...) {
            LogDebug("StringBranchHelper")
                << "could not evaluate expression: " << B.expr() << " on class: " << B.className();
            value_->push_back(defaultValue);  //push a default value to not change the indexing
          }
        }
      }
      if (selection)
        delete selection;
    }
  }

private:
  value value_;
};

class StringBasedNTupler : public NTupler {
public:
  StringBasedNTupler(const edm::ParameterSet& iConfig) {
    edm::ParameterSet branchesPSet = iConfig.getParameter<edm::ParameterSet>("branchesPSet");
    std::vector<std::string> branches;
    branchesPSet.getParameterSetNames(branches);
    const std::string separator = branchesPSet.getUntrackedParameter<std::string>("separator", ":");
    for (uint b = 0; b != branches.size(); ++b) {
      edm::ParameterSet bPSet = branchesPSet.getParameter<edm::ParameterSet>(branches[b]);
      std::string className = "";
      if (bPSet.exists("class"))
        className = bPSet.getParameter<std::string>("class");
      else
        className = bPSet.getParameter<std::string>("Class");
      edm::InputTag src = edm::Service<InputTagDistributorService>()->retrieve("src", bPSet);
      edm::ParameterSet leavesPSet = bPSet.getParameter<edm::ParameterSet>("leaves");
      std::string order = "";
      if (bPSet.exists("order"))
        order = bPSet.getParameter<std::string>("order");
      std::string selection = "";
      if (bPSet.exists("selection"))
        selection = bPSet.getParameter<std::string>("selection");
      // do it one by one with configuration [string x = "x"]
      std::vector<std::string> leaves = leavesPSet.getParameterNamesForType<std::string>();
      std::string maxName = "N" + branches[b];
      for (uint l = 0; l != leaves.size(); ++l) {
        std::string leave_expr = leavesPSet.getParameter<std::string>(leaves[l]);
        std::string branchAlias = branches[b] + "_" + leaves[l];

        //add a branch manager for this expression on this collection
        branches_[maxName].push_back(TreeBranch(className, src, leave_expr, order, selection, maxName, branchAlias));
      }  //loop the provided leaves

      //do it once with configuration [vstring vars = { "x:x" ,... } ] where ":"=separator
      if (leavesPSet.exists("vars")) {
        std::vector<std::string> leavesS = leavesPSet.getParameter<std::vector<std::string> >("vars");
        for (uint l = 0; l != leavesS.size(); ++l) {
          uint sep = leavesS[l].find(separator);
          std::string name = leavesS[l].substr(0, sep);
          //removes spaces from the variable name
          /*uint*/ int space = name.find(" ");
          while (space != -1 /*std::string::npos*/) {
            std::string first = name.substr(0, space);
            std::string second = name.substr(space + 1);
            name = first + second;
            space = name.find(" ");
          }
          std::string expr = leavesS[l].substr(sep + 1);
          std::string branchAlias = branches[b] + "_" + name;

          //add a branch manager for this expression on this collection
          branches_[maxName].push_back(TreeBranch(className, src, expr, order, selection, maxName, branchAlias));
        }
      }

    }  //loop the provided branches

    ev_ = new uint64_t;
    run_ = new uint;
    lumiblock_ = new uint;
    experimentType_ = new uint;
    bunchCrossing_ = new uint;
    orbitNumber_ = new uint;
    weight_ = new float;
    model_params_ = new std::string;

    if (branchesPSet.exists("useTFileService"))
      useTFileService_ = branchesPSet.getParameter<bool>("useTFileService");
    else
      useTFileService_ = iConfig.getParameter<bool>("useTFileService");

    if (useTFileService_) {
      if (branchesPSet.exists("treeName")) {
        treeName_ = branchesPSet.getParameter<std::string>("treeName");
        ownTheTree_ = true;
      } else {
        treeName_ = iConfig.getParameter<std::string>("treeName");
        ownTheTree_ = false;
      }
    }
  }

  uint registerleaves(edm::ProducesCollector producesCollector) override {
    uint nLeaves = 0;

    if (useTFileService_) {
      edm::Service<TFileService> fs;
      if (ownTheTree_) {
        ownTheTree_ = true;
        tree_ = fs->make<TTree>(treeName_.c_str(), "StringBasedNTupler tree");
      } else {
        TObject* object = fs->file().Get(treeName_.c_str());
        if (!object) {
          ownTheTree_ = true;
          tree_ = fs->make<TTree>(treeName_.c_str(), "StringBasedNTupler tree");
        } else {
          tree_ = dynamic_cast<TTree*>(object);
          if (!tree_) {
            ownTheTree_ = true;
            tree_ = fs->make<TTree>(treeName_.c_str(), "StringBasedNTupler tree");
          } else
            ownTheTree_ = false;
        }
      }

      //reserve memory for the indexes
      indexDataHolder_ = new uint[branches_.size()];
      // loop the automated leafer
      Branches::iterator iB = branches_.begin();
      Branches::iterator iB_end = branches_.end();
      uint indexOfIndexInDataHolder = 0;
      for (; iB != iB_end; ++iB, ++indexOfIndexInDataHolder) {
        //create a branch for the index: an integer
        tree_->Branch(iB->first.c_str(), &(indexDataHolder_[indexOfIndexInDataHolder]), (iB->first + "/i").c_str());
        //loop on the "leaves"
        std::vector<TreeBranch>::iterator iL = iB->second.begin();
        std::vector<TreeBranch>::iterator iL_end = iB->second.end();
        for (; iL != iL_end; ++iL) {
          TreeBranch& b = *iL;
          //create a branch for the leaves: vector of floats
          TBranch* br = tree_->Branch(b.branchAlias().c_str(), "std::vector<float>", iL->dataHolderPtrAdress());
          br->SetTitle(b.branchTitle().c_str());
          nLeaves++;
        }
      }

      //extra leaves for event info.
      tree_->Branch("run", run_, "run/i");
      tree_->Branch("event", ev_, "event/l");
      tree_->Branch("lumiblock", lumiblock_, "lumiblock/i");
      tree_->Branch("experimentType", experimentType_, "experimentType/i");
      tree_->Branch("bunchCrossing", bunchCrossing_, "bunchCrossing/i");
      tree_->Branch("orbitNumber", orbitNumber_, "orbitNumber/i");
      tree_->Branch("weight", weight_, "weight/f");
      tree_->Branch("model_params", &model_params_);

    } else {
      // loop the automated leafer
      Branches::iterator iB = branches_.begin();
      Branches::iterator iB_end = branches_.end();
      for (; iB != iB_end; ++iB) {
        //the index. should produce it only once
        // a simple uint for the index
        producesCollector.produces<uint>(iB->first).setBranchAlias(iB->first);
        std::vector<TreeBranch>::iterator iL = iB->second.begin();
        std::vector<TreeBranch>::iterator iL_end = iB->second.end();
        for (; iL != iL_end; ++iL) {
          TreeBranch& b = *iL;
          //a vector of float for each leave
          producesCollector.produces<std::vector<float> >(b.branchName()).setBranchAlias(b.branchAlias());
          nLeaves++;
        }
      }
    }
    return nLeaves;
  }

  void fill(edm::Event& iEvent) override {
    //    if (!edm::Service<UpdaterService>()->checkOnce("StringBasedNTupler::fill")) return;
    //well if you do that, you cannot have two ntupler of the same type in the same job...

    if (useTFileService_) {
      // loop the automated leafer
      Branches::iterator iB = branches_.begin();
      Branches::iterator iB_end = branches_.end();
      uint indexOfIndexInDataHolder = 0;
      for (; iB != iB_end; ++iB, ++indexOfIndexInDataHolder) {
        std::vector<TreeBranch>::iterator iL = iB->second.begin();
        std::vector<TreeBranch>::iterator iL_end = iB->second.end();
        uint maxS = 0;
        for (; iL != iL_end; ++iL) {
          TreeBranch& b = *iL;
          // grab the vector of values from the interpretation of expression for the associated collection
          std::unique_ptr<std::vector<float> > branch(b.branch(iEvent));
          // calculate the maximum index size.
          if (branch->size() > maxS)
            maxS = branch->size();
          // transfer of (no copy) pointer to the vector of float from the std::unique_ptr to the tree data pointer
          b.assignDataHolderPtr(branch.release());
          // for memory tracing, object b is holding the data (not std::unique_ptr) and should delete it for each event (that's not completely optimum)
        }
        //assigne the maximum vector size for this collection
        indexDataHolder_[indexOfIndexInDataHolder] = maxS;
      }

      //fill event info.
      *run_ = iEvent.id().run();
      *ev_ = iEvent.id().event();
      //      *lumiblock_ = iEvent.id().luminosityBlock();
      *lumiblock_ = iEvent.luminosityBlock();
      *experimentType_ = iEvent.experimentType();
      *bunchCrossing_ = iEvent.bunchCrossing();
      *orbitNumber_ = iEvent.orbitNumber();

      *weight_ = 1;
      if (!iEvent.isRealData()) {
        edm::Handle<GenEventInfoProduct> wgeneventinfo;
        iEvent.getByLabel("generator", wgeneventinfo);
        *weight_ = wgeneventinfo->weight();
      }

      typedef std::vector<std::string>::const_iterator comments_const_iterator;
      //      using namespace edm;

      edm::Handle<LHEEventProduct> product;
      *model_params_ = "NULL";
      if (iEvent.getByLabel("source", product)) {
        comments_const_iterator c_begin = product->comments_begin();
        comments_const_iterator c_end = product->comments_end();

        for (comments_const_iterator cit = c_begin; cit != c_end; ++cit) {
          size_t found = (*cit).find("model");
          if (found != std::string::npos) {
            //std::cout << *cit << std::endl;
            *model_params_ = *cit;
          }
        }
      }

      if (ownTheTree_) {
        tree_->Fill();
      }
    } else {
      // loop the automated leafer
      Branches::iterator iB = branches_.begin();
      Branches::iterator iB_end = branches_.end();
      for (; iB != iB_end; ++iB) {
        std::vector<TreeBranch>::iterator iL = iB->second.begin();
        std::vector<TreeBranch>::iterator iL_end = iB->second.end();
        uint maxS = 0;
        for (; iL != iL_end; ++iL) {
          TreeBranch& b = *iL;
          std::unique_ptr<std::vector<float> > branch(b.branch(iEvent));
          if (branch->size() > maxS)
            maxS = branch->size();
          iEvent.put(std::move(branch), b.branchName());
        }
        //index should be put only once per branch. doe not really mattter for edm root files
        iEvent.put(std::make_unique<uint>(maxS), iB->first);
      }
    }
  }

  void callBack() {
    if (useTFileService_) {
      Branches::iterator iB = branches_.begin();
      Branches::iterator iB_end = branches_.end();
      //de-allocate memory now: allocated in branch(...) and released to the pointer.
      for (; iB != iB_end; ++iB) {
        std::vector<TreeBranch>::iterator iL = iB->second.begin();
        std::vector<TreeBranch>::iterator iL_end = iB->second.end();
        for (; iL != iL_end; ++iL) {
          TreeBranch& b = *iL;
          delete b.dataHolderPtr();
        }
      }
    }
  }

  ~StringBasedNTupler() override {
    delete indexDataHolder_;
    delete ev_;
    delete run_;
    delete lumiblock_;
    delete experimentType_;
    delete bunchCrossing_;
    delete orbitNumber_;
    delete weight_;
    delete model_params_;
  }

protected:
  typedef std::map<std::string, std::vector<TreeBranch> > Branches;
  Branches branches_;

  bool ownTheTree_;
  std::string treeName_;
  uint* indexDataHolder_;

  //event info
  uint64_t* ev_;
  uint* run_;
  uint* lumiblock_;
  uint* experimentType_;
  uint* bunchCrossing_;
  uint* orbitNumber_;
  float* weight_;
  std::string* model_params_;
};

#endif
