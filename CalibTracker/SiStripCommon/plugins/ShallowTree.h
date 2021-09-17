#ifndef Shallow_Tree_h
#define Shallow_Tree_h

/** \class ShallowTree
 * 
 *  Makes a tree out of C++ standard types and vectors of C++ standard types
 *
 *  This class, which is an EDAnalyzer, takes the same "keep" and
 *  "drop" outputCommands parameter as the PoolOutputSource, making a
 *  tree of the selected variables, which it obtains from the EDM
 *  tree.  
 *
 *  See the file python/test_cfg.py for an example configuration.
 *
 *  See the file doc/README for more detailed documentation, including
 *  advantages, disadvantages, and use philosophy.
 *  
 *  \author Burt Betchart - University of Rochester <burton.andrew.betchart@cern.ch>
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>
#include <TTree.h>

class ShallowTree : public edm::one::EDAnalyzer<edm::one::SharedResources> {
private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  template <class T>
  void eat(edm::BranchDescription const* desc) {
    consumes<T>(edm::InputTag(desc->moduleLabel(), desc->productInstanceName()));
  }

  class BranchConnector {
  public:
    virtual ~BranchConnector(){};
    virtual void connect(const edm::Event&) = 0;
  };

  template <class T>
  class TypedBranchConnector : public BranchConnector {
  private:
    std::string ml;   //module label
    std::string pin;  //product instance name
    T object_;
    T* object_ptr_;

  public:
    TypedBranchConnector(edm::BranchDescription const*, std::string, TTree*);
    void connect(const edm::Event&) override;
  };

  edm::Service<TFileService> fs_;
  TTree* tree_;
  std::vector<BranchConnector*> connectors_;

public:
  explicit ShallowTree(const edm::ParameterSet& iConfig);  // : pset(iConfig) {}

  enum LEAFTYPE {
    BOOL = 1,
    BOOL_V,
    SHORT,
    SHORT_V,
    U_SHORT,
    U_SHORT_V,
    INT,
    INT_V,
    U_INT,
    U_INT_V,
    FLOAT,
    FLOAT_V,
    DOUBLE,
    DOUBLE_V,
    LONG,
    LONG_V,
    U_LONG,
    U_LONG_V,
    CHAR,
    CHAR_V,
    U_CHAR,
    U_CHAR_V
  };
};

#endif
