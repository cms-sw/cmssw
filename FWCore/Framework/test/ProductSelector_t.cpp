#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include <memory>

#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Framework/interface/ProductSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

typedef std::vector<edm::BranchDescription const*> VCBDP;

void apply_gs(edm::ProductSelector const& gs,
              VCBDP const&  allbranches,
              std::vector<bool>& results) {

  VCBDP::const_iterator it  = allbranches.begin();
  VCBDP::const_iterator end = allbranches.end();
  for (; it != end; ++it) results.push_back(gs.selected(**it));
}

int doTest(edm::ParameterSet const& params,
             char const* testname,
             VCBDP const&  allbranches,
             std::vector<bool>& expected) {

  edm::ProductSelectorRules gsr(params, "outputCommands", testname);
  edm::ProductSelector gs;
  gs.initialize(gsr, allbranches);
  std::cout << "ProductSelector from "
            << testname
            << ": "
            << gs
            << std::endl;

  std::vector<bool> results;
  apply_gs(gs, allbranches, results);

  int rc = 0;
  if (expected != results) rc = 1;
  if (rc == 1) std::cerr << "FAILURE: " << testname << '\n';
  std::cout << "---" << std::endl;
  return rc;
}

int work() {
  edm::ParameterSet dummyProcessPset;
  dummyProcessPset.registerIt();
  auto processConfiguration = std::make_shared<edm::ProcessConfiguration>();
  processConfiguration->setParameterSetID(dummyProcessPset.id());

  edm::ParameterSet pset;
  pset.registerIt();

  edm::TypeWithDict dummyTypeWithDict;
  int rc = 0;
  // We pretend to have one module, with two products. The products
  // are of the same and, type differ in instance name.
  std::set<edm::ParameterSetID> psetsA;
  edm::ParameterSet modAparams;
  modAparams.addParameter<int>("i", 2112);
  modAparams.addParameter<std::string>("s", "hi");
  modAparams.registerIt();
  psetsA.insert(modAparams.id());

  edm::BranchDescription b1(edm::InEvent, "modA", "PROD", "UglyProdTypeA", "ProdTypeA", "i1",
                            "", pset.id(), dummyTypeWithDict);
  edm::BranchDescription b2(edm::InEvent, "modA", "PROD", "UglyProdTypeA", "ProdTypeA", "i2",
                            "", pset.id(), dummyTypeWithDict);

  // Our second pretend module has only one product, and gives it no
  // instance name.
  std::set<edm::ParameterSetID> psetsB;
  edm::ParameterSet modBparams;
  modBparams.addParameter<double>("d", 2.5);
  modBparams.registerIt();
  psetsB.insert(modBparams.id());

  edm::BranchDescription b3(edm::InEvent, "modB", "HLT", "UglyProdTypeB", "ProdTypeB", "",
                            "", pset.id(), dummyTypeWithDict);

  // Our third pretend is like modA, except it hass processName_ of
  // "USER"

  edm::BranchDescription b4(edm::InEvent, "modA", "USER", "UglyProdTypeA",
                            "ProdTypeA", "i1", "", pset.id(), dummyTypeWithDict);
  edm::BranchDescription b5(edm::InEvent, "modA", "USER", "UglyProdTypeA",
                            "ProdTypeA", "i2", "", pset.id(), dummyTypeWithDict);

  // These are pointers to all the branches that are available. In a
  // framework program, these would come from the ProductRegistry
  // which is used to initialze the OutputModule being configured.
  VCBDP allbranches;
  allbranches.push_back(&b1); // ProdTypeA_modA_i1. (PROD)
  allbranches.push_back(&b2); // ProdTypeA_modA_i2. (PROD)
  allbranches.push_back(&b3); // ProdTypeB_modB_HLT. (no instance name)
  allbranches.push_back(&b4); // ProdTypeA_modA_i1_USER.
  allbranches.push_back(&b5); // ProdTypeA_modA_i2_USER.

  // Test default parameters
  {
    bool wanted[] = { true, true, true, true, true };
    std::vector<bool> expected(wanted, wanted+sizeof(wanted)/sizeof(bool));
    edm::ParameterSet noparams;

    rc += doTest(noparams, "default parameters", allbranches, expected);
  }

  // Keep all branches with instance name i2.
  {
    bool wanted[] = { false, true, false, false, true };
    std::vector<bool> expected(wanted, wanted+sizeof(wanted)/sizeof(bool));

    edm::ParameterSet keep_i2;
    std::string const keep_i2_rule = "keep *_*_i2_*";
    std::vector<std::string> cmds;
    cmds.push_back(keep_i2_rule);
    keep_i2.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    keep_i2.registerIt();

    rc += doTest(keep_i2, "keep_i2 parameters", allbranches, expected);
  }

  // Drop all branches with instance name i2.
  {
    bool wanted[] = { true, false, true, true, false };
    std::vector<bool> expected(wanted, wanted+sizeof(wanted)/sizeof(bool));

    edm::ParameterSet drop_i2;
    std::string const drop_i2_rule1 = "keep *";
    std::string const drop_i2_rule2 = "drop *_*_i2_*";
    std::vector<std::string> cmds;
    cmds.push_back(drop_i2_rule1);
    cmds.push_back(drop_i2_rule2);
    drop_i2.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    drop_i2.registerIt();

    rc += doTest(drop_i2, "drop_i2 parameters", allbranches, expected);
  }

  // Now try dropping all branches with product type "foo". There are
  // none, so all branches should be written.
  {
    bool wanted[] = { true, true, true, true, true };
    std::vector<bool> expected(wanted, wanted+sizeof(wanted)/sizeof(bool));

    edm::ParameterSet drop_foo;
    std::string const drop_foo_rule1 = "keep *_*_*_*"; // same as "keep *"
    std::string const drop_foo_rule2 = "drop foo_*_*_*";
    std::vector<std::string> cmds;
    cmds.push_back(drop_foo_rule1);
    cmds.push_back(drop_foo_rule2);
    drop_foo.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    drop_foo.registerIt();

    rc += doTest(drop_foo, "drop_foo parameters", allbranches, expected);
  }

  // Now try dropping all branches with product type "ProdTypeA".
  {
    bool wanted[] = { false, false, true, false, false };
    std::vector<bool> expected(wanted, wanted+sizeof(wanted)/sizeof(bool));

    edm::ParameterSet drop_ProdTypeA;
    std::string const drop_ProdTypeA_rule1 = "keep *";
    std::string const drop_ProdTypeA_rule2 = "drop ProdTypeA_*_*_*";
    std::vector<std::string> cmds;
    cmds.push_back(drop_ProdTypeA_rule1);
    cmds.push_back(drop_ProdTypeA_rule2);
    drop_ProdTypeA.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    drop_ProdTypeA.registerIt();

    rc += doTest(drop_ProdTypeA,
                 "drop_ProdTypeA",
                 allbranches, expected);
  }

  // Keep only branches with instance name 'i1', from Production.
  {
    bool wanted[] = { true, false, false, false, false };
    std::vector<bool> expected(wanted, wanted+sizeof(wanted)/sizeof(bool));

    edm::ParameterSet keep_i1prod;
    std::string const keep_i1prod_rule = "keep *_*_i1_PROD";
    std::vector<std::string> cmds;
    cmds.push_back(keep_i1prod_rule);
    keep_i1prod.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    keep_i1prod.registerIt();

    rc += doTest(keep_i1prod,
                 "keep_i1prod",
                 allbranches, expected);
  }

  // First say to keep everything,  then  to drop everything, then  to
  // keep it again. The end result should be to keep everything.
  {
    bool wanted[] = { true, true, true, true, true };
    std::vector<bool> expected(wanted, wanted+sizeof(wanted)/sizeof(bool));

    edm::ParameterSet indecisive;
    std::string const indecisive_rule1 = "keep *";
    std::string const indecisive_rule2 = "drop *";
    std::string const indecisive_rule3 = "keep *";
    std::vector<std::string> cmds;
    cmds.push_back(indecisive_rule1);
    cmds.push_back(indecisive_rule2);
    cmds.push_back(indecisive_rule3);
    indecisive.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    indecisive.registerIt();

    rc += doTest(indecisive,
                 "indecisive",
                 allbranches, expected);
  }

  // Keep all things, bu drop all things from modA, but later keep all
  // things from USER.
  {
    bool wanted[] = { false, false, true, true, true };
    std::vector<bool> expected(wanted, wanted+sizeof(wanted)/sizeof(bool));

    edm::ParameterSet params;
    std::string const rule1 = "keep *";
    std::string const rule2 = "drop *_modA_*_*";
    std::string const rule3 = "keep *_*_*_USER";
    std::vector<std::string> cmds;
    cmds.push_back(rule1);
    cmds.push_back(rule2);
    cmds.push_back(rule3);
    params.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    params.registerIt();

    rc += doTest(params,
                 "drop_modA_keep_user",
                 allbranches, expected);
  }

  // Exercise the wildcards * and ?
  {
    bool wanted[] = { true, true, true, false, false };
    std::vector<bool> expected(wanted, wanted+sizeof(wanted)/sizeof(bool));

    edm::ParameterSet params;
    std::string const rule1 = "drop *";
    std::string const rule2 = "keep Pr*A_m?dA_??_P?O*";
    std::string const rule3 = "keep *?*?***??*????*?***_??***?__*?***T";
    std::vector<std::string> cmds;
    cmds.push_back(rule1);
    cmds.push_back(rule2);
    cmds.push_back(rule3);
    params.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    params.registerIt();

    rc += doTest(params,
                 "excercise wildcards1",
                 allbranches, expected);
  }

  {
    // Now try an illegal specification: not starting with 'keep' or 'drop'
    try {
        edm::ParameterSet bad;
        std::string const bad_rule = "beep *_*_i2_*";
        std::vector<std::string> cmds;
        cmds.push_back(bad_rule);
        bad.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
        bad.registerIt();
        edm::ProductSelectorRules gsr(bad, "outputCommands", "ProductSelectorTest");
        edm::ProductSelector gs;
        gs.initialize(gsr, allbranches);
        std::cerr << "Failed to throw required exception\n";
        rc += 1;
    }
    catch (edm::Exception const& x) {
        // OK, we should be here... now check exception type
        assert (x.categoryCode() == edm::errors::Configuration);
    }
    catch (...) {
        std::cerr << "Wrong exception type\n";
        rc += 1;
    }
  }
  return rc;
}

int main() {
  int rc = 0;
  try {
      rc = work();
  }
  catch (edm::Exception& x) {
      std::cerr << "edm::Exception caught:\n" << x << '\n';
      rc = 1;
  }
  catch (...) {
      std::cerr << "Unknown exception caught\n";
      rc = 2;
  }
  return rc;
}
