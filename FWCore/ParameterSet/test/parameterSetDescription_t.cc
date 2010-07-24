
// Test code for the ParameterSetDescription and ParameterDescription
// classes.

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionBase.h"
#include "FWCore/ParameterSet/interface/IfExistsDescription.h"
#include "FWCore/ParameterSet/interface/AllowedLabelsDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/MinimalEventID.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <vector>
#include <string>
#include <cassert>
#include <iostream>

namespace testParameterSetDescription {

  void testDesc(edm::ParameterDescriptionNode const& node,
                edm::ParameterSetDescription const& psetDesc,
                edm::ParameterSet & pset,
                bool exists,
                bool validates) {
    assert(node.exists(pset) == exists);
    assert(node.partiallyExists(pset) == exists);
    assert(node.howManyXORSubNodesExist(pset) == (exists ? 1 : 0));
    if (validates) {
      psetDesc.validate(pset);
    }
    else {
      try {
        psetDesc.validate(pset);
        assert(0);
      }
      catch(edm::Exception) {
        // There should be an exception
      }
    }
  }

  void testWildcards() {

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<int> w("*", edm::RequireZeroOrMore, true);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<int>("x", 1);
      testDesc(w, set, pset, true, true);
      pset.addParameter<int>("y", 1);
      testDesc(w, set, pset, true, true);
      pset.addParameter<unsigned>("z", 1);
      testDesc(w, set, pset, true, false);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<unsigned> w("*", edm::RequireExactlyOne, false);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, false);
      pset.addUntrackedParameter<unsigned>("x", 1);
      testDesc(w, set, pset, true, true);
      pset.addUntrackedParameter<unsigned>("y", 1);
      testDesc(w, set, pset, false, false);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<unsigned> w("*", edm::RequireExactlyOne, false);
      set.addOptionalNode(w, false);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, true);
      pset.addUntrackedParameter<unsigned>("x", 1);
      testDesc(w, set, pset, true, true);
      pset.addUntrackedParameter<unsigned>("y", 1);
      testDesc(w, set, pset, false, false);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<unsigned> w("*", edm::RequireAtLeastOne, false);
      set.addOptionalNode(w, false);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, true);
      pset.addUntrackedParameter<unsigned>("x", 1);
      testDesc(w, set, pset, true, true);
      pset.addUntrackedParameter<unsigned>("y", 1);
      testDesc(w, set, pset, true, true);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<double> w("*", edm::RequireAtLeastOne, true);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, false);
      pset.addParameter<double>("x", 1);
      testDesc(w, set, pset, true, true);
      pset.addParameter<double>("y", 1);
      testDesc(w, set, pset, true, true);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<double> w("*", edm::RequireAtLeastOne, true);
      set.addNode(w);
      set.add<int>("testTypeChecking1", 11);
      try {
        set.add<double>("testTypeChecking2", 11.0);
        assert(0);
      }
        catch(edm::Exception) {
        // There should be an exception
      }
    }

    try {
      edm::ParameterWildcard<int> wrong("a*", edm::RequireZeroOrMore, true);
      assert(0);
    }
      catch(edm::Exception) {
      // There should be an exception
    }

    edm::ParameterSet nestedPset;
    nestedPset.addUntrackedParameter<unsigned>("n1", 1);

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<edm::ParameterSetDescription> w("*", edm::RequireZeroOrMore, true);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested1", nestedPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested2", nestedPset);
      testDesc(w, set, pset, true, true);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<edm::ParameterSetDescription> w(std::string("*"), edm::RequireZeroOrMore, true);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested1", nestedPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested2", nestedPset);
      testDesc(w, set, pset, true, true);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<edm::ParameterSetDescription> w("*", edm::RequireZeroOrMore, true,
                                                             edm::ParameterSetDescription());
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested1", nestedPset);
      testDesc(w, set, pset, true, false);
      pset.addParameter<edm::ParameterSet>("nested2", nestedPset);
      testDesc(w, set, pset, true, false);
    }

    edm::ParameterSetDescription nestedDesc;
    nestedDesc.addUntracked<unsigned>("n1", 1);

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<edm::ParameterSetDescription> w("*", edm::RequireZeroOrMore, true,
                                                             nestedDesc);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested1", nestedPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested2", nestedPset);
      testDesc(w, set, pset, true, true);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<edm::ParameterSetDescription> w(std::string("*"), edm::RequireExactlyOne, true,
                                                             nestedDesc);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, false);
      pset.addParameter<edm::ParameterSet>("nested1", nestedPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested2", nestedPset);
      testDesc(w, set, pset, false, false);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<edm::ParameterSetDescription> w("*", edm::RequireAtLeastOne, true,
                                                             nestedDesc);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, false);
      pset.addParameter<edm::ParameterSet>("nested1", nestedPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested2", nestedPset);
      testDesc(w, set, pset, true, true);
    }

    std::vector<edm::ParameterSet> nestedVPset;
    edm::ParameterSet vectorElement;
    vectorElement.addUntrackedParameter<unsigned>("n11", 1);
    nestedVPset.push_back(vectorElement);
    nestedVPset.push_back(vectorElement);

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<std::vector<edm::ParameterSet> > w("*", edm::RequireZeroOrMore, true);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet> >("nested1", nestedVPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet> >("nested2", nestedVPset);
      testDesc(w, set, pset, true, true);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<std::vector<edm::ParameterSet> > w(std::string("*"), edm::RequireZeroOrMore, true);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet> >("nested1", nestedVPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet> >("nested2", nestedVPset);
      testDesc(w, set, pset, true, true);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<std::vector<edm::ParameterSet> > w("*", edm::RequireZeroOrMore, true,
                                                                edm::ParameterSetDescription());
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet> >("nested1", nestedVPset);
      testDesc(w, set, pset, true, false);
      pset.addParameter<std::vector<edm::ParameterSet> >("nested2", nestedVPset);
      testDesc(w, set, pset, true, false);
    }

    edm::ParameterSetDescription descElement;
    descElement.addUntracked<unsigned>("n11", 1);
    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<std::vector<edm::ParameterSet> > w("*", edm::RequireZeroOrMore, true,
                                                                descElement);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet> >("nested1", nestedVPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet> >("nested2", nestedVPset);
      testDesc(w, set, pset, true, true);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<std::vector<edm::ParameterSet> > w(std::string("*"), edm::RequireExactlyOne, true,
                                                                descElement);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, false);
      pset.addParameter<std::vector<edm::ParameterSet> >("nested1", nestedVPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet> >("nested2", nestedVPset);
      testDesc(w, set, pset, false, false);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<std::vector<edm::ParameterSet> > w("*", edm::RequireAtLeastOne, true,
                                                                descElement);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, false);
      pset.addParameter<std::vector<edm::ParameterSet> >("nested1", nestedVPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet> >("nested2", nestedVPset);
      testDesc(w, set, pset, true, true);
    }

    return;
  }

// ---------------------------------------------------------------------------------

  void testSwitch() {

    // Duplicate case values not allowed
    edm::ParameterSetDescription psetDesc;
    try {
      psetDesc.ifValue( edm::ParameterDescription<int>("oiswitch", 1, true),
                        0 >> edm::ParameterDescription<int>("oivalue", 100, true) or
                        1 >> (edm::ParameterDescription<double>("oivalue1", 101.0, true) and
                              edm::ParameterDescription<double>("oivalue2", 101.0, true)) or
                        1 >> edm::ParameterDescription<std::string>("oivalue", "102", true));
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // Types used in case parameters cannot duplicate type already used in a wildcard
    edm::ParameterSetDescription psetDesc1;
    edm::ParameterWildcard<double> w("*", edm::RequireAtLeastOne, true);
    psetDesc1.addNode(w);

    try {
      psetDesc1.ifValue( edm::ParameterDescription<int>("oiswitch", 1, true),
                         0 >> edm::ParameterDescription<int>("oivalue", 100, true) or
                         1 >> (edm::ParameterDescription<double>("oivalue1", 101.0, true) and
                               edm::ParameterDescription<double>("oivalue2", 101.0, true)) or
                         2 >> edm::ParameterDescription<std::string>("oivalue", "102", true));
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // Types used in the switch parameter cannot duplicate type already used in a wildcard
    edm::ParameterSetDescription psetDesc2;
    edm::ParameterWildcard<int> w1("*", edm::RequireAtLeastOne, true);
    psetDesc2.addNode(w1);

    try {
      psetDesc2.ifValue( edm::ParameterDescription<int>("aswitch", 1, true),
                        1 >> (edm::ParameterDescription<unsigned>("avalue1", 101, true) and
                              edm::ParameterDescription<unsigned>("avalue2", 101, true)) or
                        2 >> edm::ParameterDescription<std::string>("avalue", "102", true));
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // Type used in the switch parameter cannot duplicate type in a case wildcard
    edm::ParameterSetDescription psetDesc3;
    try {
      psetDesc3.ifValue( edm::ParameterDescription<int>("xswitch", 1, true),
                         0 >> edm::ParameterWildcard<int>("*", edm::RequireAtLeastOne, true) or
                         1 >> (edm::ParameterDescription<double>("xvalue1", 101.0, true) and
                               edm::ParameterDescription<double>("xvalue2", 101.0, true)) );
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // Type used in a parameter cannot duplicate type in a case wildcard
    edm::ParameterSetDescription psetDesc4;
    psetDesc4.add<unsigned>("testunsigned", 1U);
    try {
      psetDesc4.ifValue( edm::ParameterDescription<int>("xswitch", 1, true),
                         0 >> edm::ParameterWildcard<unsigned>("*", edm::RequireAtLeastOne, true) or
                         1 >> (edm::ParameterDescription<double>("xvalue1", 101.0, true) and
                               edm::ParameterDescription<double>("xvalue2", 101.0, true)) );
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // No problem is wildcard type and parameter type are the same for different cases.
    edm::ParameterSetDescription psetDesc5;
    psetDesc5.ifValue( edm::ParameterDescription<int>("uswitch", 1, true),
                       0 >> edm::ParameterWildcard<unsigned>("*", edm::RequireAtLeastOne, true) or
                       1 >> (edm::ParameterDescription<unsigned>("uvalue1", 101, true) and
                             edm::ParameterDescription<unsigned>("uvalue2", 101, true)) );

    // The switch parameter label cannot be the same as a label that already exists
    edm::ParameterSetDescription psetDesc6;
    psetDesc6.add<unsigned>("xswitch", 1U);
    try {
      psetDesc6.ifValue( edm::ParameterDescription<int>("xswitch", 1, true),
                         0 >> edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
                         1 >> (edm::ParameterDescription<double>("xvalue1", 101.0, true) and
                               edm::ParameterDescription<double>("xvalue2", 101.0, true)) );
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // Case labels cannot be the same as a label that already exists
    edm::ParameterSetDescription psetDesc7;
    psetDesc7.add<unsigned>("xvalue1", 1U);
    try {
      psetDesc7.ifValue( edm::ParameterDescription<int>("xswitch", 1, true),
                         0 >> edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
                         1 >> (edm::ParameterDescription<double>("xvalue1", 101.0, true) and
                               edm::ParameterDescription<double>("xvalue2", 101.0, true)) );
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // Case labels cannot be the same as a switch label
    edm::ParameterSetDescription psetDesc8;
    try {
      psetDesc8.ifValue( edm::ParameterDescription<int>("xswitch", 1, true),
                         0 >> edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
                         1 >> (edm::ParameterDescription<double>("xswitch", 101.0, true) and
                               edm::ParameterDescription<double>("xvalue2", 101.0, true)) );
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // Parameter set switch value must be one of the defined cases
    edm::ParameterSetDescription psetDesc9;
    try {
      psetDesc9.ifValue( edm::ParameterDescription<int>("xswitch", 1, true),
                         0 >> edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
                         1 >> (edm::ParameterDescription<double>("xvalue1", 101.0, true) and
                               edm::ParameterDescription<double>("xvalue2", 101.0, true)) );
      edm::ParameterSet pset;
      pset.addParameter<int>("xswitch", 5);
      psetDesc9.validate(pset);  
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    edm::ParameterSwitch<int> pswitch( edm::ParameterDescription<int>("xswitch", 1, true),
                         0 >> edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
                         1 >> (edm::ParameterDescription<double>("xvalue1", 101.0, true) and
                               edm::ParameterDescription<double>("xvalue2", 101.0, true)) );
    edm::ParameterSetDescription psetDesc10;
    psetDesc10.addNode(pswitch);
    edm::ParameterSet pset10;
    testDesc(pswitch, psetDesc10, pset10, false, true);
    pset10.addParameter<int>("xswitch", 1);
    testDesc(pswitch, psetDesc10, pset10, true, true);
  }

// ---------------------------------------------------------------------------------

  void testXor() {
    edm::ParameterSetDescription psetDesc1;
    std::auto_ptr<edm::ParameterDescriptionNode> node1(
              edm::ParameterDescription<double>("x1", 101.0, true) xor
              (edm::ParameterDescription<double>("x1", 101.0, true) and
               edm::ParameterDescription<double>("x2", 101.0, true)) xor
              edm::ParameterDescription<double>("x1", 101.0, true) xor
              (edm::ParameterDescription<double>("x1", 101.0, true) or
               edm::ParameterDescription<double>("x2", 101.0, true))
             );

    edm::ParameterSet pset1;

    edm::ParameterSet pset2;
    pset2.addParameter("x1", 11.0);
    pset2.addParameter("x2", 12.0);

    assert(node1->exists(pset1) == false);
    assert(node1->partiallyExists(pset1) == false);
    assert(node1->howManyXORSubNodesExist(pset1) == 0);

    assert(node1->exists(pset2) == false);
    assert(node1->partiallyExists(pset2) == false);
    assert(node1->howManyXORSubNodesExist(pset2) == 4);

    // 0 of the options existing should fail validation
    psetDesc1.addNode(node1);
    try {
      psetDesc1.validate(pset1);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */}

    // More than one of the options existing should also fail
    try {
      psetDesc1.validate(pset2);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */}

    // One of the labels cannot already exist in the description
    edm::ParameterSetDescription psetDesc2;
    psetDesc2.add<unsigned>("xvalue1", 1U);
    std::auto_ptr<edm::ParameterDescriptionNode> node2(
              edm::ParameterDescription<double>("x1", 101.0, true) xor
              (edm::ParameterDescription<double>("x1", 101.0, true) and
               edm::ParameterDescription<double>("x2", 101.0, true)) xor
              edm::ParameterDescription<double>("x1", 101.0, true) xor
              (edm::ParameterDescription<double>("xvalue1", 101.0, true) or
               edm::ParameterDescription<double>("x2", 101.0, true))
             );
    try {
      psetDesc2.addNode(node2);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // One of the labels cannot already exist in the description, other order
    edm::ParameterSetDescription psetDesc3;
    std::auto_ptr<edm::ParameterDescriptionNode> node3(
              edm::ParameterDescription<double>("x1", 101.0, true) xor
              (edm::ParameterDescription<double>("x1", 101.0, true) and
               edm::ParameterDescription<double>("x2", 101.0, true)) xor
              edm::ParameterDescription<double>("x1", 101.0, true) xor
              (edm::ParameterDescription<double>("xvalue1", 101.0, true) or
               edm::ParameterDescription<double>("x2", 101.0, true))
             );
    psetDesc3.addNode(node3);
    try {
      psetDesc3.add<unsigned>("xvalue1", 1U);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // A parameter cannot use the same type as a wildcard
    edm::ParameterSetDescription psetDesc4;
    std::auto_ptr<edm::ParameterDescriptionNode> node4(
              edm::ParameterDescription<double>("x1", 101.0, true) xor
              (edm::ParameterDescription<double>("x1", 101.0, true) and
               edm::ParameterDescription<unsigned>("x2", 101, true)) xor
              edm::ParameterDescription<double>("x1", 101.0, true) xor
              (edm::ParameterDescription<double>("x1", 101.0, true) or
               edm::ParameterDescription<double>("x2", 101.0, true))
             );
    psetDesc4.addNode(node4);

    edm::ParameterWildcard<unsigned> w4("*", edm::RequireAtLeastOne, true);
    try {
      psetDesc4.addNode(w4);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // A parameter cannot use the same type as a wildcard
    edm::ParameterSetDescription psetDesc5;
    std::auto_ptr<edm::ParameterDescriptionNode> node5(
              edm::ParameterDescription<double>("x1", 101.0, true) xor
              (edm::ParameterDescription<double>("x1", 101.0, true) and
               edm::ParameterWildcard<unsigned>("*", edm::RequireAtLeastOne, true)) xor
              edm::ParameterDescription<double>("x1", 101.0, true) xor
              (edm::ParameterDescription<double>("x1", 101.0, true) or
               edm::ParameterWildcard<unsigned>("*", edm::RequireAtLeastOne, true))
             );
    psetDesc5.addNode(node5);

    edm::ParameterDescription<unsigned> n5("z5", edm::RequireAtLeastOne, true);
    try {
      psetDesc5.addNode(n5);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }
  }

// ---------------------------------------------------------------------------------

  void testOr() {

    edm::ParameterSetDescription psetDesc1;
    std::auto_ptr<edm::ParameterDescriptionNode> node1(
              edm::ParameterDescription<double>("x1", 101.0, true) or
              (edm::ParameterDescription<double>("x2", 101.0, true) and
               edm::ParameterDescription<double>("x3", 101.0, true)) or
              edm::ParameterDescription<double>("x4", 101.0, true) or
              (edm::ParameterDescription<double>("x5", 101.0, true) xor
               edm::ParameterDescription<double>("x6", 101.0, true))
             );

    edm::ParameterSet pset1;

    edm::ParameterSet pset2;
    pset2.addParameter("x1", 11.0);
    pset2.addParameter("x2", 12.0);
    pset2.addParameter("x3", 13.0);
    pset2.addParameter("x4", 14.0);
    pset2.addParameter("x5", 15.0);

    assert(node1->exists(pset1) == false);
    assert(node1->partiallyExists(pset1) == false);
    assert(node1->howManyXORSubNodesExist(pset1) == 0);

    assert(node1->exists(pset2) == true);
    assert(node1->partiallyExists(pset2) == true);
    assert(node1->howManyXORSubNodesExist(pset2) == 1);

    // 0 of the options existing should fail validation
    psetDesc1.addNode(node1);
    psetDesc1.validate(pset1);

    // More than one of the options existing should succeed
    psetDesc1.validate(pset2);

    // One of the labels cannot already exist in the description
    edm::ParameterSetDescription psetDesc2;
    psetDesc2.add<unsigned>("x1", 1U);
    std::auto_ptr<edm::ParameterDescriptionNode> node2(
              edm::ParameterDescription<double>("x1", 101.0, true) or
              (edm::ParameterDescription<double>("x2", 101.0, true) and
               edm::ParameterDescription<double>("x3", 101.0, true)) or
              edm::ParameterDescription<double>("x4", 101.0, true) );
    try {
      psetDesc2.addNode(node2);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // One of the labels cannot already exist in the description, other order
    edm::ParameterSetDescription psetDesc3;
    std::auto_ptr<edm::ParameterDescriptionNode> node3(
              edm::ParameterDescription<double>("x1", 101.0, true) or
              (edm::ParameterDescription<double>("x2", 101.0, true) and
               edm::ParameterDescription<double>("x3", 101.0, true)) or
              edm::ParameterDescription<double>("x4", 101.0, true) );
    psetDesc3.addNode(node3);

    try {
      psetDesc3.add<unsigned>("x1", 1U);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }
    
    // Put the duplicate labels in different nodes of the "or" expression
    edm::ParameterSetDescription psetDesc4;
    std::auto_ptr<edm::ParameterDescriptionNode> node4(
              edm::ParameterDescription<double>("x1", 101.0, true) or
              (edm::ParameterDescription<double>("x2", 101.0, true) and
               edm::ParameterDescription<double>("x3", 101.0, true)) or
              edm::ParameterDescription<double>("x1", 101.0, true) );
    try {
      psetDesc4.addNode(node4);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // A type used in a wildcard should not be the same as a type
    // used for another parameter
    edm::ParameterSetDescription psetDesc5;
    std::auto_ptr<edm::ParameterDescriptionNode> node5(
	      edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
              (edm::ParameterDescription<double>("x2", 101.0, true) and
               edm::ParameterDescription<unsigned>("x3", 101U, true)) or
              edm::ParameterDescription<unsigned>("x1", 101U, true) );
    try {
      psetDesc5.addNode(node5);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // A type used in a wildcard should not be the same as a type
    // used for another parameter node
    edm::ParameterSetDescription psetDesc6;
    psetDesc6.add<double>("x0", 1.0);
    std::auto_ptr<edm::ParameterDescriptionNode> node6(
	      edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
              (edm::ParameterDescription<unsigned>("x2", 101U, true) and
               edm::ParameterDescription<unsigned>("x3", 101U, true)) or
              edm::ParameterDescription<unsigned>("x1", 101U, true) );
    try {
      psetDesc6.addNode(node6);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // A type used in a wildcard should not be the same as a type
    // used for another parameter node
    edm::ParameterSetDescription psetDesc7;
    psetDesc7.addWildcard<double>("*");
    std::auto_ptr<edm::ParameterDescriptionNode> node7(
	      edm::ParameterDescription<double>("x0", 1.0, true) or
              (edm::ParameterDescription<unsigned>("x2", 101U, true) and
               edm::ParameterDescription<unsigned>("x3", 101U, true)) or
              edm::ParameterDescription<unsigned>("x1", 101U, true) );
    try {
      psetDesc7.addNode(node7);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }
  }

// ---------------------------------------------------------------------------------

  void testAnd() {

    edm::ParameterSetDescription psetDesc1;
    std::auto_ptr<edm::ParameterDescriptionNode> node1(
              edm::ParameterDescription<double>("x1", 101.0, true) and
              (edm::ParameterDescription<double>("x2", 101.0, true) or
               edm::ParameterDescription<double>("x3", 101.0, true)) and
              edm::ParameterDescription<double>("x4", 101.0, true) and
              (edm::ParameterDescription<double>("x5", 101.0, true) xor
               edm::ParameterDescription<double>("x6", 101.0, true))
             );

    edm::ParameterSet pset1;

    edm::ParameterSet pset2;
    pset2.addParameter("x1", 11.0);
    pset2.addParameter("x2", 12.0);
    pset2.addParameter("x3", 13.0);
    pset2.addParameter("x4", 14.0);
    pset2.addParameter("x5", 15.0);

    edm::ParameterSet pset3;
    pset3.addParameter("x3", 13.0);

    assert(node1->exists(pset1) == false);
    assert(node1->partiallyExists(pset1) == false);
    assert(node1->howManyXORSubNodesExist(pset1) == 0);

    assert(node1->exists(pset2) == true);
    assert(node1->partiallyExists(pset2) == true);
    assert(node1->howManyXORSubNodesExist(pset2) == 1);

    assert(node1->exists(pset3) == false);
    assert(node1->partiallyExists(pset3) == true);
    assert(node1->howManyXORSubNodesExist(pset3) == 0);

    psetDesc1.addNode(node1);
    psetDesc1.validate(pset1);
    psetDesc1.validate(pset2);
    psetDesc1.validate(pset3);

    // One of the labels cannot already exist in the description
    edm::ParameterSetDescription psetDesc2;
    psetDesc2.add<unsigned>("x1", 1U);
    std::auto_ptr<edm::ParameterDescriptionNode> node2(
              edm::ParameterDescription<double>("x1", 101.0, true) and
              (edm::ParameterDescription<double>("x2", 101.0, true) or
               edm::ParameterDescription<double>("x3", 101.0, true)) and
              edm::ParameterDescription<double>("x4", 101.0, true) );
    try {
      psetDesc2.addNode(node2);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // One of the labels cannot already exist in the description, other order
    edm::ParameterSetDescription psetDesc3;
    std::auto_ptr<edm::ParameterDescriptionNode> node3(
              edm::ParameterDescription<double>("x1", 101.0, true) and
              (edm::ParameterDescription<double>("x2", 101.0, true) or
               edm::ParameterDescription<double>("x3", 101.0, true)) and
              edm::ParameterDescription<double>("x4", 101.0, true) );
    psetDesc3.addNode(node3);

    try {
      psetDesc3.add<unsigned>("x1", 1U);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }
    
    // Put the duplicate labels in different nodes of the "and" expression
    edm::ParameterSetDescription psetDesc4;
    std::auto_ptr<edm::ParameterDescriptionNode> node4(
              edm::ParameterDescription<double>("x1", 101.0, true) and
              (edm::ParameterDescription<double>("x2", 101.0, true) or
               edm::ParameterDescription<double>("x3", 101.0, true)) and
              edm::ParameterDescription<double>("x1", 101.0, true) );
    try {
      psetDesc4.addNode(node4);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // A type used in a wildcard should not be the same as a type
    // used for another parameter
    edm::ParameterSetDescription psetDesc5;
    std::auto_ptr<edm::ParameterDescriptionNode> node5(
	      edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) and
              (edm::ParameterDescription<double>("x2", 101.0, true) or
               edm::ParameterDescription<unsigned>("x3", 101U, true)) and
              edm::ParameterDescription<unsigned>("x1", 101U, true) );
    try {
      psetDesc5.addNode(node5);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // A type used in a wildcard should not be the same as a type
    // used for another parameter node
    edm::ParameterSetDescription psetDesc6;
    psetDesc6.add<double>("x0", 1.0);
    std::auto_ptr<edm::ParameterDescriptionNode> node6(
	      edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) and
              (edm::ParameterDescription<unsigned>("x2", 101U, true) or
               edm::ParameterDescription<unsigned>("x3", 101U, true)) and
              edm::ParameterDescription<unsigned>("x1", 101U, true) );
    try {
      psetDesc6.addNode(node6);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // A type used in a wildcard should not be the same as a type
    // used for another parameter node
    edm::ParameterSetDescription psetDesc7;
    psetDesc7.addWildcard<double>("*");
    std::auto_ptr<edm::ParameterDescriptionNode> node7(
	      edm::ParameterDescription<double>("x0", 1.0, true) and
              (edm::ParameterDescription<unsigned>("x2", 101U, true) or
               edm::ParameterDescription<unsigned>("x3", 101U, true)) and
              edm::ParameterDescription<unsigned>("x1", 101U, true) );
    try {
      psetDesc7.addNode(node7);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }
  }

// ---------------------------------------------------------------------------------

  void testIfExists() {

    edm::ParameterSetDescription psetDesc1;
    std::auto_ptr<edm::ParameterDescriptionNode> node1(
      new edm::IfExistsDescription(
	edm::ParameterDescription<double>("x1", 101.0, true),
        (edm::ParameterDescription<double>("x2", 101.0, true) and
         edm::ParameterDescription<double>("x3", 101.0, true)) ) );

    std::auto_ptr<edm::ParameterDescriptionNode> node1a(
      new edm::IfExistsDescription(
	(edm::ParameterDescription<double>("x2", 101.0, true) and
         edm::ParameterDescription<double>("x3", 101.0, true)),
        edm::ParameterDescription<double>("x1", 101.0, true)) );

    std::auto_ptr<edm::ParameterDescriptionNode> node1b(
      new edm::IfExistsDescription(
	(edm::ParameterDescription<double>("x1", 101.0, true) xor
         edm::ParameterDescription<int>("x1", 101, true) ),
        (edm::ParameterDescription<double>("x2", 101.0, true) and
         edm::ParameterDescription<double>("x3", 101.0, true)) ) );

    edm::ParameterSet pset1;

    edm::ParameterSet pset2;
    pset2.addParameter("x1", 11.0);
    pset2.addParameter("x2", 12.0);
    pset2.addParameter("x3", 13.0);

    edm::ParameterSet pset3;
    pset3.addParameter("x1", 14.0);

    edm::ParameterSet pset4;
    pset4.addParameter("x2", 15.0);
    pset4.addParameter("x3", 16.0);

    assert(node1->exists(pset1) == true);
    assert(node1->partiallyExists(pset1) == true);
    assert(node1->howManyXORSubNodesExist(pset1) == 1);
    assert(node1a->exists(pset1) == true);
    assert(node1b->exists(pset1) == true);

    assert(node1->exists(pset2) == true);
    assert(node1->partiallyExists(pset2) == true);
    assert(node1->howManyXORSubNodesExist(pset2) == 1);
    assert(node1a->exists(pset2) == true);
    assert(node1b->exists(pset2) == true);

    assert(node1->exists(pset3) == false);
    assert(node1->partiallyExists(pset3) == false);
    assert(node1->howManyXORSubNodesExist(pset3) == 0);
    assert(node1a->exists(pset3) == false);
    assert(node1b->exists(pset3) == false);

    assert(node1->exists(pset4) == false);
    assert(node1->partiallyExists(pset4) == false);
    assert(node1->howManyXORSubNodesExist(pset4) == 0);
    assert(node1a->exists(pset4) == false);
    assert(node1b->exists(pset4) == false);

    psetDesc1.addNode(node1);
    psetDesc1.validate(pset1);
    psetDesc1.validate(pset2);
    psetDesc1.validate(pset3);
    psetDesc1.validate(pset3);

    // One of the labels cannot already exist in the description
    edm::ParameterSetDescription psetDesc2;
    psetDesc2.add<unsigned>("x1", 1U);
    std::auto_ptr<edm::ParameterDescriptionNode> node2(
      new edm::IfExistsDescription(
	edm::ParameterDescription<double>("x1", 101.0, true),
        (edm::ParameterDescription<double>("x2", 101.0, true) and
         edm::ParameterDescription<double>("x3", 101.0, true)) ) );

    try {
      psetDesc2.addNode(node2);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // One of the labels cannot already exist in the description, other order
    edm::ParameterSetDescription psetDesc3;
    std::auto_ptr<edm::ParameterDescriptionNode> node3(
      new edm::IfExistsDescription(
	edm::ParameterDescription<double>("x1", 101.0, true),
        (edm::ParameterDescription<double>("x2", 101.0, true) and
         edm::ParameterDescription<double>("x3", 101.0, true)) ) );
    psetDesc3.addNode(node3);

    try {
      psetDesc3.add<unsigned>("x1", 1U);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }
    
    // Put the duplicate labels in different nodes of the "and" expression
    edm::ParameterSetDescription psetDesc4;
    std::auto_ptr<edm::ParameterDescriptionNode> node4(
      new edm::IfExistsDescription(
	edm::ParameterDescription<double>("x1", 101.0, true),
        (edm::ParameterDescription<double>("x2", 101.0, true) and
         edm::ParameterDescription<double>("x1", 101.0, true)) ) );
    try {
      psetDesc4.addNode(node4);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // A type used in a wildcard should not be the same as a type
    // used for another parameter
    edm::ParameterSetDescription psetDesc5;
    std::auto_ptr<edm::ParameterDescriptionNode> node5(
      new edm::IfExistsDescription(
	edm::ParameterDescription<double>("x1", 101.0, true),
        (edm::ParameterDescription<unsigned>("x2", 101U, true) and
         edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true)) ) );
    try {
      psetDesc5.addNode(node5);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // A type used in a wildcard should not be the same as a type
    // used for another parameter node
    edm::ParameterSetDescription psetDesc6;
    psetDesc6.add<double>("x0", 1.0);
    std::auto_ptr<edm::ParameterDescriptionNode> node6(
      new edm::IfExistsDescription(
	edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true),
        (edm::ParameterDescription<unsigned>("x2", 101U, true) and
         edm::ParameterDescription<unsigned>("x3", 102U, true)) ) );
    try {
      psetDesc6.addNode(node6);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // A type used in a wildcard should not be the same as a type
    // used for another parameter node
    edm::ParameterSetDescription psetDesc7;
    psetDesc7.addWildcard<double>("*");
    std::auto_ptr<edm::ParameterDescriptionNode> node7(
      new edm::IfExistsDescription(
	edm::ParameterDescription<double>("x1", 11.0, true),
        (edm::ParameterDescription<unsigned>("x2", 101U, true) and
         edm::ParameterDescription<unsigned>("x3", 102U, true)) ) );
    try {
      psetDesc7.addNode(node7);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }
  }

// ---------------------------------------------------------------------------------

  void testAllowedLabels() {

    edm::ParameterSetDescription psetDesc1;
    std::auto_ptr<edm::ParameterDescriptionNode> node1(
      new edm::AllowedLabelsDescription<int>("allowedLabels", true));

    edm::ParameterSet pset1;

    edm::ParameterSet pset2;
    std::vector<std::string> labels;
    pset2.addParameter<std::vector<std::string> >("allowedLabels", labels);

    assert(node1->exists(pset1) == false);
    assert(node1->partiallyExists(pset1) == false);
    assert(node1->howManyXORSubNodesExist(pset1) == 0);

    assert(node1->exists(pset2) == true);
    assert(node1->partiallyExists(pset2) == true);
    assert(node1->howManyXORSubNodesExist(pset2) == 1);

    // One of the labels cannot already exist in the description
    edm::ParameterSetDescription psetDesc2;
    psetDesc2.add<unsigned>("x1", 1U);
    std::auto_ptr<edm::ParameterDescriptionNode> node2(
      new edm::AllowedLabelsDescription<int>("x1", true));

    try {
      psetDesc2.addNode(node2);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // A type used in a wildcard should not be the same as a type
    // used for another parameter node
    edm::ParameterSetDescription psetDesc3;
    psetDesc3.addWildcard<std::vector<std::string> >("*");
    std::auto_ptr<edm::ParameterDescriptionNode> node3(
      new edm::AllowedLabelsDescription<int>("x1", true));
    try {
      psetDesc3.addNode(node3);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    edm::ParameterSetDescription psetDesc4;
    psetDesc4.labelsFrom<edm::ParameterSetDescription>("allowedLabelsA");

    edm::ParameterSet psetA;
    edm::ParameterSet psetB;
    psetB.addParameter<int>("x", 1);

    psetA.addParameter<edm::ParameterSet>("setB", psetB);

    // setB is an illegal parameter
    try {
      psetDesc4.validate(psetA);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    std::vector<std::string> labels1;
    labels1.push_back(std::string("setB"));
    psetA.addParameter<std::vector<std::string> >("allowedLabelsA", labels1);

    // Now setB should be an allowed parameter 
    psetDesc4.validate(psetA);

    // Above it did not validate the contents of the nested ParameterSet
    // because a description was not passed to the labelsFrom function.

    edm::ParameterSetDescription psetDesc5;
    psetDesc5.labelsFrom<edm::ParameterSetDescription>("allowedLabelsA", edm::ParameterSetDescription());
    // Now it should fail because the description says the contained ParameterSet
    // should be empty, but it has parameter "x"
    try {
      psetDesc5.validate(psetA);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    // Now include "x" in the description and it should once again pass validation
    edm::ParameterSetDescription psetDesc6;
    edm::ParameterSetDescription psetDesc7;
    psetDesc7.add<int>("x", 1);
    psetDesc6.labelsFrom<edm::ParameterSetDescription>("allowedLabelsA", psetDesc7);
    psetDesc6.validate(psetA);

    // A minor variation, repeat with a string argument
    edm::ParameterSetDescription psetDesc8;
    edm::ParameterSetDescription psetDesc9;
    edm::ParameterSetDescription psetDesc10;
    psetDesc8.labelsFrom<edm::ParameterSetDescription>(std::string("allowedLabelsA"));
    psetDesc9.labelsFrom<edm::ParameterSetDescription>(std::string("allowedLabelsA"), edm::ParameterSetDescription());
    psetDesc10.labelsFrom<edm::ParameterSetDescription>(std::string("allowedLabelsA"), psetDesc7);
    psetDesc8.validate(psetA);
    try {
      psetDesc9.validate(psetA);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }
    psetDesc10.validate(psetA);

    // Now repeat what was done above with the variations
    // necessary to test the vector<ParameterSet> case
    {
      edm::ParameterSetDescription psetDesc14;
      psetDesc14.labelsFrom<std::vector<edm::ParameterSet> >("allowedLabelsC");

      edm::ParameterSet psetC;
      edm::ParameterSet psetD;
      psetD.addParameter<int>("y", 1);

      std::vector<edm::ParameterSet> vpset;
      vpset.push_back(psetD);
      vpset.push_back(psetD);

      psetC.addParameter<std::vector<edm::ParameterSet> >("vSetD", vpset);

      // vSetD is an illegal parameter
      try {
        psetDesc14.validate(psetC);
        assert(0);
      }
      catch(edm::Exception) { /* There should be an exception */ }

      std::vector<std::string> labels1;
      labels1.push_back(std::string("vSetD"));
      psetC.addParameter<std::vector<std::string> >("allowedLabelsC", labels1);

      // Now vSetB should be an allowed parameter 
      psetDesc14.validate(psetC);

      // Above it did not validate the contents of the nested ParameterSet
      // because a description was not passed to the labelsFrom function.

      edm::ParameterSetDescription psetDesc15;
      psetDesc15.labelsFrom<std::vector<edm::ParameterSet> >("allowedLabelsC", edm::ParameterSetDescription());
      // Now it should fail because the description says the contained vector<ParameterSet>
      // should have empty ParameterSets, but they have parameter "y"
      try {
        psetDesc15.validate(psetC);
        assert(0);
      }
      catch(edm::Exception) { /* There should be an exception */ }

      // Now include "y" in the description and it should once again pass validation
      edm::ParameterSetDescription psetDesc6;
      edm::ParameterSetDescription psetDesc7;
      psetDesc7.add<int>("y", 1);

      psetDesc6.labelsFrom<std::vector<edm::ParameterSet> >("allowedLabelsC", psetDesc7);
      psetDesc6.validate(psetC);

      // A minor variation, repeat with a string argument
      edm::ParameterSetDescription psetDesc18;
      edm::ParameterSetDescription psetDesc19;
      edm::ParameterSetDescription psetDesc20;
      psetDesc18.labelsFrom<std::vector<edm::ParameterSet> >(std::string("allowedLabelsC"));
      psetDesc19.labelsFrom<std::vector<edm::ParameterSet> >(std::string("allowedLabelsC"), edm::ParameterSetDescription());
      psetDesc20.labelsFrom<std::vector<edm::ParameterSet> >(std::string("allowedLabelsC"), psetDesc7);
      psetDesc18.validate(psetC);
      try {
        psetDesc19.validate(psetC);
        assert(0);
      }
      catch(edm::Exception) { /* There should be an exception */ }
      psetDesc20.validate(psetC);
    }
  }
// ---------------------------------------------------------------------------------

  void testNoDefault() {

    edm::ParameterSetDescription psetDesc;
    psetDesc.add<int>("x");
    edm::ParameterSet pset;

    try {
      psetDesc.validate(pset);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    pset.addParameter<int>("x", 1);
    psetDesc.validate(pset);

    psetDesc.addVPSet("y", edm::ParameterSetDescription());
    try {
      psetDesc.validate(pset);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }
  }

// ---------------------------------------------------------------------------------

  void testWrongTrackiness() {

    edm::ParameterSet pset1;
    pset1.addParameter<int>("test1", 1);

    edm::ParameterSetDescription psetDesc1;
    psetDesc1.addUntracked<int>("test1", 1);
    try {
      psetDesc1.validate(pset1);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    edm::ParameterSet pset2;
    pset2.addParameter<edm::ParameterSet>("test2", edm::ParameterSet());

    edm::ParameterSetDescription psetDesc2;
    psetDesc2.addUntracked<edm::ParameterSetDescription>("test2", edm::ParameterSetDescription());
    try {
      psetDesc2.validate(pset2);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    edm::ParameterSet pset3;
    pset3.addParameter<std::vector<edm::ParameterSet> >("test3", std::vector<edm::ParameterSet>());

    edm::ParameterSetDescription psetDesc3;
    psetDesc3.addVPSetUntracked("test3", edm::ParameterSetDescription());
    try {
      psetDesc3.validate(pset3);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }
  }

// ---------------------------------------------------------------------------------

  void testWrongType() {

    edm::ParameterSet pset1;
    pset1.addParameter<unsigned int>("test1", 1);

    edm::ParameterSetDescription psetDesc1;
    psetDesc1.add<int>("test1", 1);
    try {
      psetDesc1.validate(pset1);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    edm::ParameterSet pset2;
    pset2.addParameter<int>("test2", 1);

    edm::ParameterSetDescription psetDesc2;
    psetDesc2.add<edm::ParameterSetDescription>("test2", edm::ParameterSetDescription());
    try {
      psetDesc2.validate(pset2);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }

    edm::ParameterSet pset3;
    pset3.addParameter<int>("test3", 1);

    edm::ParameterSetDescription psetDesc3;
    psetDesc3.addVPSetUntracked("test3", edm::ParameterSetDescription());
    try {
      psetDesc3.validate(pset3);
      assert(0);
    }
    catch(edm::Exception) { /* There should be an exception */ }
  }

// ---------------------------------------------------------------------------------

}

int main(int argc, char* argv[]) {

  std::cout << "Running TestFWCoreParameterSetDescription from parameterSetDescription_t.cc" << std::endl;

  {
    edm::ParameterSetDescription psetDesc;
    assert(!psetDesc.anythingAllowed());
    assert(!psetDesc.isUnknown());
    assert(psetDesc.begin() == psetDesc.end());

    edm::ParameterSet params;
    psetDesc.validate(params);

    params.addParameter<std::string>("testname",
                                   std::string("testvalue"));

    // Expect this to throw, parameter not in description
    try {
      psetDesc.validate(params);
      assert(0);
    }
    catch(edm::Exception) {
      // OK
    }

    psetDesc.setAllowAnything();
    assert(psetDesc.anythingAllowed());

    psetDesc.validate(params);
  }

  {
    edm::ParameterSetDescription psetDesc;

    edm::ParameterSet params;
    params.addParameter<std::string>("testname",
                                   std::string("testvalue"));
    psetDesc.setUnknown();
    assert(psetDesc.isUnknown());

    psetDesc.validate(params);
  }

  {
    // Test this type separately because I do not know how to
    // add an entry into a ParameterSet without FileInPath pointing
    // at a real file.
    edm::ParameterSetDescription psetDesc;
    edm::ParameterDescriptionBase * par = psetDesc.add<edm::FileInPath>("fileInPath", edm::FileInPath());
    assert(par->type() == edm::k_FileInPath);
    assert(edm::parameterTypeEnumToString(par->type()) == std::string("FileInPath"));
  }

  edm::ParameterSetDescription psetDesc;
  edm::ParameterSet pset;

  psetDesc.reserve(2);

  int a = 1;
  edm::ParameterDescriptionBase * par = psetDesc.add<int>(std::string("ivalue"), a);
  assert(par->exists(pset) == false);
  assert(par->partiallyExists(pset) == false);
  assert(par->howManyXORSubNodesExist(pset) == 0);
  pset.addParameter<int>("ivalue", a);
  assert(par != 0); 
  assert(par->label() == std::string("ivalue"));
  assert(par->type() == edm::k_int32);
  assert(par->isTracked() == true);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("int32"));
  assert(par->exists(pset) == true);
  assert(par->partiallyExists(pset) == true);
  assert(par->howManyXORSubNodesExist(pset) == 1);

  edm::ParameterSet psetWrongTrackiness;
  psetWrongTrackiness.addUntrackedParameter("ivalue", a);
  try {
    psetDesc.validate(psetWrongTrackiness);
    assert(0);
  }
  catch(edm::Exception) {
    // There should be an exception
  }

  edm::ParameterSet psetWrongType;
  psetWrongType.addUntrackedParameter("ivalue", 1U);
  try {
    psetDesc.validate(psetWrongType);
    assert(0);
  }
  catch(edm::Exception) {
    // There should be an exception
  }

  edm::ParameterSetDescription::const_iterator parIter = psetDesc.begin();
  assert(parIter->node().operator->() == par);


  unsigned b = 2;
  par = psetDesc.add<unsigned>("uvalue", b);
  pset.addParameter<unsigned>("uvalue", b);
  assert(par != 0);
  assert(par->label() == std::string("uvalue"));
  assert(par->type() == edm::k_uint32);
  assert(par->isTracked() == true);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("uint32"));

  parIter = psetDesc.begin();
  ++parIter;
  assert(parIter->node().operator->() == par);

  long long c = 3;
  par = psetDesc.addUntracked<long long>(std::string("i64value"), c);
  pset.addUntrackedParameter<long long>("i64value", c);
  assert(par != 0);
  assert(par->label() == std::string("i64value"));
  assert(par->type() == edm::k_int64);
  assert(par->isTracked() == false);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("int64"));

  unsigned long long d = 4;
  par = psetDesc.addUntracked<unsigned long long>("u64value", d);
  pset.addUntrackedParameter<unsigned long long>("u64value", d);
  assert(par != 0);
  assert(par->label() == std::string("u64value"));
  assert(par->type() == edm::k_uint64);
  assert(par->isTracked() == false);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("uint64"));

  double e = 5;
  par = psetDesc.addOptional<double>(std::string("dvalue"), e);
  pset.addParameter<double>("dvalue", e);
  assert(par != 0);
  assert(par->label() == std::string("dvalue"));
  assert(par->type() == edm::k_double);
  assert(par->isTracked() == true);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("double"));

  bool f = true;
  par = psetDesc.addOptional<bool>("bvalue", f);
  pset.addParameter<bool>("bvalue", f);
  assert(par != 0);
  assert(par->label() == std::string("bvalue"));
  assert(par->type() == edm::k_bool);
  assert(par->isTracked() == true);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("bool"));

  std::string g;
  par = psetDesc.addOptionalUntracked<std::string>(std::string("svalue"), g);
  pset.addUntrackedParameter<std::string>("svalue", g);
  assert(par != 0);
  assert(par->label() == std::string("svalue"));
  assert(par->type() == edm::k_string);
  assert(par->isTracked() == false);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("string"));

  edm::MinimalEventID h;
  par = psetDesc.addOptionalUntracked<edm::MinimalEventID>("evalue", h);
  pset.addUntrackedParameter<edm::MinimalEventID>("evalue", h);
  assert(par != 0);
  assert(par->label() == std::string("evalue"));
  assert(par->type() == edm::k_EventID);
  assert(par->isTracked() == false);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("EventID"));

  edm::LuminosityBlockID i;
  par = psetDesc.add<edm::LuminosityBlockID>("lvalue", i);
  pset.addParameter<edm::LuminosityBlockID>("lvalue", i);
  assert(par->type() == edm::k_LuminosityBlockID);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("LuminosityBlockID"));

  edm::InputTag j;
  par = psetDesc.add<edm::InputTag>("input", j);
  pset.addParameter<edm::InputTag>("input", j);
  assert(par->type() == edm::k_InputTag);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("InputTag"));

  std::vector<int> v1;
  par = psetDesc.add<std::vector<int> >("v1", v1);
  pset.addParameter<std::vector<int> >("v1", v1);
  assert(par->type() == edm::k_vint32);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("vint32"));

  std::vector<unsigned> v2;
  par = psetDesc.add<std::vector<unsigned> >("v2", v2);
  pset.addParameter<std::vector<unsigned> >("v2", v2);
  assert(par->type() == edm::k_vuint32);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("vuint32"));

  std::vector<long long> v3;
  par = psetDesc.add<std::vector<long long> >("v3", v3);
  pset.addParameter<std::vector<long long> >("v3", v3);
  assert(par->type() == edm::k_vint64);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("vint64"));

  std::vector<unsigned long long> v4;
  par = psetDesc.add<std::vector<unsigned long long> >("v4", v4);
  pset.addParameter<std::vector<unsigned long long> >("v4", v4);
  assert(par->type() == edm::k_vuint64);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("vuint64"));

  std::vector<double> v5;
  par = psetDesc.add<std::vector<double> >("v5", v5);
  pset.addParameter<std::vector<double> >("v5", v5);
  assert(par->type() == edm::k_vdouble);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("vdouble"));

  std::vector<std::string> v6;
  par = psetDesc.add<std::vector<std::string> >("v6", v6);
  pset.addParameter<std::vector<std::string> >("v6", v6);
  assert(par->type() == edm::k_vstring);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("vstring"));

  std::vector<edm::MinimalEventID> v7;
  par = psetDesc.add<std::vector<edm::MinimalEventID> >("v7", v7);
  pset.addParameter<std::vector<edm::MinimalEventID> >("v7", v7);
  assert(par->type() == edm::k_VEventID);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("VEventID"));

  std::vector<edm::LuminosityBlockID> v8;
  par = psetDesc.add<std::vector<edm::LuminosityBlockID> >("v8", v8);
  pset.addParameter<std::vector<edm::LuminosityBlockID> >("v8", v8);
  assert(par->type() == edm::k_VLuminosityBlockID);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("VLuminosityBlockID"));

  std::vector<edm::InputTag> v9;
  par = psetDesc.add<std::vector<edm::InputTag> >("v9", v9);
  pset.addParameter<std::vector<edm::InputTag> >("v9", v9);
  assert(par->type() == edm::k_VInputTag);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("VInputTag"));

  edm::ParameterSetDescription m;
  par = psetDesc.add<edm::ParameterSetDescription>("psetDesc", m);
  assert(par->exists(pset) == false);
  assert(par->partiallyExists(pset) == false);
  assert(par->howManyXORSubNodesExist(pset) == 0);
  edm::ParameterSet p1;
  pset.addParameter<edm::ParameterSet>("psetDesc", p1);
  assert(par->type() == edm::k_PSet);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("PSet"));
  assert(par->exists(pset) == true);
  assert(par->partiallyExists(pset) == true);
  assert(par->howManyXORSubNodesExist(pset) == 1);

  edm::ParameterSetDescription v10;
  par = psetDesc.addVPSet("psetVectorDesc", v10);
  assert(par->exists(pset) == false);
  assert(par->partiallyExists(pset) == false);
  assert(par->howManyXORSubNodesExist(pset) == 0);
  std::vector<edm::ParameterSet> vp1;
  pset.addParameter<std::vector<edm::ParameterSet> >("psetVectorDesc", vp1);
  assert(par->type() == edm::k_VPSet);
  assert(edm::parameterTypeEnumToString(par->type()) == std::string("VPSet"));
  assert(par->exists(pset) == true);
  assert(par->partiallyExists(pset) == true);
  assert(par->howManyXORSubNodesExist(pset) == 1);

  psetDesc.validate(pset);

  // Add a ParameterSetDescription nested in a ParameterSetDescription nested in
  // a vector in the top level ParameterSetDescription to see if the nesting is
  // working properly.

  edm::ParameterSet nest2;
  nest2.addParameter<int>("intLevel2a", 1);
  nest2.addUntrackedParameter<int>("intLevel2b", 1);
  nest2.addParameter<int>("intLevel2e", 1);
  nest2.addUntrackedParameter<int>("intLevel2f", 1);

  edm::ParameterSet nest1;
  nest1.addParameter<int>("intLevel1a", 1);
  nest1.addParameter<edm::ParameterSet>("nestLevel1b", nest2);

  std::vector<edm::ParameterSet> vPset;
  vPset.push_back(edm::ParameterSet());
  vPset.push_back(nest1);

  pset.addUntrackedParameter<std::vector<edm::ParameterSet> >("nestLevel0", vPset);

  std::vector<edm::ParameterSetDescription> testDescriptions;
  testDescriptions.push_back(psetDesc);
  testDescriptions.push_back(psetDesc);
  testDescriptions.push_back(psetDesc);

  for (int i = 0; i < 3; ++i) {

    edm::ParameterSetDescription nestLevel2;

    // for the first test do not put a parameter in the description
    // so there will be an extra parameter in the ParameterSet and
    // validation should fail.
    if (i > 0) par = nestLevel2.add<int>("intLevel2a", 1);

    // for the next test validation should pass

    // For the last test add an extra required parameter in the
    // description that is not in the ParameterSet.
    if (i == 2) par = nestLevel2.add<int>("intLevel2extra", 11);

    par = nestLevel2.addUntracked<int>("intLevel2b", 1);
    par = nestLevel2.addOptional<int>("intLevel2c", 1);
    par = nestLevel2.addOptionalUntracked<int>("intLevel2d", 1);
    par = nestLevel2.addOptional<int>("intLevel2e", 1);
    par = nestLevel2.addOptionalUntracked<int>("intLevel2f", 1);

    edm::ParameterSetDescription nestLevel1;
    par = nestLevel1.add<int>("intLevel1a", 1);
    par = nestLevel1.add<edm::ParameterSetDescription>("nestLevel1b", nestLevel2);

    testDescriptions[i].addVPSetUntracked("nestLevel0", nestLevel1);
  }

  // Now run the validation and make sure we get the expected results
  try {
    testDescriptions[0].validate(pset);
    assert(0);
  }
  catch(edm::Exception) {
    // There should be an exception
  }

  // This one should pass validation with no exception
  testDescriptions[1].validate(pset);

  // This validation should also pass and it should insert
  // the missing parameter into the ParameterSet
  testDescriptions[2].validate(pset);

  std::vector<edm::ParameterSet> vpset = 
    pset.getUntrackedParameter<std::vector<edm::ParameterSet> >("nestLevel0");
  edm::ParameterSet psetInPset = vpset[1].getParameter<edm::ParameterSet>("nestLevel1b");
  assert(psetInPset.getParameter<int>("intLevel2extra") ==  11);

  // One more iteration, this time the purpose is to
  // test the parameterSetDescription accessors.
  edm::ParameterSetDescription nestLevel2;
  par = nestLevel2.add<int>("intLevel2a", 1);
  par->setComment("testComment");
  assert(par->parameterSetDescription() == 0);
  edm::ParameterDescriptionBase const& constParRef = *par;
  assert(constParRef.parameterSetDescription() == 0);

  par = nestLevel2.addUntracked<int>("intLevel2b", 1);
  par = nestLevel2.addOptional<int>("intLevel2c", 1);
  par = nestLevel2.addOptionalUntracked<int>("intLevel2d", 1);
  par = nestLevel2.addOptional<int>("intLevel2e", 1);
  par = nestLevel2.addOptionalUntracked<int>("intLevel2f", 1);
  nestLevel2.setAllowAnything();

  edm::ParameterSetDescription nestLevel1;
  par = nestLevel1.add<int>("intLevel1a", 1);
  par->setComment("testComment1");
  par = nestLevel1.add<edm::ParameterSetDescription>("nestLevel1b", nestLevel2);
  assert(par->parameterSetDescription() != 0);
  assert(par->parameterSetDescription()->begin()->node()->comment() == std::string("testComment"));
  edm::ParameterDescriptionBase const& constParRef2 = *par;
  assert(constParRef2.parameterSetDescription() != 0);
  assert(constParRef2.parameterSetDescription()->begin()->node()->comment() == std::string("testComment"));

  assert(par->parameterSetDescription()->anythingAllowed() == true);
  assert(constParRef2.parameterSetDescription()->anythingAllowed() == true);

  par = psetDesc.addVPSetUntracked("nestLevel0", nestLevel1);
  assert(par->parameterSetDescription() != 0);
  assert(par->parameterSetDescription()->begin()->node()->comment() == std::string("testComment1"));
  edm::ParameterDescriptionBase const& constParRef3 = *par;
  assert(constParRef3.parameterSetDescription() != 0);
  assert(constParRef3.parameterSetDescription()->begin()->node()->comment() == std::string("testComment1"));

  psetDesc.validate(pset);

  testParameterSetDescription::testWildcards();
  testParameterSetDescription::testSwitch();
  testParameterSetDescription::testXor();
  testParameterSetDescription::testOr();
  testParameterSetDescription::testAnd();
  testParameterSetDescription::testIfExists();
  testParameterSetDescription::testAllowedLabels();
  testParameterSetDescription::testNoDefault();
  testParameterSetDescription::testWrongTrackiness();
  testParameterSetDescription::testWrongType();
  
  return 0;
}
