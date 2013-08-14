// -*- C++ -*-
//
// Package:     Provenance
// Class  :     transientproductlookupmap_t
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed May  6 08:52:17 CDT 2009
//

// system include files
#include <iostream>
#include <cppunit/extensions/HelperMacros.h>

// user include files
#include "DataFormats/Provenance/interface/TransientProductLookupMap.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"

class testTransientProductLookupMap: public CppUnit::TestFixture {
   CPPUNIT_TEST_SUITE(testTransientProductLookupMap);

   CPPUNIT_TEST(constructTest);

   CPPUNIT_TEST(reorderTest);

   CPPUNIT_TEST_SUITE_END();
public:
   void setUp(){}
   void tearDown(){}

   void constructTest();
   void reorderTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testTransientProductLookupMap);

void
testTransientProductLookupMap::constructTest() {
   using namespace edm;
   TransientProductLookupMap::FillFromMap fillFromMap;

   //use a list because pointers into its elements are always stable
   std::list<ConstBranchDescription> cbdlist;

   {
      ParameterSetID psetID("thisisadummyitem");
      ProductTransientIndex index(0);
      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "A",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "a",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "A",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "b",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "A",
                                                                 "edm::BranchID",
                                                                 "edmBranchID",
                                                                 "",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::BranchID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::BranchID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;


      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "B",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "a",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "B",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "b",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "B",
                                                                 "edm::BranchID",
                                                                 "edmBranchID",
                                                                 "",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::BranchID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::BranchID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;


      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "C",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "a",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "C",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "b",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "C",
                                                                 "edm::BranchID",
                                                                 "edmBranchID",
                                                                 "",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::BranchID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::BranchID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      //NOTE: need to put stuff in Run and Lumi since that went wrong elsewhere
      cbdlist.push_back(ConstBranchDescription(BranchDescription(InRun,
                                                                 "foo",
                                                                 "A",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "a",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InRun,
                                                                 "foo",
                                                                 "A",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "b",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InRun,
                                                                 "foo",
                                                                 "A",
                                                                 "edm::BranchID",
                                                                 "edmBranchID",
                                                                 "",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::BranchID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::BranchID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;


      cbdlist.push_back(ConstBranchDescription(BranchDescription(InRun,
                                                                 "foo",
                                                                 "B",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "a",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InRun,
                                                                 "foo",
                                                                 "B",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "b",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InRun,
                                                                 "foo",
                                                                 "B",
                                                                 "edm::BranchID",
                                                                 "edmBranchID",
                                                                 "",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::BranchID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::BranchID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;


      cbdlist.push_back(ConstBranchDescription(BranchDescription(InRun,
                                                                 "foo",
                                                                 "C",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "a",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InRun,
                                                                 "foo",
                                                                 "C",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "b",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InRun,
                                                                 "foo",
                                                                 "C",
                                                                 "edm::BranchID",
                                                                 "edmBranchID",
                                                                 "",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::BranchID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::BranchID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;
   }

   TransientProductLookupMap tplm;
   tplm.fillFrom(fillFromMap);

   std::pair<TransientProductLookupMap::const_iterator, TransientProductLookupMap::const_iterator> range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent));
   CPPUNIT_ASSERT(6 == range.second - range.first);

   char const* const order[] = {"A", "a", "B", "a", "C", "a", "A", "b", "B", "b", "C", "b"};
   char const* const* itOrder = order;
   unsigned int const indexOrder[] = {0, 1, 2, 0, 1, 2};
   unsigned int const* itIndexOrder = indexOrder;
   int index = 0;
   for(TransientProductLookupMap::const_iterator it = range.first; it != range.second; ++it, ++itOrder, ++index) {
      //std::cout << "ordering " << it->branchDescription()->processName() << " expect " << *itOrder << std::endl;
      CPPUNIT_ASSERT(it->branchDescription()->processName() == *(itOrder++));
      //std::cout << "         " << it->branchDescription()->productInstanceName() << " expect " << *itOrder << std::endl;
      CPPUNIT_ASSERT(it->branchDescription()->productInstanceName() == *itOrder);
      CPPUNIT_ASSERT(it->branchDescription()->className() == "edm::ProductID");
      //std::cout << "         " << it->isFirst() << " expect " << not static_cast<bool>(index % 3) << std::endl;
      CPPUNIT_ASSERT(it->isFirst() == !static_cast<bool>(index % 3));
      CPPUNIT_ASSERT(it->processIndex() == *(itIndexOrder++));
   }

   range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::BranchID)), InEvent));
   CPPUNIT_ASSERT(3 == range.second - range.first);

   range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::BranchID)), InLumi));
   CPPUNIT_ASSERT(0 == range.second - range.first);

   range = tplm.equal_range(TypeInBranchType(TypeID(typeid(double)), InEvent));
   CPPUNIT_ASSERT(0 == range.second - range.first);

   range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InRun));
   CPPUNIT_ASSERT(6 == range.second - range.first);

   range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent), std::string("foo"), std::string("Q"));
   CPPUNIT_ASSERT(0 == range.second - range.first);

   range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent), std::string("foo"), std::string("b"));
   CPPUNIT_ASSERT(range.first->branchDescription()->moduleLabel() == std::string("foo"));
   CPPUNIT_ASSERT(range.first->branchDescription()->productInstanceName() == std::string("b"));

}

void
testTransientProductLookupMap::reorderTest()
{
   using namespace edm;
   TransientProductLookupMap::FillFromMap fillFromMap;

   //use a list because pointers into its elements are always stable
   std::list<ConstBranchDescription> cbdlist;

   {
      ParameterSetID psetID("thisisadummyitem");
      ProductTransientIndex index(0);
      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "A",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "a",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "A",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "b",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "A",
                                                                 "edm::BranchID",
                                                                 "edmBranchID",
                                                                 "",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::BranchID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::BranchID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;


      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "B",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "a",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "B",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "b",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "B",
                                                                 "edm::BranchID",
                                                                 "edmBranchID",
                                                                 "",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::BranchID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::BranchID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;


      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "C",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "a",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "C",
                                                                 "edm::ProductID",
                                                                 "edmProductID",
                                                                 "b",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::ProductID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

      cbdlist.push_back(ConstBranchDescription(BranchDescription(InEvent,
                                                                 "foo",
                                                                 "C",
                                                                 "edm::BranchID",
                                                                 "edmBranchID",
                                                                 "",
                                                                 "foo",
                                                                 psetID,
                                                                 TypeID(typeid(edm::ProductID)))));
      fillFromMap[std::make_pair(TypeInBranchType(TypeID(typeid(edm::BranchID)), cbdlist.back().branchType()), &(cbdlist.back()))] = index++;

   }

   TransientProductLookupMap tplm;
   tplm.fillFrom(fillFromMap);

   ParameterSetID dummyParameterSetID;

   //NOTE: order is oldest to newest
   std::vector<ProcessConfiguration> vPC;
   {
      vPC.push_back(ProcessConfiguration("C", dummyParameterSetID, ReleaseVersion(""), PassID("")));
      vPC.push_back(ProcessConfiguration("A", dummyParameterSetID, ReleaseVersion(""), PassID("")));
      //vPC.push_back(ProcessConfiguration("B", dummyParameterSetID, ReleaseVersion(""), PassID("")));
      ProcessHistory ph(vPC);

      tplm.reorderIfNecessary(InEvent, ph, "B");

      std::pair<TransientProductLookupMap::const_iterator, TransientProductLookupMap::const_iterator> range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent));
      CPPUNIT_ASSERT(6 == range.second - range.first);

      char const* const order[] = {"B", "a", "A", "a", "C", "a", "B", "b", "A", "b", "C", "b"};
      char const* const* itOrder = order;
      unsigned int const indexOrder[] = {0, 1, 2, 0, 1, 2};
      unsigned int const* itIndexOrder = indexOrder;
      int index = 0;
      for(TransientProductLookupMap::const_iterator it = range.first; it != range.second; ++it, ++itOrder, ++index) {
         //std::cout << "ordering " << it->branchDescription()->processName() << " expect " << *itOrder << std::endl;
         CPPUNIT_ASSERT(it->branchDescription()->processName() == *(itOrder++));
         //std::cout << "         " << it->branchDescription()->productInstanceName() << " expect " << *itOrder << std::endl;
         CPPUNIT_ASSERT(it->branchDescription()->productInstanceName() == *itOrder);
         CPPUNIT_ASSERT(it->branchDescription()->className() == "edm::ProductID");
         CPPUNIT_ASSERT(it->isFirst() == !static_cast<bool>(index % 3));
         CPPUNIT_ASSERT(it->processIndex() == *(itIndexOrder++));
      }

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::BranchID)), InEvent));
      CPPUNIT_ASSERT(3 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::BranchID)), InRun));
      CPPUNIT_ASSERT(0 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(double)), InEvent));
      CPPUNIT_ASSERT(0 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent), std::string("foo"), std::string("Q"));
      CPPUNIT_ASSERT(0 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent), std::string("foo"), std::string("b"));
      CPPUNIT_ASSERT(range.first->branchDescription()->moduleLabel() == std::string("foo"));
      CPPUNIT_ASSERT(range.first->branchDescription()->productInstanceName() == std::string("b"));
   }
   {
      //No A this time
      vPC.clear();
      vPC.push_back(ProcessConfiguration("C", dummyParameterSetID, ReleaseVersion(""), PassID("")));

      ProcessHistory ph(vPC);

      tplm.reorderIfNecessary(InEvent, ph, "B");

      std::pair<TransientProductLookupMap::const_iterator, TransientProductLookupMap::const_iterator> range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent));
      CPPUNIT_ASSERT(6 == range.second - range.first);

      char const* const order[] = {"B", "C", "A", "B", "C", "A"};
      char const* const* itOrder = order;
      for(TransientProductLookupMap::const_iterator it = range.first; it != range.second; ++it, ++itOrder) {
         //std::cout << "ordering " << it->branchDescription()->processName() << " expect " << *itOrder << std::endl;
         CPPUNIT_ASSERT(it->branchDescription()->processName() == *itOrder);
      }

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::BranchID)), InEvent));
      CPPUNIT_ASSERT(3 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::BranchID)), InRun));
      CPPUNIT_ASSERT(0 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(double)), InEvent));
      CPPUNIT_ASSERT(0 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent), std::string("foo"), std::string("Q"));
      CPPUNIT_ASSERT(0 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent), std::string("foo"), std::string("b"));
      CPPUNIT_ASSERT(range.first->branchDescription()->moduleLabel() == std::string("foo"));
      CPPUNIT_ASSERT(range.first->branchDescription()->productInstanceName() == std::string("b"));
   }

   {
      //Add several steps which have no data saved in final file
      vPC.clear();
      vPC.push_back(ProcessConfiguration("Z", dummyParameterSetID, ReleaseVersion(""), PassID("")));
      vPC.push_back(ProcessConfiguration("C", dummyParameterSetID, ReleaseVersion(""), PassID("")));
      vPC.push_back(ProcessConfiguration("Y", dummyParameterSetID, ReleaseVersion(""), PassID("")));
      vPC.push_back(ProcessConfiguration("A", dummyParameterSetID, ReleaseVersion(""), PassID("")));
      vPC.push_back(ProcessConfiguration("X", dummyParameterSetID, ReleaseVersion(""), PassID("")));

      ProcessHistory ph(vPC);

      tplm.reorderIfNecessary(InEvent, ph, "B");

      std::pair<TransientProductLookupMap::const_iterator, TransientProductLookupMap::const_iterator> range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent));
      CPPUNIT_ASSERT(6 == range.second - range.first);

      char const* const order[] = {"B", "A", "C", "B", "A", "C"};
      char const* const* itOrder = order;
      for(TransientProductLookupMap::const_iterator it = range.first; it != range.second; ++it, ++itOrder) {
         //std::cout << "ordering " << it->branchDescription()->processName() << " expect " << *itOrder << std::endl;
         CPPUNIT_ASSERT(it->branchDescription()->processName() == *itOrder);
      }

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::BranchID)), InEvent));
      CPPUNIT_ASSERT(3 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::BranchID)), InRun));
      CPPUNIT_ASSERT(0 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(double)), InEvent));
      CPPUNIT_ASSERT(0 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent), std::string("foo"), std::string("Q"));
      CPPUNIT_ASSERT(0 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent), std::string("foo"), std::string("b"));
      CPPUNIT_ASSERT(range.first->branchDescription()->moduleLabel() == std::string("foo"));
      CPPUNIT_ASSERT(range.first->branchDescription()->productInstanceName() == std::string("b"));

   }

   {
      //There is no data in the present configuration
      vPC.clear();
      vPC.push_back(ProcessConfiguration("C", dummyParameterSetID, ReleaseVersion(""), PassID("")));
      vPC.push_back(ProcessConfiguration("B", dummyParameterSetID, ReleaseVersion(""), PassID("")));
      vPC.push_back(ProcessConfiguration("A", dummyParameterSetID, ReleaseVersion(""), PassID("")));

      ProcessHistory ph(vPC);

      tplm.reorderIfNecessary(InEvent, ph, "E");

      std::pair<TransientProductLookupMap::const_iterator, TransientProductLookupMap::const_iterator> range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent));
      CPPUNIT_ASSERT(6 == range.second - range.first);

      char const* const order[] = {"A", "B", "C", "A", "B", "C"};
      char const* const* itOrder = order;
      for(TransientProductLookupMap::const_iterator it = range.first; it != range.second; ++it, ++itOrder) {
         //std::cout << "ordering " << it->branchDescription()->processName() << " expect " << *itOrder << std::endl;
         CPPUNIT_ASSERT(it->branchDescription()->processName() == *itOrder);
      }

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::BranchID)), InEvent));
      CPPUNIT_ASSERT(3 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::BranchID)), InRun));
      CPPUNIT_ASSERT(0 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(double)), InEvent));
      CPPUNIT_ASSERT(0 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent), std::string("foo"), std::string("Q"));
      CPPUNIT_ASSERT(0 == range.second - range.first);

      range = tplm.equal_range(TypeInBranchType(TypeID(typeid(edm::ProductID)), InEvent), std::string("foo"), std::string("b"));
      CPPUNIT_ASSERT(range.first->branchDescription()->moduleLabel() == std::string("foo"));
      CPPUNIT_ASSERT(range.first->branchDescription()->productInstanceName() == std::string("b"));
   }

}



