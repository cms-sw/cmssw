#include "Utilities/Testing/interface/CppUnit_testdriver.icpp" //gives main
#include <cppunit/extensions/HelperMacros.h>

#include "TClass.h"
#include "TList.h"
#include "TDataMember.h"

#include <string>
#include <set>
#include <tuple>
#include <iostream>
#include <cassert>


class TestSchemaEvolution : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestSchemaEvolution);
  CPPUNIT_TEST(checkVersions);
  CPPUNIT_TEST_SUITE_END();

public:
  TestSchemaEvolution() = default;
  ~TestSchemaEvolution() = default;
  void setUp() {}
  void tearDown() {}
  void checkVersions();

private:
  void fillBaseline();
  void gatherAllClasses();
  void runComparison();
  void loopOnDataMembers(TClass *);
  void loopOnBases(TClass *);
  void analyseClass(TClass *);
  std::set<std::tuple<std::string, short>> unique_classes_;
  std::set<std::tuple<std::string, short>> unique_classes_current_;

};

CPPUNIT_TEST_SUITE_REGISTRATION( TestSchemaEvolution );

void TestSchemaEvolution::fillBaseline() {
  unique_classes_current_.insert(std::make_tuple("TArray", 1));
  unique_classes_current_.insert(std::make_tuple("TArrayD", 1));
  unique_classes_current_.insert(std::make_tuple("TArrayF", 1));
  unique_classes_current_.insert(std::make_tuple("TArrayI", 1));
  unique_classes_current_.insert(std::make_tuple("TArrayS", 1));
  unique_classes_current_.insert(std::make_tuple("TAtt3D", 1));
  unique_classes_current_.insert(std::make_tuple("TAttAxis", 4));
  unique_classes_current_.insert(std::make_tuple("TAttFill", 2));
  unique_classes_current_.insert(std::make_tuple("TAttLine", 2));
  unique_classes_current_.insert(std::make_tuple("TAttMarker", 2));
  unique_classes_current_.insert(std::make_tuple("TAxis", 10));
  unique_classes_current_.insert(std::make_tuple("TH1", 8));
  unique_classes_current_.insert(std::make_tuple("TH1D", 2));
  unique_classes_current_.insert(std::make_tuple("TH1F", 2));
  unique_classes_current_.insert(std::make_tuple("TH1I", 2));
  unique_classes_current_.insert(std::make_tuple("TH1S", 2));
  unique_classes_current_.insert(std::make_tuple("TH2", 4));
  unique_classes_current_.insert(std::make_tuple("TH2D", 3));
  unique_classes_current_.insert(std::make_tuple("TH2F", 3));
  unique_classes_current_.insert(std::make_tuple("TH2I", 3));
  unique_classes_current_.insert(std::make_tuple("TH2S", 3));
  unique_classes_current_.insert(std::make_tuple("TH3", 5));
  unique_classes_current_.insert(std::make_tuple("TH3D", 3));
  unique_classes_current_.insert(std::make_tuple("TH3F", 3));
  unique_classes_current_.insert(std::make_tuple("TH3I", 3));
  unique_classes_current_.insert(std::make_tuple("TH3S", 3));
  unique_classes_current_.insert(std::make_tuple("TNamed", 1));
  unique_classes_current_.insert(std::make_tuple("TObject", 1));
  unique_classes_current_.insert(std::make_tuple("TProfile", 6));
  unique_classes_current_.insert(std::make_tuple("TProfile2D", 7));
  unique_classes_current_.insert(std::make_tuple("TString", 2));
}

void TestSchemaEvolution::runComparison() {
  CPPUNIT_ASSERT(unique_classes_current_.size() == unique_classes_.size());
  for (auto cl : unique_classes_current_) {
    std::cout << "Checking " << std::get<0>(cl) << " " << std::get<1>(cl) << std::endl;
    CPPUNIT_ASSERT(unique_classes_.find(cl) != unique_classes_.end());
  }
}

void TestSchemaEvolution::checkVersions() {
  fillBaseline();
  gatherAllClasses();
  runComparison();
}

void TestSchemaEvolution::gatherAllClasses() {
  static const char *classes[] =
  {
    "TH1F", "TH1S", "TH1D", "TH1I",
    "TH2F", "TH2S", "TH2D", "TH2I",
    "TH3F", "TH3S", "TH3D", "TH3I",
    "TProfile", "TProfile2D", 0
  };

  int i = 0;
  while (classes[i])
  {
    TClass *tcl = TClass::GetClass(classes[i]);
    if (!tcl)
      continue;
    unique_classes_.insert(std::make_tuple(classes[i], tcl->GetClassVersion()));
    analyseClass(tcl);
    ++i;
  }
}

void TestSchemaEvolution::loopOnDataMembers(TClass *tcl)
{
  TList *dms = tcl->GetListOfDataMembers();
  TIter next(dms);
  while (TObject *obj = next()) {
    TClass *cl = TClass::GetClass(((TDataMember *)obj)->GetFullTypeName());
    if (cl && cl->HasDictionary()) {
      unique_classes_.insert(std::make_tuple(cl->GetName(), cl->GetClassVersion()));
      analyseClass(cl);
    }
  }
}


void TestSchemaEvolution::loopOnBases(TClass *tcl)
{
  TList *bases = tcl->GetListOfBases();
  TIter next(bases);
  while (TObject *obj = next()) {
    TClass *cl = TClass::GetClass(obj->GetName());
    if (cl && cl->HasDictionary()) {
      unique_classes_.insert(std::make_tuple(cl->GetName(), cl->GetClassVersion()));
      analyseClass(cl);
    }
  }
}


void TestSchemaEvolution::analyseClass(TClass *cl)
{
  loopOnBases(cl);
  loopOnDataMembers(cl);
}

