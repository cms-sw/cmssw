#include "Utilities/Testing/interface/CppUnit_testdriver.icpp" //gives main
#include <cppunit/extensions/HelperMacros.h>

#include "TClass.h"
#include "TList.h"
#include "TDataMember.h"

#include <string>
#include <unordered_map>
#include <iostream>
#include <cassert>

using std::unordered_map;
using std::string;

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


  unordered_map<string, short> unique_classes_;
  unordered_map<string, short> unique_classes_current_;

};

CPPUNIT_TEST_SUITE_REGISTRATION( TestSchemaEvolution );

void TestSchemaEvolution::fillBaseline() {
  unique_classes_current_.insert(std::make_pair("vector<double>", 6));
  unique_classes_current_.insert(std::make_pair("TF1", 10));
  unique_classes_current_.insert(std::make_pair("TH3S", 3));
  unique_classes_current_.insert(std::make_pair("TAtt3D", 1));
  unique_classes_current_.insert(std::make_pair("TH3", 5));
  unique_classes_current_.insert(std::make_pair("TH3F", 3));
  unique_classes_current_.insert(std::make_pair("TH2D", 3));
  unique_classes_current_.insert(std::make_pair("TH2S", 3));
  unique_classes_current_.insert(std::make_pair("TArrayI", 1));
  unique_classes_current_.insert(std::make_pair("TH2I", 3));
  unique_classes_current_.insert(std::make_pair("TH2F", 3));
  unique_classes_current_.insert(std::make_pair("TH1I", 2));
  unique_classes_current_.insert(std::make_pair("TString", 2));
  unique_classes_current_.insert(std::make_pair("TAttLine", 2));
  unique_classes_current_.insert(std::make_pair("TObject", 1));
  unique_classes_current_.insert(std::make_pair("TH3D", 3));
  unique_classes_current_.insert(std::make_pair("TH1S", 2));
  unique_classes_current_.insert(std::make_pair("TH3I", 3));
  unique_classes_current_.insert(std::make_pair("TAttFill", 2));
  unique_classes_current_.insert(std::make_pair("TNamed", 1));
  unique_classes_current_.insert(std::make_pair("TH1F", 2));
  unique_classes_current_.insert(std::make_pair("TH2", 4));
  unique_classes_current_.insert(std::make_pair("TH1", 8));
  unique_classes_current_.insert(std::make_pair("TProfile", 6));
  unique_classes_current_.insert(std::make_pair("TAttMarker", 2));
  unique_classes_current_.insert(std::make_pair("TArray", 1));
  unique_classes_current_.insert(std::make_pair("TAxis", 10));
  unique_classes_current_.insert(std::make_pair("TProfile2D", 7));
  unique_classes_current_.insert(std::make_pair("TH1D", 2));
  unique_classes_current_.insert(std::make_pair("TArrayS", 1));
  unique_classes_current_.insert(std::make_pair("TAttAxis", 4));
  unique_classes_current_.insert(std::make_pair("TArrayD", 1));
  unique_classes_current_.insert(std::make_pair("TArrayF", 1));
}

void TestSchemaEvolution::runComparison() {
  CPPUNIT_ASSERT(unique_classes_current_.size() == unique_classes_.size());
  for (auto cl : unique_classes_) {
    std::cout << "Checking " << cl.first << " " << cl.second << std::endl;
    //std::cout << "unique_classes_current_.insert(std::make_pair(\"" << cl.first << "\", " << cl.second << "));" << std::endl;
    CPPUNIT_ASSERT(unique_classes_.find(cl.first) != unique_classes_.end());
    CPPUNIT_ASSERT(unique_classes_[cl.first] == unique_classes_current_[cl.first]);
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
    "TProfile", "TProfile2D", "TF1", 0
  };

  int i = 0;
  while (classes[i])
  {
    TClass *tcl = TClass::GetClass(classes[i]);
    if (!tcl)
      continue;
    unique_classes_.insert(std::make_pair(classes[i], tcl->GetClassVersion()));
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
      unique_classes_.insert(std::make_pair(cl->GetName(), cl->GetClassVersion()));
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
      unique_classes_.insert(std::make_pair(cl->GetName(), cl->GetClassVersion()));
      analyseClass(cl);
    }
  }
}


void TestSchemaEvolution::analyseClass(TClass *cl)
{
  loopOnBases(cl);
  loopOnDataMembers(cl);
}

