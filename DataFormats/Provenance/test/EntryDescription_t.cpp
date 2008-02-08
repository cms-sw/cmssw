#include "DataFormats/Provenance/interface/EntryDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include <assert.h>

// In addition to the tests here, the EntryDescription
// is also tested in FWCore/Integration/test/run_RunMerge.sh

int main()
{
  edm::EntryDescription ed1;
  assert(ed1 == ed1);
  edm::EntryDescription ed2;
  assert(ed1 == ed2);

  ed2.parents_ = std::vector<edm::ProductID>(1);
  edm::EntryDescription ed3;
  ed3.parents_ = std::vector<edm::ProductID>(2);

  edm::EntryDescriptionID id1 = ed1.id();
  edm::EntryDescriptionID id2 = ed2.id();
  edm::EntryDescriptionID id3 = ed3.id();

  assert(id1 != id2);
  assert(ed1 != ed2);
  assert(id1 != id3);
  assert(ed1 != ed3);
  assert(id2 != id3);
  assert(ed2 != ed3); 

  edm::EntryDescription ed4;
  ed4.parents_ = std::vector<edm::ProductID>(1);
  edm::EntryDescriptionID id4 = ed4.id();
  assert(ed4 == ed2);
  assert (id4 == id2);

  // Test Merging

  edm::ModuleDescription md1;
  md1.moduleName_ = "class1";

  edm::ModuleDescription md2;
  md2.moduleName_ = "class2";

  edm::EntryDescription ed101;
  ed101.moduleDescriptionID_ = md1.id();

  edm::EntryDescription ed102;
  ed102.moduleDescriptionID_ = md1.id();

  edm::EntryDescription ed103;
  ed103.moduleDescriptionID_ = md1.id();

  ed101.mergeEntryDescription(&ed102);
  assert(ed101 == ed103);

  edm::EntryDescription ed104;
  ed104.moduleDescriptionID_ = md2.id();

  ed101.mergeEntryDescription(&ed104);
  assert(ed101 != ed103);
  assert(ed101.moduleDescriptionID_ == edm::ModuleDescriptionID());

  edm::ProductID pid1(1); 
  edm::ProductID pid2(2); 
  edm::ProductID pid3(3);

  ed101.parents_.push_back(pid3);
  ed101.parents_.push_back(pid2);

  ed102.parents_.push_back(pid2);
  ed102.parents_.push_back(pid1);

  ed101.mergeEntryDescription(&ed102);
  assert(ed101.parents_[0] == pid1);
  assert(ed101.parents_[1] == pid2);
  assert(ed101.parents_[2] == pid3);  
}
