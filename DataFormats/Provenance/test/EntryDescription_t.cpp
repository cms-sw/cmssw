#include "DataFormats/Provenance/interface/EventEntryDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionID.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include <assert.h>

int main()
{
  edm::EventEntryDescription ed1;
  assert(ed1 == ed1);
  edm::EventEntryDescription ed2;
  assert(ed1 == ed2);

  ed2.parents() = std::vector<edm::BranchID>(1);
  edm::EventEntryDescription ed3;
  ed3.parents() = std::vector<edm::BranchID>(2);

  edm::EntryDescriptionID id1 = ed1.id();
  edm::EntryDescriptionID id2 = ed2.id();
  edm::EntryDescriptionID id3 = ed3.id();

  assert(id1 != id2);
  assert(ed1 != ed2);
  assert(id1 != id3);
  assert(ed1 != ed3);
  assert(id2 != id3);
  assert(ed2 != ed3); 

  edm::EventEntryDescription ed4;
  ed4.parents() = std::vector<edm::BranchID>(1);
  edm::EntryDescriptionID id4 = ed4.id();
  assert(ed4 == ed2);
  assert (id4 == id2);
}
