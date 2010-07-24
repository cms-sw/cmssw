#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include <assert.h>

int main() {
  edm::Parentage ed1;
  assert(ed1 == ed1);
  edm::Parentage ed2;
  assert(ed1 == ed2);

  ed2.parents() = std::vector<edm::BranchID>(1);
  edm::Parentage ed3;
  ed3.parents() = std::vector<edm::BranchID>(2);

  edm::ParentageID id1 = ed1.id();
  edm::ParentageID id2 = ed2.id();
  edm::ParentageID id3 = ed3.id();

  assert(id1 != id2);
  assert(ed1 != ed2);
  assert(id1 != id3);
  assert(ed1 != ed3);
  assert(id2 != id3);
  assert(ed2 != ed3); 

  edm::Parentage ed4;
  ed4.parents() = std::vector<edm::BranchID>(1);
  edm::ParentageID id4 = ed4.id();
  assert(ed4 == ed2);
  assert (id4 == id2);
}
