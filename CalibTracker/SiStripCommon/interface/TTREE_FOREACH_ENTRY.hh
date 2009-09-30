#include <cassert>

#define LEAF(leaf)                                                \
    leaf = 0;                                                     \
    assert(TTREE_FOREACH_ENTRY_tree->FindLeaf( #leaf ) != 0);     \
    TTREE_FOREACH_ENTRY_tree->SetBranchStatus( #leaf ,1);         \
    TTREE_FOREACH_ENTRY_tree->SetBranchAddress( #leaf , &(leaf)); \

#define RENAMED_LEAF(var, leafname)                               \
    var = 0;                                                      \
    { /*scope to evaluate leafname no more than once*/            \
      const char* name = leafname ;                               \
      assert(TTREE_FOREACH_ENTRY_tree->FindLeaf( name ) != 0);    \
      TTREE_FOREACH_ENTRY_tree->SetBranchStatus( name , 1);       \
      TTREE_FOREACH_ENTRY_tree->SetBranchAddress( name , &(var)); \
    }                                                             \


#define TTREE_FOREACH_ENTRY(tree, leaves)                                       \
  TTree* TTREE_FOREACH_ENTRY_tree = tree;                                       \
  TTREE_FOREACH_ENTRY_tree->SetBranchStatus("*",0);                             \
  leaves                                                                        \
  for ( Long64_t                                                                \
	  TTREE_FOREACH_ENTRY_index=0,                                          \
	  TTREE_FOREACH_ENTRY_total=TTREE_FOREACH_ENTRY_tree->GetEntries();     \
	TTREE_FOREACH_ENTRY_index < TTREE_FOREACH_ENTRY_total                   \
	  && -1 < TTREE_FOREACH_ENTRY_tree->LoadTree(TTREE_FOREACH_ENTRY_index) \
	  && -1 < TTREE_FOREACH_ENTRY_tree->GetEntry(TTREE_FOREACH_ENTRY_index);\
	TTREE_FOREACH_ENTRY_index++ )                                           \

