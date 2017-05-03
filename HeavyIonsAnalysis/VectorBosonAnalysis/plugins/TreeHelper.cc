#include "TreeHelper.h"
#include "TLeaf.h"
#include <assert.h>
//Clear event vectors
void TreeHelper::clear(){
  clear(intList_);
  clear(unsignedList_);
  clear(unsigned64List_);
  clear(floatList_);
  clear(doubleList_);
  clear(intVectorList_);
  clear(unsignedVectorList_);
  clear(unsigned64VectorList_);
  clear(floatVectorList_);
  clear(doubleVectorList_);
  clear(boolVectorList_);
}

void TreeHelper::defineBit(const char* branchName, int bit, const char* bitDescription){
  if(!bitFieldTree_) return;
  if(bit < 0 || bit > 63) return;
  TBranch* br = bitFieldTree_->FindBranch(branchName);
  std::vector<std::string>* vec;
  if(!br){
    vec = new std::vector<std::string>(64);
    allocatedStringVectors_.push_back(vec);
    bitFieldTree_->Branch(branchName, vec);
  } else{
    TLeaf* l = br->GetLeaf(branchName);
    assert(l);
    vec = (std::vector<std::string>*) l->GetValuePointer();
  }
  (*vec)[bit] = bitDescription;
}
