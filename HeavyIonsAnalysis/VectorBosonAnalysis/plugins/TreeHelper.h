#ifndef TREEHELPER_H
#define TREEHELPER_H

/** Class to handle ROOT tree leaves.
 */

#include "TTree.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include "Rtypes.h"

#include <iostream> //for debug messages

/** Helper class for TTree. This helper takes care of resetting
 * the variable storage after a tree fill. The values of the simple
 * type variable are set to 0 and the vectors are empty.
 * Supported simple types: bool, int, float, double
 * Supported vector types: std::vector of above simple types
 */
class TreeHelper{
public:
  /** Constructor
   * @param tree ROOT tree to store the events
   * @param descTree secondary ROOT tree to store the description of the
   * branches of the first tree. This tree will be filled with one entry
   * only. It is ignored if the pointer is null
   * @param bitFiledTree secondary ROOT tree to store the description of 
   * the individual bits of bit field branches contained in the first tree.
   * This tree will be filled with one entry only. It is ignored if the pointer
   * is null
   */
  TreeHelper(TTree* tree, TTree* descTree = 0, TTree* bitFieldTree = 0):
    tree_(tree), descTree_(descTree), bitFieldTree_(bitFieldTree){
  }

  /** Destructor
   */
  ~TreeHelper(){    
    while(!descriptions_.empty()){
      delete[] descriptions_.back();
      descriptions_.pop_back();
    }
    
    while(!allocatedStringVectors_.empty()){
      delete allocatedStringVectors_.back();
      allocatedStringVectors_.pop_back();
    }
  }
  
  /** Add a branch of one the supported vector types
   * @param branchName name of the new branch
   * @param v auto pointer to store the variable attached to the branch
   */
  template<typename T>
  void addBranch(const char* branchName, std::auto_ptr<std::vector<T> >& v,
		 const char* branchDescription = 0){
    std::vector<T>* p = new std::vector<T>;
    v = std::auto_ptr<std::vector<T> >(p);
    addVar(p);
    tree_->Branch(branchName, p);
    addDescription(branchName, branchDescription);
  }

  /** Add a branch of one the supported simple types
   * @param branchName name of the new branch
   * @param v auto pointer to store the variable attached to the branch
   */
  template<typename T>
  void addBranch(const char* branchName, std::auto_ptr<T>& v,
		 const char* branchDescription = 0){
    T* p = new T;
    v = std::auto_ptr<T> (p);
    *v = 0;
    addVar(p);
    tree_->Branch(branchName, v.get());
    addDescription(branchName, branchDescription);
  }

  void defineBit(const char* branchName, int bit, const char* bitDescription);

  /** Reset the variables attached to the tree branches
   */
  void clear();

  /** Fill tree and reset variables attached to the tree branches.
   * The description tree is filled at the first call.
   */
  void fill(){
    tree_->Fill();
    clear();
  }

  /** Add a description branch. This method is already called by
   * the addBranch methods when a description argument is provided.
   * A typical usage of direct call to this method is to provide
   * a description for a group of branches. In this case branchName
   * will designate the group.
   */
  void addDescription(const char* branchName, const char* description){
    if(descTree_ && description){
      descriptions_.push_back(new char[strlen(description) + 1]);
      strcpy(descriptions_.back(), description);
      descTree_->Branch(branchName, descriptions_.back(), (std::string(branchName) + "/C").c_str());
    }
  }


  /** To be called once all descriptionshave been entered
   * with the addDescription() and defineBit() methods.
   */
  void fillDescriptionTree(){
    if(descTree_){
      descTree_->Fill();
    }
    if(bitFieldTree_){
      bitFieldTree_->Fill();
    }
  }
  
private:

  //Clear std::vector
  //@param v list of the pointers to thevectors to clear provided as a std::vector.
  template<typename T>
  void clear(std::vector<std::vector<T>*>& v){
    for(typename std::vector<std::vector<T>*>::iterator it = v.begin(); it != v.end(); ++it){
      (*it)->clear();
    }
  }
  
  template<typename T>
  void clear(std::vector<T*>& v){
    for(typename std::vector<T*>::iterator it = v.begin(); it != v.end(); ++it){
      **it = 0;
    }
  }
  
  //add an element to the list of variables
  //to clear after a tree fill
  void addVar(std::vector<int>* v){
    intVectorList_.push_back(v);
  }
  void addVar(std::vector<unsigned>* v){
    unsignedVectorList_.push_back(v);
  }
  void addVar(std::vector<ULong64_t>* v){
    unsigned64VectorList_.push_back(v);
  }
  void addVar(std::vector<float>* v){
    floatVectorList_.push_back(v);
  }
  void addVar(std::vector<double>* v){
    doubleVectorList_.push_back(v);
  }
  void addVar(std::vector<bool>* v){
    boolVectorList_.push_back(v);
  }
  void addVar(int* a){
    intList_.push_back(a);
  }
  void addVar(unsigned* a){
    unsignedList_.push_back(a);
  }
  void addVar(ULong64_t* a){
    unsigned64List_.push_back(a);
  }
  void addVar(float* a){
    floatList_.push_back(a);
  }
  void addVar(double* a){
    doubleList_.push_back(a);
  }

private:
  std::vector<int*> intList_;
  std::vector<unsigned*> unsignedList_;
  std::vector<ULong64_t*> unsigned64List_;
  std::vector<float*> floatList_;
  std::vector<double*> doubleList_;
  std::vector<std::vector<int>*> intVectorList_;
  std::vector<std::vector<unsigned>*> unsignedVectorList_;
  std::vector<std::vector<ULong64_t>*> unsigned64VectorList_;
  std::vector<std::vector<float>*> floatVectorList_;
  std::vector<std::vector<double>*> doubleVectorList_;
  std::vector<std::vector<bool>*> boolVectorList_;
  std::vector<char*> descriptions_;
  std::vector<std::vector<std::string>*> allocatedStringVectors_;
  
  TTree* tree_;
  TTree* descTree_;
  TTree* bitFieldTree_;
};
  

#endif //TREEHELPER_H not defined
