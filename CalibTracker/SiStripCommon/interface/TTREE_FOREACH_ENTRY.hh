#ifndef TTREE_FOREACH_ENTRY_H
#define TTREE_FOREACH_ENTRY_H

#include <cassert>
#include <string>
#include <algorithm>
#include <iomanip>
#include "TTree.h"

#define LEAF(leaf, tree)						\
  leaf = 0;								\
  { TBranch* branch = tree->GetBranch( #leaf ); assert(branch);		\
    branch->SetAddress( &(leaf) );					\
    branch->GetEntry( TFE_local_index,1); }				\

#define PLEAF(leaf, tree)						\
  leaf;									\
  { TBranch* branch = tree->GetBranch( #leaf ); assert(branch);		\
    void* pt = &(leaf);							\
    branch->SetAddress( &pt );						\
    branch->GetEntry( TFE_local_index,1); }				\

#define RENAMED_LEAF(var,leafname, tree)				\
  var = 0;								\
  { std::string n=leafname;						\
    TBranch* branch = tree->GetBranch( n.c_str() ); assert(branch);	\
    branch->SetAddress( &(var) );					\
    branch->GetEntry( TFE_local_index,1); }				\

#define RENAMED_PLEAF(var,leafname, tree)				\
  var;									\
  { std::string n=leafname;						\
    TBranch* branch = tree->GetBranch( n.c_str() ); assert(branch);	\
    void* pt = &(var);							\
    branch->SetAddress( &pt );						\
    branch->GetEntry( TFE_local_index,1); }				\

  
#define TTREE_FOREACH_ENTRY(tree)					\
  for ( Long64_t							\
	  TFE_index=0,							\
	  TFE_total=tree->GetEntries(),					\
	  TFE_local_index=0,						\
	  TFE_local_total=0,						\
	  TFE_freq=TFE_total/100;					\
	TFE_index < TFE_total &&					\
	  ( TFE_local_index < TFE_local_total ||			\
	    (-1 < tree->LoadTree(TFE_index) &&				\
	     -1 < ( TFE_local_index = 0) &&				\
	     -1 < ( TFE_local_total = tree->GetTree()->GetEntries())	\
	     ));							\
	TFE_index++, TFE_local_index++ )				\

    
#define TFE_MAX(max) if(TFE_index && max) { TFE_total = std::min(Long64_t(max),TFE_total); TFE_freq=TFE_total/100;}

#define TFE_PRINTSTATUS	{						\
    if ( TFE_freq && !(TFE_index%TFE_freq) ) {				\
      std::cout << "\b\b\b\b\b" << std::setw(3) << std::fixed		\
		<< (100*TFE_index)/TFE_total << "% " << std::flush;	\
    }									\
  }									\
    
#endif
