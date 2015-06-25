#ifndef __TREE_H__
#define __TREE_H__

#include <string>
#include <iostream>
//#include "TTree.h"
//#include "TFriendElement.h"
//#include "TList.h"


Int_t SetBranchAddress(TTree * tree, const std::string & name, 
		       const std::string & mother, void* addr);
TBranch * GetBranch(TTree * tree, const std::string & name, const std::string & mother);


#endif

#ifdef l1ntuple_cxx

TBranch * GetBranch(TTree * tree, const std::string & name, const std::string & mother)
{
  TObjArray * fBranches=tree->GetListOfBranches();
  Int_t nb=fBranches->GetEntriesFast();
  for (Int_t i=0;i<nb;i++)
    {
      TBranch * br = (TBranch*) fBranches->UncheckedAt(i);
 
      TObjArray * fBranches2=br->GetListOfBranches();
      Int_t nb2=fBranches2->GetEntriesFast();
      for (Int_t i2=0;i2<nb2;i2++)
      {
         TBranch * br2 = (TBranch*) fBranches2->UncheckedAt(i2);
         TBranch * brmum2 = br2->GetMother();
         if (brmum2==0) continue;
         if (std::string(br2->GetName())==name && std::string(brmum2->GetName())==mother)
	{
          return br2;
	}

      TObjArray * fBranches3=br2->GetListOfBranches();
      Int_t nb3=fBranches3->GetEntriesFast();
      for (Int_t i3=0;i3<nb3;i3++)
      {
         TBranch * br3 = (TBranch*) fBranches3->UncheckedAt(i);
         TBranch * brmum3 = br3->GetMother();
         if (brmum3==0) continue;
         if (std::string(br3->GetName())==name && std::string(brmum3->GetName())==mother)
	{
          return br3;
	}
      }
      }
    }
 
  return 0;
}

Int_t SetBranchAddress(TTree * tree, const std::string & name, 
                 const std::string & mother, void* addr)
{
  TBranch * branch = GetBranch(tree,name,mother);

  if (branch==0)
    {
      TList * myfriends = tree->GetListOfFriends();
      //      std::cout << "size="<<myfriends->GetSize()<<std::endl;
      TObjLink * lnk = myfriends -> FirstLink();
      TFriendElement * fe =0;
      while (lnk)
	{
          fe = (TFriendElement*) lnk->GetObject();
	  TTree* t = fe->GetTree();
          if (t)
	    {
	      //std::cout << "FriendName : "<<fe->GetName()<<std::endl;
	      branch=GetBranch(t,name,mother);
	      if (branch!=0) break;
	    }
          lnk=lnk->Next();
	}

    }

  if (branch==0)
    {
      std::cout << "ERROR unknown branch -> "<<name<<" from "<<mother<<std::endl;
      return -1;
    }

  branch->SetAddress(addr);
  return 0;
  


}



#endif
