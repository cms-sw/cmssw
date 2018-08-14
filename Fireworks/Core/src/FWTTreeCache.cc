#include "FWTTreeCache.h"

#include "TChain.h"
#include "TTree.h"
#include "TEventList.h"

#include <climits>

FWTTreeCache::FWTTreeCache() :
  TTreeCache()
{}

FWTTreeCache::FWTTreeCache(TTree *tree, Int_t buffersize) :
  TTreeCache(tree, buffersize)
{}

FWTTreeCache::~FWTTreeCache()
{}

//==============================================================================

bool FWTTreeCache::s_logging      = false;
bool FWTTreeCache::s_prefetching  = false;
int  FWTTreeCache::s_default_size = 50 * 1024 * 1024;

void FWTTreeCache::LoggingOn()  { s_logging = true;  }
void FWTTreeCache::LoggingOff() { s_logging = false; }
bool FWTTreeCache::IsLogging()  { return s_logging;  }

void FWTTreeCache::PrefetchingOn()  { s_prefetching = true;  }
void FWTTreeCache::PrefetchingOff() { s_prefetching = false; }
bool FWTTreeCache::IsPrefetching()  { return s_prefetching;  }

void FWTTreeCache::SetDefaultCacheSize(int def_size) { s_default_size = def_size; }
int  FWTTreeCache::GetDefaultCacheSize()             { return s_default_size; }

//==============================================================================

Int_t FWTTreeCache::AddBranchTopLevel (const char* bname)
{
   if (bname == nullptr || bname[0] == 0)
   {
      printf("FWTTreeCache::AddBranchTopLevel Invalid branch name <%s>\n",
             bname == nullptr ? "nullptr" : "empty string");
      return -1;
   }

   Int_t ret = 0;
   if ( ! is_branch_in_cache(bname))
   {
      if (s_logging)
         printf("FWTTreeCache::AddBranchTopLevel '%s' -- adding\n", bname);
      {  LearnGuard _lg(this);
         ret = AddBranch(bname, true);
      }
      if (ret == 0)
         m_branch_set.insert(bname);
   }
   else
   {
      if (s_logging)
         printf("FWTTreeCache::AddBranchTopLevel '%s' -- already in cache\n", bname);
   }

   return ret;
}

Int_t FWTTreeCache::DropBranchTopLevel(const char* bname)
{
   if (bname == nullptr || bname[0] == 0)
   {
      printf("FWTTreeCache::AddBranchTopLevel Invalid branch name");
      return -1;
   }

   Int_t ret = 0;
   if (is_branch_in_cache(bname))
   {
      if (s_logging)
         printf("FWTTreeCache::DropBranchTopLevel '%s' -- dropping\n", bname);
      m_branch_set.erase(bname);
      LearnGuard _lg(this);
      ret = DropBranch(bname, true);
   }
   else
   {
      if (s_logging)
         printf("FWTTreeCache::DropBranchTopLevel '%s' -- not in cache\n", bname);
   }
   return ret;
}

void FWTTreeCache::BranchAccessCallIn(const TBranch *b)
{
   if (s_logging)
      printf("FWTTreeCache::BranchAccessCallIn '%s'\n", b->GetName());

   AddBranchTopLevel(b->GetName());
}

//==============================================================================

Int_t FWTTreeCache::AddBranch(TBranch *b, Bool_t subbranches)
{
   if (s_logging && ! m_silent_low_level)
      printf("FWTTreeCache::AddBranch by ptr '%s', subbp=%d\n", b->GetName(), subbranches);
   return TTreeCache::AddBranch(b, subbranches);
}

Int_t FWTTreeCache::AddBranch(const char *branch, Bool_t subbranches)
{
   if (s_logging)
      printf("FWTTreeCache::AddBranch by name '%s', subbp=%d\n", branch, subbranches);
   if (strcmp(branch,"*") == 0)
      m_silent_low_level = true;
   return TTreeCache::AddBranch(branch, subbranches);
}

Int_t FWTTreeCache::DropBranch(TBranch *b, Bool_t subbranches)
{
   if (s_logging && ! m_silent_low_level)
      printf("FWTTreeCache::DropBranch by ptr '%s', subbp=%d\n", b->GetName(), subbranches);
   return TTreeCache::DropBranch(b, subbranches);
}

Int_t FWTTreeCache::DropBranch(const char *branch, Bool_t subbranches)
{
   if (s_logging)
      printf("FWTTreeCache::DropBranch by name '%s', subbp=%d\n", branch, subbranches);
   Int_t ret = TTreeCache::DropBranch(branch, subbranches);
   if (strcmp(branch,"*") == 0)
      m_silent_low_level = false;
   return ret;
}
