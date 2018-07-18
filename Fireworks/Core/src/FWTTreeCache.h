#ifndef Fireworks_Core_FWTTreeCache_h
#define Fireworks_Core_FWTTreeCache_h

#include "TTreeCache.h"

#include <set>
#include <string>

class FWTTreeCache : public TTreeCache
{
   std::set<std::string> m_branch_set;
   bool                  m_silent_low_level = false;

   static int            s_default_size;
   static bool           s_logging;
   static bool           s_prefetching;

protected:
   bool is_branch_in_cache(const char* name) { return (m_branch_set.find(name) != m_branch_set.end()); }

   bool start_learning() { if (fIsLearning) return true; fIsLearning = true; return false; }
   void stop_learning()  { fIsLearning = false; }

   struct LearnGuard
   {
      FWTTreeCache *f_tc;
      bool          f_was_learning;
      bool          f_was_silent_ll;

      LearnGuard(FWTTreeCache *tc) : f_tc(tc)
      {
         f_was_learning = f_tc->start_learning();
         f_was_silent_ll = f_tc->m_silent_low_level;
         f_tc->m_silent_low_level = true;
      }
      ~LearnGuard()
      {
         if ( ! f_was_learning) f_tc->stop_learning();
         f_tc->m_silent_low_level = f_was_silent_ll;
      }
   };
   
public:
   FWTTreeCache();
   FWTTreeCache(TTree *tree, Int_t buffersize=0);
   ~FWTTreeCache() override;

   static void LoggingOn();
   static void LoggingOff();
   static bool IsLogging();
   static void PrefetchingOn();
   static void PrefetchingOff();
   static bool IsPrefetching();
   static void SetDefaultCacheSize(int def_size);
   static int  GetDefaultCacheSize();

   Int_t AddBranchTopLevel (const char* bname);
   Int_t DropBranchTopLevel(const char* bname);

   void  BranchAccessCallIn(const TBranch *b);

   // virtuals from TTreeCache, just wrappers for info printouts
   Int_t AddBranch(TBranch *b, Bool_t subbranches = kFALSE) override;
   Int_t AddBranch(const char *branch, Bool_t subbranches = kFALSE) override;
   Int_t DropBranch(TBranch *b, Bool_t subbranches = kFALSE) override;
   Int_t DropBranch(const char *branch, Bool_t subbranches = kFALSE) override;
};

#endif
