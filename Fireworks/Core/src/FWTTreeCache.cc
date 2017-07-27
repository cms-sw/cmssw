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

void FWTTreeCache::AddBranch(TBranch *b, Bool_t subbranches)
{
   // printf("FWTTreeCache::AddBranch %s\n", b->GetName());

   TTreeCache::AddBranch(b, subbranches);
}

void FWTTreeCache::AddBranch(const char *branch, Bool_t subbranches)
{
   // printf("FWTTreeCache::AddBranch %s\n", branch);

   TTreeCache::AddBranch(branch, subbranches);
}

//==============================================================================

Int_t FWTTreeCache::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
   // printf("FWTTreeCache::ReadBuffer \n");

   if (!fEnabled) return 0;

   if (fEnablePrefetching)
      return ReadBufferPrefetch(buf, pos, len);
   else
      return ReadBufferNormal(buf, pos, len);
}

Int_t FWTTreeCache::ReadBufferNormal(char *buf, Long64_t pos, Int_t len)
{
   // printf("FWTTreeCache::ReadBufferNormal \n");

   //Is request already in the cache?
   if (TFileCacheRead::ReadBuffer(buf,pos,len) == 1){
      fNReadOk++;
      return 1;
   }

   //not found in cache. Do we need to fill the cache?
   Bool_t bufferFilled = FillBuffer();
   if (bufferFilled) {
      Int_t res = TFileCacheRead::ReadBuffer(buf,pos,len);

      if (res == 1)
         fNReadOk++;
      else if (res == 0)
         fNReadMiss++;

      return res;
   }
   fNReadMiss++;

   return 0;
}

Int_t FWTTreeCache::ReadBufferPrefetch(char *buf, Long64_t pos, Int_t len)
{
   // printf("FWTTreeCache::ReadBufferPrefetch \n");
   
   if (TFileCacheRead::ReadBuffer(buf, pos, len) == 1){
      //call FillBuffer to prefetch next block if necessary
      //(if we are currently reading from the last block available)
      FillBuffer();
      fNReadOk++;
      return 1;
   }

   //keep on prefetching until request is satisfied
   // try to prefetch a couple of times and if request is still not satisfied then
   // fall back to normal reading without prefetching for the current request
   Int_t counter = 0;
   while (1) {
      if(TFileCacheRead::ReadBuffer(buf, pos, len)) {
         break;
      }
      FillBuffer();
      fNReadMiss++;
      counter++;
      if (counter>1) {
        return 0;
      }
   }

   fNReadOk++;
   return 1;
}
