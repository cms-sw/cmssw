#include "TTreeCache.h"

class FWTTreeCache : public TTreeCache
{

public:
   FWTTreeCache();
   FWTTreeCache(TTree *tree, Int_t buffersize=0);
   virtual ~FWTTreeCache();

   // virtual, these two return Int_t in root-6 master
   void AddBranch(TBranch *b, Bool_t subbranches = kFALSE);
   void AddBranch(const char *branch, Bool_t subbranches = kFALSE);

   // virtual
   Int_t ReadBuffer        (char *buf, Long64_t pos, Int_t len);
   Int_t ReadBufferNormal  (char *buf, Long64_t pos, Int_t len);
   Int_t ReadBufferPrefetch(char *buf, Long64_t pos, Int_t len);
};
