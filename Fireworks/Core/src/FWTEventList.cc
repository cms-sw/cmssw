#include "TMath.h"
#include "TCut.h"

#include "Fireworks/Core/interface/FWTEventList.h"

//______________________________________________________________________________
void FWTEventList::Add(const TEventList *alist)
{
   // Merge contents of alist with this list.
   //
   // Both alist and this list are assumed to be sorted prior to this call

   Int_t i;
   Int_t an = alist->GetN();
   if (!an) return;
   Long64_t *alst = alist->GetList();
   if (!fList) {
      fList = new Long64_t[an];
      for (i=0;i<an;i++) fList[i] = alst[i];
      fN = an;
      fSize = an;
      return;
   }
   Int_t newsize = fN + an;
   Long64_t *newlist = new Long64_t[newsize];
   Int_t newpos, alpos;
   newpos = alpos = 0;
   for (i=0;i<fN;i++) {
      while (alpos < an && fList[i] > alst[alpos]) {
         newlist[newpos] = alst[alpos];
         newpos++;
         alpos++;
      }
      if (alpos < an && fList[i] == alst[alpos]) alpos++;
      newlist[newpos] = fList[i];
      newpos++;
   }
   while (alpos < an) {
      newlist[newpos] = alst[alpos];
      newpos++;
      alpos++;
   }
   delete [] fList;
   fN    = newpos;
   fSize = newsize;
   fList = newlist;

   TCut orig = GetTitle();
   TCut added = alist->GetTitle();
   TCut updated = orig || added;
   SetTitle(updated.GetTitle());
}

//______________________________________________________________________________
void FWTEventList::Enter(Long64_t entry)
{
   // Enter element entry into the list.

   if (!fList) {
      fList = new Long64_t[fSize];
      fList[0] = entry;
      fN = 1;
      return;
   }
   if (fN>0 && entry==fList[fN-1]) return;
   if (fN >= fSize) {
      Int_t newsize = TMath::Max(2*fSize,fN+fDelta);
      Resize(newsize-fSize);
   }
   if(fN==0 || entry>fList[fN-1]) {
      fList[fN] = entry;
      ++fN;
   } else {
      Int_t pos = TMath::BinarySearch(fN, fList, entry);
      if(pos>=0 && entry==fList[pos])
         return;
      ++pos;
      memmove( &(fList[pos+1]), &(fList[pos]), 8*(fN-pos));
      fList[pos] = entry;
      ++fN;
   }
}
