#include "Fireworks/TableWidget/interface/GlobalContexts.h"

#include "TGClient.h"
#include "TVirtualX.h"
#include "TSystem.h"
#include "TGFont.h"
#include "TGResourcePool.h"
#include "TGGC.h"

namespace fireworks
{ 
const TGGC& boldGC()
{
   static TGGC s_boldGC(*gClient->GetResourcePool()->GetFrameGC());
 
   TGFontPool *pool = gClient->GetFontPool();
   //TGFont *font = pool->FindFontByHandle(s_boldGC.GetFont());
   //FontAttributes_t attributes = font->GetFontAttributes();
  
   /*
     This doesn't seem to work:
     attributes.fWeight = 1; 
     TGFont *newFont = pool->GetFont(attributes.fFamily, 9,
     attributes.fWeight, attributes.fSlant);

     But this does:
   */
  
   TGFont* newFont = pool->GetFont("-*-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");
  
   if ( ! newFont )
      return s_boldGC;

   s_boldGC.SetFont(newFont->GetFontHandle());
  
   return s_boldGC;
}

const TGGC& greenGC()
{
   static TGGC s_greenGC(*gClient->GetResourcePool()->GetFrameGC());
   s_greenGC.SetForeground(gVirtualX->GetPixel(kGreen-5));
   return s_greenGC;
}
  
const TGGC& redGC()
{
   static TGGC s_redGC(*gClient->GetResourcePool()->GetFrameGC());
   s_redGC.SetForeground(gVirtualX->GetPixel(kRed-5));
   return s_redGC;
}

const TGGC& italicGC()
{
   static TGGC s_italicGC(*gClient->GetResourcePool()->GetFrameGC());
 
   TGFontPool *pool = gClient->GetFontPool();
   TGFont *font = pool->FindFontByHandle(s_italicGC.GetFont());
   FontAttributes_t attributes = font->GetFontAttributes();
  
   attributes.fSlant = 1;
   TGFont *newFont = pool->GetFont(attributes.fFamily, 9,
                                   attributes.fWeight, attributes.fSlant);
   
   s_italicGC.SetFont(newFont->GetFontHandle());
  
   return s_italicGC;
}
}
