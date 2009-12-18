#include "TGLIncludes.h"
#include "TROOT.h"
#include "TColor.h"
#include "TGLUtil.h"
#include "TGLCamera.h"
#include "TGLRnrCtx.h"
#include "TGLSelectRecord.h"
#include "TGLViewerBase.h"
#include "TObjString.h"
#include "TGLViewer.h"
#include "TMath.h"


#include "TImage.h"
#include <KeySymbols.h>

#include "Fireworks/Core/interface/CmsAnnotation.h"

CmsAnnotation::CmsAnnotation(TGLViewerBase *parent, Float_t posx, Float_t posy) :
   TGLOverlayElement(TGLOverlayElement::kUser),

   fPosX(posx), fPosY(posy),
   fMouseX(0),  fMouseY(0),
   fDrag(kNone),
   fParent(0),
   fSize(0.15),
   fActive(false)
{
   // Constructor.
   // Create annotation as plain text

   parent->AddOverlayElement(this);
   fParent = (TGLViewer*)parent;
}


CmsAnnotation::~CmsAnnotation()
{
   // Destructor.

   fParent->RemoveOverlayElement(this);
}


void
CmsAnnotation::Render(TGLRnrCtx& rnrCtx)
{
   static UInt_t ttid = 0;

   if (ttid == 0)
   {
      TImage* img = TImage::Open("$(CMSSW_BASE)/src/Fireworks/Core/icons/CMSlogo.png");
      // !!!!! Check if 0 !!!!!

      glGenTextures(1, &ttid);

      glBindTexture(GL_TEXTURE_2D, ttid);

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

      glPixelStorei(GL_UNPACK_ALIGNMENT,  1);
      glPixelStorei(GL_UNPACK_SWAP_BYTES, 1);

      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img->GetWidth(), img->GetHeight(), 0,
                   GL_BGRA, GL_UNSIGNED_BYTE, img->GetArgbArray());

      glPixelStorei(GL_UNPACK_SWAP_BYTES, 0);

      delete img;
   }


   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POLYGON_BIT );
   TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);


   // reset matrix
   glMatrixMode(GL_PROJECTION);
   glPushMatrix();
   glLoadIdentity();

   if (rnrCtx.Selection())
   {
      TGLRect rect(*rnrCtx.GetPickRectangle());
      rnrCtx.GetCamera()->WindowToViewport(rect);
      gluPickMatrix(rect.X(), rect.Y(), rect.Width(), rect.Height(),
                    (Int_t*) rnrCtx.GetCamera()->RefViewport().CArr());
   }
   const TGLRect& vp = rnrCtx.RefCamera().RefViewport();
   glOrtho(vp.X(), vp.Width(), vp.Y(), vp.Height(), 0, 1);
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glLoadIdentity();

   // move to pos
   Float_t posX = vp.Width()  * fPosX;
   Float_t posY = vp.Height() * fPosY;
   glTranslatef(posX, posY, -0.99);

   glDisable(GL_DEPTH_TEST);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   glEnable(GL_TEXTURE_2D);
   glBindTexture(GL_TEXTURE_2D, ttid);
   glTexEnvi (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_BLEND);
   // env color
   UChar_t* fg =  rnrCtx.ColorSet().Foreground().Arr();
   GLfloat col[4] = { fg[0]/256.0, fg[1]/256.0, fg[2]/256.0, fg[3]/256.0};
   glTexEnvfv (GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, col);

   // logo
   glPushName(kMove);
   TGLUtil::Color(rnrCtx.ColorSet().Background().GetColorIndex());
   glBegin(GL_QUADS);
   Float_t z =  0.9;
   Float_t a = fSize * vp.Width();
   glTexCoord2f(0, 1); glVertex3f( 0, -a, z);
   glTexCoord2f(1, 1); glVertex3f( a, -a, z);
   glTexCoord2f(1, 0); glVertex3f( a,  0, z);
   glTexCoord2f(0, 0); glVertex3f( 0,  0, z);  
   glEnd();
   glPopName();


   glDisable(GL_TEXTURE_2D);

   if (fActive)
   {
      glTranslatef(a, -a, 0);
      a *= 0.15;
      z = 0.95;
      // resize button
      glPushName(kResize);
      TGLUtil::ColorTransparency(kGreen, 60);
      glBegin(GL_QUADS);
      glVertex3f( 0, 0, z);
      glVertex3f( 0, a, z);
      glVertex3f(-a, a, z);
      glVertex3f(-a, 0, z);
      glEnd();
      glPopName();
      // delete
      glPushName(7);
      glTranslatef(0, a, 0);
      TGLUtil::ColorTransparency(kRed, 60);
      glBegin(GL_QUADS);
      glVertex3f( 0, 0, z);
      glVertex3f( 0, a, z);
      glVertex3f(-a, a, z);
      glVertex3f(-a, 0, z);
      glEnd();
      glPopName();
   }

  

   glEnable(GL_DEPTH_TEST);
   glMatrixMode(GL_PROJECTION);
   glPopMatrix();
   glMatrixMode(GL_MODELVIEW);
   glPopMatrix();

   glPopAttrib();
}


Bool_t CmsAnnotation::Handle(TGLRnrCtx&          rnrCtx,
                             TGLOvlSelectRecord& selRec,
                             Event_t*            event)
{
   // Handle overlay event.
   // Return TRUE if event was handled.

   if (selRec.GetN() < 2) return kFALSE;
   int recID = selRec.GetItem(1);

   switch (event->fType)
   {
      case kButtonPress:
      {
         fMouseX = event->fX;
         fMouseY = event->fY;
         fDrag = (recID == kResize) ? kResize : kMove;
         return kTRUE;
      }
      case kButtonRelease:
      {
         fDrag = kNone;
         if (recID == 7)
         {
            delete this;
            fParent->RequestDraw(rnrCtx.ViewerLOD());
         }
      }
      case kMotionNotify:
      {
         const TGLRect& vp = rnrCtx.RefCamera().RefViewport();
         if (vp.Width() == 0 || vp.Height() == 0) return false;
         if (fDrag != kNone)
         {
            if (fDrag == kMove)
            {
               fPosX += (Float_t)(event->fX - fMouseX) / vp.Width();
               fPosY -= (Float_t)(event->fY - fMouseY) / vp.Height();
               fMouseX = event->fX;
               fMouseY = event->fY;

               Float_t w = fSize;
               Float_t h = fSize*vp.Aspect();

               // Make sure we don't go offscreen (use fDraw variables set in draw)
               if (fPosX < 0)
                  fPosX = 0;
               else if (fPosX +w > 1.0f)
                  fPosX = 1.0f - w;
               if (fPosY < h)
                  fPosY = h;
               else if (fPosY > 1.0f)
                  fPosY = 1.0f;
            }
            else
            {
               fMouseX = event->fX;
               fMouseY = event->fY;
               Float_t dX = (Float_t)(fMouseX) / vp.Width() - fPosX;
               Float_t dY = 1.0f - (Float_t)(fMouseY) / vp.Height() + fPosY + 0.5f*fSize;
               fSize = TMath::Min(dX, dY);
               fSize = TMath::Max(fSize, 0.05f); // lock above min
               fSize = TMath::Min(fSize, 0.50f); // lock below max
            }
         }
         return kTRUE;
      }
      default:
      {
         return kFALSE;
      }
   }
}

//______________________________________________________________________________
Bool_t CmsAnnotation::MouseEnter(TGLOvlSelectRecord& /*rec*/)
{
   // Mouse has entered overlay area.

   fActive = kTRUE;
   return kTRUE;
}

//______________________________________________________________________
void CmsAnnotation::MouseLeave()
{
   // Mouse has left overlay area.

   fActive = kFALSE;
}
