#include "Fireworks/Core/src/FWGeoTopNodeGL.h"
#include "Fireworks/Core/interface/FWGeoTopNode.h"

#include "TGLIncludes.h"
#include "TGLRnrCtx.h"
#include "TGLViewer.h"

//______________________________________________________________________________
FWGeoTopNodeGL::FWGeoTopNodeGL() :
   TGLObject()
{
   // Constructor.
}

//______________________________________________________________________________
void FWGeoTopNodeGL::SetBBox()
{
   // Set bounding box.

   SetAxisAlignedBBox(((FWGeoTopNode*)fExternalObj)->AssertBBox());
}

//______________________________________________________________________________
Bool_t FWGeoTopNodeGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   fM = SetModelDynCast<FWGeoTopNode>(obj);
   return kTRUE;
}

//______________________________________________________________________________
void FWGeoTopNodeGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // Draw quad-set with GL.

   static const TEveException eH("TEveQuadSetGL::DirectDraw ");

   // printf("FWGeoTopNodeGL::DirectDraw\n");

   // glPushAttrib(GL_POINT_BIT);
   // glPointSize(20);
   // glBegin(GL_POINTS);
   // glVertex3d(1,1,1);
   // glEnd();
   // glPopAttrib();
}

//______________________________________________________________________________
void FWGeoTopNodeGL::ProcessSelection(TGLRnrCtx& rnrCtx, TGLSelectRecord& rec)
{
   // Processes secondary selection from TGLViewer.
   // Calls DigitSelected(Int_t) in the model object with index of
   // selected point as the argument.

   // printf("FWGeoTopNodeGL::ProcessSelection who knows what we've got ...\n");
   // rec.Print();

   TGLViewer *v = dynamic_cast<TGLViewer*>(rnrCtx.GetViewer());
   /*
   if (v)
   {
      printf("  but we know the first selection was what we actually want!\n");
      printf("  and this is in rnrctx.viewer.selrec\n");
      printf("  log=%p, this=%p\n", v->GetSelRec().GetLogShape(), this);
   }
   */
   TGLPhysicalShape *p = v->GetSelRec().GetPhysShape();

   if (rec.GetHighlight())
   {
      fM->ProcessSelection(rec, fM->fHted, p);
   }
   else
   {
      fM->ProcessSelection(rec, fM->fSted, p);
   }

   // Also, do something in UnSelected / UnHighlighted XXXXX
}
