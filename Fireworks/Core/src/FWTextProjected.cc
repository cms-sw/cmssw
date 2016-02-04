// -*- C++ -*-
//
// Package:     Core
// Class  :     FWTextProjected
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Aug 12 01:12:18 CEST 2011
// $Id: FWTextProjected.cc,v 1.3 2011/08/16 21:43:27 amraktad Exp $
//

// system include files

// user include files
#include <iostream>

#include "Fireworks/Core/interface/FWTextProjected.h"
#include "TEveProjectionManager.h"
#include "TEveTrans.h"
#include "TGLBoundingBox.h"

#include "TGLIncludes.h"

#include "TGLRnrCtx.h"
#include "TGLUtil.h"
#include "TGLCamera.h"
//______________________________________________________________________________
TClass* FWEveText::ProjectedClass(const TEveProjection*) const
{
   // Virtual from TEveProjectable, returns TEvePointSetProjected class.

   return FWEveTextProjected::Class();
}


//______________________________________________________________________________
void FWEveTextProjected::UpdateProjection()
{
   //   printf("update projection \n");

   FWEveText      & als  = * dynamic_cast<FWEveText*>(fProjectable);
   TEveTrans      *tr   =   als.PtrMainTrans(kFALSE);

   fText = als.GetText();
   *fMainColorPtr = als.GetMainColor();
   float pos[3];
   tr->GetPos(pos);

   TEveProjection& proj = * fManager->GetProjection();
   proj.ProjectPoint(pos[0],pos[1], pos[2], fDepth);

   RefMainTrans().SetPos(pos[0], pos[1], pos[2] + als.m_offsetZ);
}
//==============================================================================


void FWEveTextGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{

   Int_t fm = fM->GetFontMode();
   if (fm == TGLFont::kBitmap || fm == TGLFont::kPixmap || fm == TGLFont::kTexture)
      rnrCtx.RegisterFont(fM->GetFontSize(), fM->GetFontFile(), fM->GetFontMode(), fFont);
   else
      rnrCtx.RegisterFontNoScale(fM->GetFontSize(), fM->GetFontFile(), fM->GetFontMode(), fFont);


   // rendering
   glPushMatrix();
   fFont.PreRender(fM->GetAutoLighting(), fM->GetLighting());
  

   const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr(); 

   GLdouble mm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetIntegerv(GL_VIEWPORT, vp);

   fX[0][0] = fX[0][1] = fX[0][2] = 0;
   GLdouble x, y, z;
   gluProject(fX[0][0], fX[0][1], fX[0][2], mm, pm, vp, &x, &y, &z);
   Float_t bbox[6];
   fFont.BBox(fM->GetText(), bbox[0], bbox[1], bbox[2],
              bbox[3], bbox[4], bbox[5]);

 
   gluUnProject(x + bbox[0], y + bbox[1], z, mm, pm, vp, &fX[0][0], &fX[0][1], &fX[0][2]);
   gluUnProject(x + bbox[3], y + bbox[1], z, mm, pm, vp, &fX[1][0], &fX[1][1], &fX[1][2]);
   gluUnProject(x + bbox[3], y + bbox[4], z, mm, pm, vp, &fX[2][0], &fX[2][1], &fX[2][2]);
   gluUnProject(x + bbox[0], y + bbox[4], z, mm, pm, vp, &fX[3][0], &fX[3][1], &fX[3][2]);
   glEnable(GL_POLYGON_OFFSET_FILL);
           
   FWEveText* model = (FWEveText*)fM; 
   double xm =  fX[0][0] - model->m_textPad;
   double xM =  fX[2][0] + model->m_textPad;
   double ym =  fX[0][1] - model->m_textPad;
   double yM =  fX[2][1] + model->m_textPad;


   //   TGLUtil::Color(1016);
   if (rnrCtx.ColorSet().Background().GetRed())
      TGLUtil::Color(kWhite);
   else
      TGLUtil::Color(kBlack);

   glPolygonOffset(1,1 );
   glBegin(GL_POLYGON);
   glVertex2d(xm, ym);
   glVertex2d(xM, ym);
   glVertex2d(xM, yM);
   glVertex2d(xm, yM);

   glEnd();
 
   TGLUtil::Color(fM->GetMainColor());
   if (1) {
      glPolygonOffset(0, 0 );
      glBegin(GL_LINE_LOOP);
      glVertex2d(xm, ym);
      glVertex2d(xM, ym);
      glVertex2d(xM, yM);
      glVertex2d(xm, yM);
      glEnd();
   }

   glPolygonOffset(0, 0  );

   glRasterPos3i(0, 0, 0);
   fFont.Render(fM->GetText());
   fFont.PostRender();
   glPopMatrix();
}
