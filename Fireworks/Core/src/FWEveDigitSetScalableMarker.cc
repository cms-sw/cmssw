#include "TGLIncludes.h"
#include "Fireworks/Core/src/FWEveDigitSetScalableMarker.h"

void 
FWEveDigitSetScalableMarkerGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT | GL_POINT_BIT);
   glEnable(GL_POINT_SMOOTH);
   glDisable(GL_LIGHTING);
          
   TEveChunkManager::iterator qi(fM->GetPlex());
   if (rnrCtx.Highlight() && fHighlightSet)
      qi.fSelection = fHighlightSet;
      
   if (rnrCtx.SecSelection()) glPushName(0);
      
   glPointSize(((FWEveDigitSetScalableMarker*)fM)->GetMarkerSize());
   while (qi.next()) {
      TEveQuadSet::QFreeQuad_t* q =  (TEveQuadSet::QFreeQuad_t*) qi();
      if (q->fValue < 0)
         continue;
      TGLUtil::ColorAlpha(Color_t(q->fValue));
      if (rnrCtx.SecSelection()) glLoadName(qi.index());
      float* p = &q->fVertices[0];
      glBegin(GL_LINES);
      float c[3]  =  {0.5f*(p[0]+p[6]), 0.5f*(p[1]+p[7]), 0.5f*(p[2]+p[8])};
         
      float d = p[6] - p[0];
      glVertex3f( c[0] -d, c[1], c[2]); glVertex3f(c[0] + d, c[1], c[2]);
      glVertex3f( c[0] , c[1] -d, c[2]); glVertex3f(c[0] , c[1] +d, c[2]);
      glVertex3f( c[0] , c[1], c[2]-d); glVertex3f(c[0] , c[1], c[2] +d);
         
      glEnd();
         
      glBegin(GL_POINTS);
      glVertex3fv(&c[0]);
      glEnd();
         
   }
      
   glPopAttrib();
}
