#include "Fireworks/Vertices/interface/TEveEllipsoidGL.h"

#include "TEveProjections.h" // AMT missing getter for projection center / beam-spot

#include "Fireworks/Vertices/interface/TEveEllipsoidGL.h"
#include "Fireworks/Vertices/interface/TEveEllipsoid.h"
#include "TEveProjectionManager.h"

#include "TMath.h"

#include "TGLRnrCtx.h"

#include "TMatrixDEigen.h"
#include "TMatrixDSym.h"

#include "TDecompSVD.h"
#include "TGLIncludes.h"

//==============================================================================
// TEveEllipsoidGL
//==============================================================================

//______________________________________________________________________________
// OpenGL renderer class for TEveEllipsoid.
//



//______________________________________________________________________________
TEveEllipsoidGL::TEveEllipsoidGL() :
   TGLObject(), fE(nullptr)
{
   // Constructor.

   // fDLCache = kFALSE; // Disable display list.
}

//______________________________________________________________________________
Bool_t TEveEllipsoidGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   fE = SetModelDynCast<TEveEllipsoid>(obj);
   return kTRUE;
}

//______________________________________________________________________________
void TEveEllipsoidGL::SetBBox()
{
   // Set bounding box.
   ( (TEveEllipsoid*)fExternalObj)->ComputeBBox();
   SetAxisAlignedBBox(((TEveEllipsoid*)fExternalObj)->AssertBBox());
}

namespace {
GLUquadric* quad = nullptr; // !!!! AMT check why TGLQuadric crashes on mac
}

//______________________________________________________________________________
void TEveEllipsoidGL::DirectDraw(TGLRnrCtx& /*rnrCtx*/) const
{
   // Render with OpenGL.

   // printf("TEveEllipsoidGL::DirectDraw LOD %s\n", fE->GetName());

   glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT | GL_LIGHTING_BIT);
   glEnable(GL_NORMALIZE );
   if(!quad)
      quad = gluNewQuadric();

   glPushMatrix();

   TMatrixDSym xxx(3);
   for(int i=0;i<3;i++)
      for(int j=0;j<3;j++)
      {
         xxx(i,j) = fE->RefEMtx()(i+1,j+1);
      }
   TMatrixDEigen eig(xxx);


   // rewrite for multmatrix ....
   TEveTrans x;
   for(int i=0;i<3;i++)
      for(int j=0;j<3;j++)
      {
         x(i+1, j+1) =  eig.GetEigenVectors()(i,j);
      }

   TVector3 a =  x.GetBaseVec(1);
   TVector3 c = a.Cross(x.GetBaseVec(2));
   x.SetBaseVec(3, c);

   glTranslatef(fE->RefPos()[0], fE->RefPos()[1], fE->RefPos()[2]);
   glMultMatrixd(x.Array());
   glScalef(fE->RefExtent3D()[0] , fE->RefExtent3D()[1], fE->RefExtent3D()[2]);
   gluSphere(quad,1. , 30, 30);

   glPopMatrix();
   glPopAttrib();


   // gluDeleteQuadric(quad);
}


//==============================================================================
// TEveEllipsoidProjectedGL
//==============================================================================

//______________________________________________________________________________
// OpenGL renderer class for TEveEllipsoidProjected.
//



//______________________________________________________________________________
TEveEllipsoidProjectedGL::TEveEllipsoidProjectedGL() :
   fM(nullptr)
{
   // Constructor.

   // fDLCache = kFALSE; // Disable display list.
   fMultiColor = kTRUE;
}

//______________________________________________________________________________
Bool_t TEveEllipsoidProjectedGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   fM = SetModelDynCast<TEveEllipsoidProjected>(obj);
   fE = dynamic_cast<TEveEllipsoid*>(fM->GetProjectable());
   return fE != nullptr;
}

//______________________________________________________________________________
void TEveEllipsoidProjectedGL::SetBBox()
{
   // Set bounding box.

   SetAxisAlignedBBox(((TEveEllipsoidProjected*)fExternalObj)->AssertBBox());
}

//______________________________________________________________________________
void TEveEllipsoidProjectedGL::DirectDraw(TGLRnrCtx& rnrCtx) const
{
   // Render with OpenGL.
 
   TEveProjection *proj = fM->GetManager()->GetProjection();
   
   
   glPushAttrib(GL_ENABLE_BIT| GL_POLYGON_BIT | GL_LINE_BIT | GL_POINT_BIT);
   glDisable(GL_LIGHTING);
   glDisable(GL_CULL_FACE);

   glPushMatrix();
   if ( proj->GetType() == TEveProjection::kPT_RPhi)
      DrawRhoPhi();
   else
      DrawRhoZ();

   glPopMatrix();
   glPopAttrib();  
}

//______________________________________________________________________________
void TEveEllipsoidProjectedGL::drawArch(float phiStart, float phiEnd, float phiStep, TEveVector& v0,  TEveVector& v1, TEveVector& v2) const
{
   TEveProjection *proj = fM->GetManager()->GetProjection();
   float phi = phiStart;
   while (phi < phiEnd ) {
      TEveVector v = v0 + v1*((float)cos(phi)) + v2*((float)sin(phi));
      proj->ProjectVector(v, fM->fDepth);
      glVertex3fv(v.Arr());
      
      phi += phiStep;
   }
   TEveVector v = v0 + v1*((float)cos(phiEnd)) + v2*((float)sin(phiEnd));
   proj->ProjectVector(v, fM->fDepth);
   glVertex3fv(v.Arr());   
}

 //-------------------------------------------------------------------------------
void TEveEllipsoidProjectedGL::DrawRhoPhi() const
{
   // printf("TEveEllipsoidProjectedGL::DirectDraw [%s ]\n", fE->GetName() );

   TMatrixDSym xxx(3);
   for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
      {
         xxx(i,j) = fE->RefEMtx()(i+1,j+1);
      }
   
   TMatrixDEigen eig(xxx);
   TVectorD xxxEig ( eig.GetEigenValuesRe());

 
   // Projection supports only floats  :(
   TEveVector v0(fE->RefPos()[0], fE->RefPos()[1], 0);
   TEveVector v1(eig.GetEigenVectors()(0, 0) , eig.GetEigenVectors()(0, 1), 0 );
   v1 *= fE->fEScale *  sqrt(TMath::Abs(xxxEig[0]));
   TEveVector v2(eig.GetEigenVectors()(1, 0) , eig.GetEigenVectors()(1, 1), 0 );
   v2 *= fE->fEScale *  sqrt(TMath::Abs(xxxEig[1]));
   
   TEveProjection *proj = fM->GetManager()->GetProjection();
      
  // fill
   glBegin(GL_POLYGON);   
   drawArch(0, TMath::TwoPi(),  TMath::TwoPi()/20, v0, v1, v2);
   glEnd();
   
   // frame
   TGLUtil::LineWidth(fE->fLineWidth);
   TGLUtil::Color(fE->fLineColor);
   
   glBegin(GL_LINE_LOOP);   
   drawArch(0, TMath::TwoPi(),  TMath::TwoPi()/20, v0, v1, v2);
   glEnd();
   
   glBegin(GL_LINES);
   {
      // glColor3f(1, 0, 0);
      TEveVector p1 = v0 - v1;
      TEveVector p2 = v0 + v1;
      proj->ProjectVector(p1, fM->fDepth);
      proj->ProjectVector(p2, fM->fDepth);
      glVertex3fv(p1.Arr());
      glVertex3fv(p2.Arr());
   }
   {
      //  glColor3f(0, 1, 0);
      TEveVector p1 = v0 - v2;
      TEveVector p2 = v0 + v2;
      proj->ProjectVector(p1, fM->fDepth);
      proj->ProjectVector(p2, fM->fDepth);
      glVertex3fv(p1.Arr());
      glVertex3fv(p2.Arr());
   }
   glEnd();
   
   
   
}

//--------------------------------------------------------------------
void TEveEllipsoidProjectedGL::DrawRhoZ() const
{
   // printf("TEveEllipsoidProjectedGL::DirectDraw [%s ]\n", fE->GetTitle() );
 
   TEveVector v0(fE->RefPos()[0], fE->RefPos()[1], fE->RefPos()[2]);
   
   TMatrixDSym xxx(3);
   xxx(0, 0) = 1;
   for(int i=1;i<3;i++)
      for(int j=1;j<3;j++)
      {
         xxx(i,j) = fE->RefEMtx()(i+1,j+1);
      }   
   TMatrixDEigen eig(xxx);
   TVectorD xxxEig ( eig.GetEigenValuesRe());


   TEveVector v1(0, eig.GetEigenVectors()(1, 2), eig.GetEigenVectors()(2, 2));
   v1 *= fE->fEScale * sqrt(TMath::Abs(xxxEig[2]));

   TEveVector v2(0, eig.GetEigenVectors()(1, 1), eig.GetEigenVectors()(2, 1));
   v2 *= fE->fEScale * sqrt(TMath::Abs(xxxEig[1]));
   if (v1[1]*v2[2] > v1[2]*v2[1])
      v2 *= -1;

   
   TEveProjection *proj = fM->GetManager()->GetProjection();
   
   // ellipse intersection with projection center
   bool splitted = false;
   int N = 20;
   double phiStep = TMath::TwoPi()/N;
   
   // projection center can be moved in beam-spot 
   float bs = 0;
   if (proj->GetDisplaceOrigin())
      bs = proj->RefCenter()[1];

   float da = v2[1]*v2[1] + v1[1]*v1[1];
   float db = 2 * v1[1] * (v0[1]-bs);
   float dc = (v0[1]-bs)*(v0[1]-bs) - v2[1]*v2[1];
   float disc = (db*db -4*da*dc);
   
   if ( disc > 0) {
      disc = sqrt(disc);
      float cosS1 = ( -db + disc)/(2 * da);
      float cosS2 = ( -db - disc)/(2 * da);
      if (TMath::Abs(cosS1) < 1) {
         splitted = true;
         // printf("splitted \n");

         double phi1 = acos(cosS1);
         double phi2 = acos(cosS2);
         TEveVector ps1 = v0 + v1*((float)cos(phi1)) + v2*((float)sin(phi1));
         TEveVector ps2 = v0 + v1*((float)cos(phi2)) + v2*((float)sin(phi2));
         
        

         // acos has values [0, Pi] , check the symetry over x axis (mirroring)
         if (TMath::Abs(ps1[1] - bs) > 1e-5)
            phi1 = TMath::TwoPi() -phi1;
      
         if (TMath::Abs(ps2[1] - bs) > 1e-5) 
            phi2 = TMath::TwoPi() -phi2;         
         
         int N = 20;
         double phiStep = TMath::TwoPi()/N;
         double phiOffset = phiStep*0.1;
         double phiMin = TMath::Min(phi1, phi2);
         double phiMax = TMath::Max(phi1, phi2);         
         // printf(" %f %f \n",phi1*TMath::RadToDeg(), phi2*TMath::RadToDeg() );

         // fill
         // upper clothing
         glBegin(GL_POLYGON);
         drawArch(phiMin + phiOffset, phiMax - phiOffset, phiStep, v0, v1, v2);
         glEnd();
         // bottom clothing
         glBegin(GL_POLYGON);
         drawArch(phiMax + phiOffset, phiMax + TMath::TwoPi() - (phiMax -phiMin) - phiOffset, phiStep, v0, v1, v2);
         glEnd();
         
         // frame
         TGLUtil::LineWidth(fE->fLineWidth);
         TGLUtil::Color(fE->fLineColor);
         // upper clothing
         glBegin(GL_LINE_LOOP);
         drawArch(phiMin + phiOffset, phiMax - phiOffset, phiStep, v0, v1, v2);
         glEnd();
         // bottom clothing
         glBegin(GL_LINE_LOOP);
         drawArch(phiMax + phiOffset, phiMax + TMath::TwoPi() - (phiMax -phiMin) - phiOffset, phiStep, v0, v1, v2);
         glEnd();
         
      }
   }
   
   
   if (!splitted) {
       glBegin(GL_POLYGON);
      drawArch(0, TMath::TwoPi(), phiStep, v0, v1, v2);
      glEnd();
      TGLUtil::LineWidth(fE->fLineWidth);
      TGLUtil::Color(fE->fLineColor);
      glBegin(GL_LINE_LOOP);
      drawArch(0, TMath::TwoPi(), phiStep, v0, v1, v2);
      glEnd();  
   }
   
   drawRhoZAxis(v0, v2);
   drawRhoZAxis(v0, v1);
}

//______________________________________________________________________________
void TEveEllipsoidProjectedGL::drawRhoZAxis(TEveVector& v0, TEveVector& v2) const
{
   glBegin(GL_LINES);
   TEveProjection* proj =   fM->GetManager()->GetProjection();
   
   float bs = 0;
   if (proj->GetDisplaceOrigin())
      bs = proj->RefCenter()[1];
   
   float off = (v2[1] > v0[1] ) ? 0.01 : -0.01;
   TEveVector alu = v0 + v2;
   proj->ProjectVector(alu, fM->fDepth);
   glVertex3fv(alu.Arr());   
   
   if (TMath::Abs(v0[1]/v2[1]) < 1 )
   {
      alu = v0 - ((float) ((1-off) *(v0[1]-bs)/v2[1])) * v2;
      proj->ProjectVector(alu, fM->fDepth);
      glVertex3fv(alu.Arr());   
      
      //============================
      
      alu = v0 - ((float) ((1+off) * (v0[1]-bs)/v2[1])) * v2;
      proj->ProjectVector(alu, fM->fDepth);
      glVertex3fv(alu.Arr());  
   }
   
   alu = v0 - v2;
   proj->ProjectVector(alu, fM->fDepth);
   glVertex3fv(alu.Arr());   

   glEnd();
}
