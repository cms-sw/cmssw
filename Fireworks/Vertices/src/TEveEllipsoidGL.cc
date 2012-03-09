#include "Fireworks/Vertices/interface/TEveEllipsoidGL.h"

#define protected public  
#include "TEveProjections.h" // AMT missing getter for projection center / beam-spot
#undef protected

#include "TEveEllipsoidGL.h"
#include "TEveEllipsoid.h"
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
   TGLObject(), fE(0)
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
GLUquadric* quad = 0; // !!!! AMT check why TGLQuadric crashes on mac
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
   fM(0)
{
   // Constructor.

   // fDLCache = kFALSE; // Disable display list.
}

//______________________________________________________________________________
Bool_t TEveEllipsoidProjectedGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   fM = SetModelDynCast<TEveEllipsoidProjected>(obj);
   fE = dynamic_cast<TEveEllipsoid*>(fM->GetProjectable());
   return fE != 0;
}

//______________________________________________________________________________
void TEveEllipsoidProjectedGL::SetBBox()
{
   // Set bounding box.

   SetAxisAlignedBBox(((TEveEllipsoidProjected*)fExternalObj)->AssertBBox());
}



//______________________________________________________________________________
void TEveEllipsoidProjectedGL::DirectDraw(TGLRnrCtx& /*rnrCtx*/) const
{
   // Render with OpenGL.
 
   TEveProjection *proj = fM->GetManager()->GetProjection();
   
   glPushAttrib(GL_ENABLE_BIT| GL_LINE_BIT | GL_POINT_BIT);
   glDisable(GL_LIGHTING);
   glLineWidth(2);
   glPushMatrix();
   if ( proj->GetType() == TEveProjection::kPT_RPhi)
      DrawRhoPhi();
   else
      DrawRhoZ();

   glPopMatrix();
   glPopAttrib();  
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
   
   // axis

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
   
   
   // ellipse
   int N = 20;
   double phiStep = TMath::TwoPi()/N;
   glBegin(GL_LINE_LOOP);
   for (int i=0; i < N; ++i)
   {
      TEveVector p = v0 + v1 * ((float)cos(i*phiStep)) + v2 * ((float)sin(i*phiStep));
      proj->ProjectVector(p, fM->fDepth);
      glVertex3fv(p.Arr());
   }
   glEnd();
   
}
//--------------------------------------------------------------------
void TEveEllipsoidProjectedGL::DrawRhoZ() const
{
   //  printf("TEveEllipsoidProjectedGL::DirectDraw [%s ]\n", fE->GetName() );
 
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
   
   /*   
   // axis 
   {
   glBegin(GL_LINES);

   TEveVector p0 = v0;
   TEveVector p1 = v1 + v0;
   TEveVector p2 = v2 + v0;

   proj->ProjectVector(p0, fM->fDepth); 
   proj->ProjectVector(p1, fM->fDepth); 
   proj->ProjectVector(p2, fM->fDepth);
      
      
   glColor3f(1, 0, 0);
   glVertex3fv(p0.Arr());
   glVertex3fv(p1.Arr());
      
   glColor3f(0, 1, 0);
   glVertex3fv(p0.Arr());
   glVertex3fv(p2.Arr());
   glEnd();
   }
   */

   // ellipse intersection with projection center
   bool splitted = false;

   // projection center can be moved in beam-spot 
   float bs = 0;
   if (proj->GetDisplaceOrigin())
      bs = proj->fCenter[1];

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
         
         //  marker before correction
         /*
           {
           glColor3f(1, 1, 0);
           TEveVector pps1 = ps1; 
           proj->ProjectVector(pps1, fM->fDepth);
           glVertex3fv(pps1.Arr());
            
           TEveVector pps2 = ps2; 
           proj->ProjectVector(pps2, fM->fDepth);
           glVertex3fv(pps2.Arr());
           }
         */

         // acos has values [0, Pi] , check the symetry over x axis (mirroring)
         if (TMath::Abs(ps1[1] - bs) > 1e-5)
            phi1 = TMath::TwoPi() -phi1;
      
         if (TMath::Abs(ps2[1] - bs) > 1e-5) 
            phi2 = TMath::TwoPi() -phi2;


         //  marker after correction
         /*
           glPointSize(5);
           glBegin(GL_POINTS);
     
           ps1 = v0 + v1*((float)cos(phi1)) + v2*((float)sin(phi1));
           ps2 = v0 + v1*((float)cos(phi2)) + v2*((float)sin(phi2));
         
           glColor3f(0, 1, 1);
           proj->ProjectVector(ps1, fM->fDepth);
           glVertex3fv(ps1.Arr());

           glColor3f(1, 0, 1);
           proj->ProjectVector(ps2, fM->fDepth);
           glVertex3fv(ps2.Arr());
           glEnd();
         */
         
         
         int N = 20;
         double phiStep = TMath::TwoPi()/N;
         double phiOffset = phiStep*0.1;
         double phiMin = TMath::Min(phi1, phi2);
         double phiMax = TMath::Max(phi1, phi2);         
         // printf(" %f %f \n",phi1*TMath::RadToDeg(), phi2*TMath::RadToDeg() );

         {
            // upper clothing
            glBegin(GL_LINE_LOOP);
            double phi = phiMin + phiOffset;
            double phiEnd = phiMax - phiOffset;
            while (phi < phiEnd ) {
               TEveVector v = v0 + v1*((float)cos(phi)) + v2*((float)sin(phi));
               proj->ProjectVector(v, fM->fDepth);
               glVertex3fv(v.Arr());

               phi += phiStep;
            }
            TEveVector v = v0 + v1*((float)cos(phiEnd)) + v2*((float)sin(phiEnd));
            proj->ProjectVector(v, fM->fDepth);
            glVertex3fv(v.Arr());

            glEnd();
         }
         
         {
            // bottom clothing
            glBegin(GL_LINE_LOOP);
            double phi = phiMax + phiOffset;
            double phiEnd = phi + TMath::TwoPi() - (phiMax -phiMin) -2 * phiOffset;
            while (phi < phiEnd ) {
               TEveVector v = v0 + v1*((float)cos(phi)) + v2*((float)sin(phi));
               proj->ProjectVector(v, fM->fDepth);
               glVertex3fv(v.Arr());
               phi += phiStep;
            }
            TEveVector v = v0 + v1*(float)cos(phiEnd) + v2*(float)sin(phiEnd);
            proj->ProjectVector(v, fM->fDepth);
            glVertex3fv(v.Arr());

            glEnd();
         }
         
      }
   }
   
   if (!splitted) {
      glBegin(GL_LINE_LOOP);
      int N = 20;
      float phiStep = TMath::TwoPi()/N;
      for (int i = 0; i < N; ++i)
      {
         float phi =i*phiStep;
         TEveVector v = v0 + v1*(float)cos(phi) + v2*(float)sin(phi);
         proj->ProjectVector(v, fM->fDepth);
         glVertex3fv(v.Arr());
      }
      glEnd();  
   }
}
//______________________________________________________________________________

/*
void TEveEllipsoidProjectedGL::DrawYZ() const
{
   TMatrixDSym xxx(2);
   for(int i=1;i<3;i++)
      for(int j=1;j<3;j++)
      {
         xxx(i-1,j-1) = fE->RefEMtx()(i+1,j+1);
      }

   TMatrixDEigen eig(xxx);
   TVectorD xxxEig ( eig.GetEigenValuesRe());
   //xxxEig.Print();
   // eeeig.GetEigenVectors().Print();

   float a = fE->fEScale * sqrt(xxxEig[0]);
   float b = fE->fEScale * sqrt(xxxEig[1]);
   TEveVector2D v0(fE->RefPos()[2], fE->RefPos()[1]);
   TEveVector2D v1( eig.GetEigenVectors()(0, 1),   eig.GetEigenVectors()(0, 0));
   TEveVector2D v2(eig.GetEigenVectors()(1, 1),  eig.GetEigenVectors()(1, 0));
   v1 *= a;
   v2 *= b;
   if (v1[0]*v2[1] < v1[1]*v2[0]) {
      printf("!!!!!!! fix to right handed \n");
      v2 *= -1;
   }


   // axis
   {
      glBegin(GL_LINES);

      glColor3f(1, 0, 0);
      glVertex2dv(v0.Arr());
      glVertex2dv((v1 + v0).Arr());

      glColor3f(0, 1, 0);
      glVertex2dv(v0.Arr());
      glVertex2dv((v2 + v0).Arr());
      glEnd();
   }
   // line intersection, discriminant 
   
   bool splitted = false;
   double da = v2[1]*v2[1] + v1[1]*v1[1];
   double db = 2 * v1[1] * v0[1];
   double dc = v0[1]*v0[1] - v2[1]*v2[1];
   
   double disc = (db*db -4*da*dc);
   
   if (disc > 0) {
      disc = sqrt(disc);
      double cosS1 = ( -db + disc)/(2 * da);
      double cosS2 = ( -db - disc)/(2 * da);
      if (TMath::Abs(cosS1) < 1) {
         splitted = true;
         double phi1 = acos(cosS1);
         double phi2 = acos(cosS2);
         // printf("cos   %f %f \n acos %f %f \n", cosS1, cosS2, phi1*TMath::RadToDeg(), phi2*TMath::RadToDeg());
         printf("1 : acos %f cos %f \n", cosS1, phi1*TMath::RadToDeg());
         printf("2 : acos %f cos %f \n", cosS2, phi2*TMath::RadToDeg());
         
         TEveVector2D ps1 = v0 + v1*cos(phi1) + v2*sin(phi1);
         TEveVector2D ps2 = v0 + v1*cos(phi2) + v2*sin(phi2);
         
         glPointSize(5);
         glBegin(GL_POINTS);
         glColor3f(0, 1, 1);
         
         glColor3f(1, 1, 0);
         if (TMath::Abs(ps1[1]) > 1e-7) {
            phi1 = TMath::TwoPi() -phi1;
            printf("fix phi1 %f\n", phi1*TMath::RadToDeg());
            ps1 = v0 + v1*cos(phi1) + v2*sin(phi1);
            glVertex2dv(ps1.Arr());
         }
         if (TMath::Abs(ps2[1]) > 1e-9) 
         {
            phi2 = TMath::TwoPi() -phi2;
            printf("fix phi2 %f\n", phi2*TMath::RadToDeg());
            ps2 = v0 + v1*cos(phi2) + v2*sin(phi2);
            glVertex2dv(ps2.Arr());
         }
         
         glVertex2dv(ps1.Arr());
         glVertex2dv(ps2.Arr());
         glEnd();
         
         
         glColor3f(0.2,0.2,0.1);
         
         int N = 20;
         double phiStep = TMath::TwoPi()/N;
         double phiOffset = phiStep*0.1;
         
         double phiMin = TMath::Min(phi1, phi2);
         double phiMax = TMath::Max(phi1, phi2);
         
         // upper clothing
         {
            glBegin(GL_LINE_LOOP);
            double phi = phiMin + phiOffset;
            double phiEnd = phiMax - phiOffset;
            while (phi < phiEnd ) {
               TEveVector2D v = v0 + v1*cos(phi) + v2*sin(phi);
               glVertex2dv(v.Arr());

               phi += phiStep;
            }
            TEveVector2D v = v0 + v1*cos(phiEnd) + v2*sin(phiEnd);
            glVertex2dv(v.Arr());

            glEnd();
         }
         // bottom clothing
         if (1) {
            glColor3f(1,1,1);
            glBegin(GL_LINE_LOOP);
            double phi = phiMax + phiOffset;
            double phiEnd = phi + TMath::TwoPi() - (phiMax -phiMin) -phiOffset;
            while (phi < phiEnd ) {
               TEveVector2D v = v0 + v1*cos(phi) + v2*sin(phi);
               glVertex2dv(v.Arr());
               phi += phiStep;
            }
            TEveVector2D v = v0 + v1*cos(phiEnd) + v2*sin(phiEnd);
            glVertex2dv(v.Arr());

            glEnd();
         }
         
      }
   }
   
   if (!splitted) {
      
      glBegin(GL_LINE_LOOP);
      glColor3f(0.2,0.2,0.1);
      using namespace TMath;
      int N = 20;
      float phiStep = TwoPi()/N;
      for (int i = 0; i < N; ++i)
      {
         float phi =i*phiStep;
         TEveVector2D v = v0 + v1*cos(phi) + v2*sin(phi);
         glVertex2dv(v.Arr());
      }
      glEnd();  
   }
}
*/
