/** \file AlignmentTransformations.cc
 *
 *  $Date: 2006/11/30 09:48:47 $
 *  $Revision: 1.3 $
 *  $Author$ did last update.
 */

#include <stdlib.h>
#include <math.h>

#include "CLHEP/Vector/Rotation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// This class's header

#include "Alignment/CommonAlignmentParametrization/interface/AlignmentTransformations.h"


//__________________________________________________________________________________________________
Surface::RotationType 
AlignmentTransformations::globalToLocalMatrix(const Surface::RotationType &aliDetRot,
                                              const Surface::RotationType &detRot) const
{
  
  AlgebraicMatrix aliDet(3,3);	
  AlgebraicMatrix Det(3,3);
  aliDet = this->algebraicMatrix(aliDetRot);
  Det = this->algebraicMatrix(detRot);
  AlgebraicMatrix localMatrix(3,3);
  localMatrix = Det*aliDet*Det.T();
  
  return this->rotationType(localMatrix);
}


//__________________________________________________________________________________________________
Surface::RotationType  
AlignmentTransformations::localToGlobalMatrix(const Surface::RotationType &aliDetRot,
                                              const Surface::RotationType &detRot) const
{
  
  AlgebraicMatrix aliDet(3,3);	
  AlgebraicMatrix Det(3,3);
  aliDet = this->algebraicMatrix(aliDetRot);
  Det = this->algebraicMatrix(detRot);
  AlgebraicMatrix globalMatrix(3,3);
  globalMatrix = Det.T()*aliDet*Det;
  
  return(this->rotationType(globalMatrix));
}


//__________________________________________________________________________________________________
AlgebraicMatrix AlignmentTransformations::algebraicMatrix(const Surface::RotationType &rot) const
{
  
  AlgebraicMatrix testMatrix(3,3);
  testMatrix[0][0]=rot.xx();
  testMatrix[0][1]=rot.xy();
  testMatrix[0][2]=rot.xz();
  testMatrix[1][0]=rot.yx();
  testMatrix[1][1]=rot.yy();
  testMatrix[1][2]=rot.yz();
  testMatrix[2][0]=rot.zx();
  testMatrix[2][1]=rot.zy();
  testMatrix[2][2]=rot.zz();

  return testMatrix;
}

//__________________________________________________________________________________________________
AlgebraicVector AlignmentTransformations::algebraicVector(const Surface::RotationType &rot) const
{

  AlgebraicVector vec(9);
  vec[0] = rot.xx(); vec[1] = rot.xy(); vec[2] = rot.xz();
  vec[3] = rot.yx(); vec[4] = rot.yy(); vec[5] = rot.yz();
  vec[6] = rot.zx(); vec[7] = rot.zy(); vec[8] = rot.zz();
  
  return vec;
}


//__________________________________________________________________________________________________
AlgebraicVector AlignmentTransformations::algebraicVector(const GlobalVector &globalVector) const
{

  AlgebraicVector algV(3);
  algV[0] = globalVector.x();
  algV[1] = globalVector.y();
  algV[2] = globalVector.z();

  return algV;
}


//__________________________________________________________________________________________________
Surface::RotationType AlignmentTransformations::rotationType(const AlgebraicMatrix &algM) const
{

  double mat[9];
  mat[0] = algM[0][0];
  mat[1] = algM[0][1];
  mat[2] = algM[0][2];
  mat[3] = algM[1][0];
  mat[4] = algM[1][1];
  mat[5] = algM[1][2];
  mat[6] = algM[2][0];
  mat[7] = algM[2][1];
  mat[8] = algM[2][2];

  Surface::RotationType rot(mat[0],mat[1],mat[2],
                            mat[3],mat[4],mat[5],
                            mat[6],mat[7],mat[8]);
  return rot;
}


//__________________________________________________________________________________________________
AlgebraicMatrix AlignmentTransformations::rotMatrix3(const AlgebraicVector &a) const
{

  // New convention, consistent with misalignment scenario
  AlgebraicMatrix orig(3,3);
  orig[0][0] = cos(a[1])*cos(a[2]);
  orig[0][1] = sin(a[0])*sin(a[1])*cos(a[2])+cos(a[0])*sin(a[2]);
  orig[0][2] =-cos(a[0])*sin(a[1])*cos(a[2])+sin(a[0])*sin(a[2]);
  orig[1][0] =-cos(a[1])*sin(a[2]);
  orig[1][1] =-sin(a[0])*sin(a[1])*sin(a[2])+cos(a[0])*cos(a[2]);
  orig[1][2] = cos(a[0])*sin(a[1])*sin(a[2])+sin(a[0])*cos(a[2]);
  orig[2][0] = sin(a[1]);
  orig[2][1] =-sin(a[0])*cos(a[1]);
  orig[2][2] = cos(a[0])*cos(a[1]);  

  return orig;
}


//__________________________________________________________________________________________________
AlgebraicVector AlignmentTransformations::eulerAngles(const Surface::RotationType &rot,
                                                      int flag) const
{
  AlgebraicMatrix orig = this->algebraicMatrix(rot);
  AlgebraicMatrix testangle(3,2);
  AlgebraicVector returnangle(3);
 
  if(orig[2][0]!=1.0) // If angle1 is not +-PI/2
    {
      if (fabs(orig[2][0] - 1.0) < 1.e-4) {
        edm::LogWarning("Alignment") << "@SUB=AlignmentTransformations::eulerAngles"
                                     << "Trusting a floating point to be != 1. ! "
                                     << orig[2][0] << " -1. = " << orig[2][0] - 1.0;
      }

      if(flag==0) // assuming -PI/2 < angle1 < PI/2 
        testangle[1][flag] = asin(orig[2][0]); // New beta sign convention
      
      if(flag==1) // assuming angle1 < -PI/2 or angle1 >PI/2
        testangle[1][flag]=M_PI-asin(orig[2][0]); // New beta sign convention
      
      if(cos(testangle[1][flag])*orig[2][2]>0)
        testangle[0][flag]=atan(-orig[2][1]/orig[2][2]);
      else
        testangle[0][flag]=atan(-orig[2][1]/orig[2][2])+M_PI;
      
      if(cos(testangle[1][flag])*orig[0][0]>0)
        testangle[2][flag]=atan(-orig[1][0]/orig[0][0]);
      else
        testangle[2][flag]=atan(-orig[1][0]/orig[0][0])+M_PI;
    }
  
  else // if angle1 == +-PI/2
    {

      testangle[1][flag]=M_PI/2; // chose positve Solution 
      if(orig[2][2]>0)
        {
          testangle[2][flag]=atan(orig[1][2]/orig[1][1]);
          testangle[0][flag]=0;
        }
    }
  
  for(int i=0;i<3;i++)
    {
      returnangle[i]=testangle[i][flag];
    }
  
  return returnangle;
}


//__________________________________________________________________________________________________
AlgebraicVector 
AlignmentTransformations::globalToLocalEulerAngles(const AlgebraicVector &a, 
                                                   const Surface::RotationType &rot) const
{
  
  Surface::RotationType gloRot = 
    this->globalToLocalMatrix(this->rotationType(this->rotMatrix3(a)), rot);
  AlgebraicVector result = this->eulerAngles(gloRot,0);

  return result;
}


//__________________________________________________________________________________________________
AlgebraicVector 
AlignmentTransformations::localToGlobalEulerAngles(const AlgebraicVector &a, 
                                                   const Surface::RotationType &rot) const
{
  
  Surface::RotationType locRot = 
    this->globalToLocalMatrix(this->rotationType(this->rotMatrix3(a)),rot);
  AlgebraicVector result = this->eulerAngles(locRot,0);
  
  return result;
}


//__________________________________________________________________________________________________
// repair rotation matrix for rounding errors 
// (a la HepRotation::rectify)
Surface::RotationType AlignmentTransformations::rectify(const Surface::RotationType &r) const
{

  HepRotation h(HepRep3x3(r.xx(),r.xy(),r.xz(),
                          r.yx(),r.yy(),r.yz(),
                          r.zx(),r.zy(),r.zz()));
  h.rectify();
  
  Surface::RotationType p(h.xx(),h.xy(),h.xz(),
                          h.yx(),h.yy(),h.yz(),
                          h.zx(),h.zy(),h.zz());
  
  return p;
}
