/** \file FrameToFrameDerivative.cc
 *
 *  $Date: 2010/12/14 01:02:34 $
 *  $Revision: 1.6 $
 */

#include "Alignment/CommonAlignmentParametrization/interface/FrameToFrameDerivative.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"

// already in header: #include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "DataFormats/CLHEP/interface/Migration.h"

//__________________________________________________________________________________________________
AlgebraicMatrix 
FrameToFrameDerivative::frameToFrameDerivative(const Alignable* object,
					       const Alignable* composedObject) const
{

  return getDerivative( object->globalRotation(),
			composedObject->globalRotation(),
			composedObject->globalPosition() - object->globalPosition() );

}

//__________________________________________________________________________________________________
AlgebraicMatrix66 
FrameToFrameDerivative::getDerivative(const align::RotationType &objectRot,
                                      const align::RotationType &composeRot,
				      const align::GlobalPoint &objectPos,
				      const align::GlobalPoint &composePos) const
{
  return asSMatrix<6,6>(this->getDerivative(objectRot, composeRot, composePos - objectPos));
}

//__________________________________________________________________________________________________
AlgebraicMatrix 
FrameToFrameDerivative::getDerivative(const align::RotationType &objectRot,
				      const align::RotationType &composeRot,
				      const align::GlobalVector &posVec) const
{

  AlgebraicMatrix rotDet   = transform(objectRot);
  AlgebraicMatrix rotCompO = transform(composeRot);

  AlgebraicVector diffVec(3);

  diffVec(1) = posVec.x();
  diffVec(2) = posVec.y();
  diffVec(3) = posVec.z();

  AlgebraicMatrix derivative(6,6);

  AlgebraicMatrix derivAA(3,3);
  AlgebraicMatrix derivAB(3,3);
  AlgebraicMatrix derivBB(3,3);
  
  derivAA = derivativePosPos( rotDet, rotCompO );
  derivAB = derivativePosRot( rotDet, rotCompO, diffVec );
  derivBB = derivativeRotRot( rotDet, rotCompO );

  derivative[0][0] = derivAA[0][0];
  derivative[0][1] = derivAA[0][1];
  derivative[0][2] = derivAA[0][2];
  derivative[0][3] = derivAB[0][0];                                
  derivative[0][4] = derivAB[0][1];                               
  derivative[0][5] = derivAB[0][2];                             
  derivative[1][0] = derivAA[1][0];
  derivative[1][1] = derivAA[1][1];
  derivative[1][2] = derivAA[1][2];
  derivative[1][3] = derivAB[1][0];                                 
  derivative[1][4] = derivAB[1][1];                             
  derivative[1][5] = derivAB[1][2];                           
  derivative[2][0] = derivAA[2][0];
  derivative[2][1] = derivAA[2][1];
  derivative[2][2] = derivAA[2][2];
  derivative[2][3] = derivAB[2][0];            
  derivative[2][4] = derivAB[2][1];
  derivative[2][5] = derivAB[2][2];
  derivative[3][0] = 0;
  derivative[3][1] = 0;
  derivative[3][2] = 0;
  derivative[3][3] = derivBB[0][0];
  derivative[3][4] = derivBB[0][1];
  derivative[3][5] = derivBB[0][2];
  derivative[4][0] = 0;
  derivative[4][1] = 0;
  derivative[4][2] = 0;
  derivative[4][3] = derivBB[1][0];
  derivative[4][4] = derivBB[1][1];
  derivative[4][5] = derivBB[1][2];
  derivative[5][0] = 0;
  derivative[5][1] = 0;
  derivative[5][2] = 0;
  derivative[5][3] = derivBB[2][0];
  derivative[5][4] = derivBB[2][1];
  derivative[5][5] = derivBB[2][2];
  
  return(derivative.T());

}


//__________________________________________________________________________________________________
AlgebraicMatrix 
FrameToFrameDerivative::derivativePosPos(const AlgebraicMatrix &RotDet,
					 const AlgebraicMatrix &RotRot) const
{

  return RotDet * RotRot.T();

}


//__________________________________________________________________________________________________
AlgebraicMatrix 
FrameToFrameDerivative::derivativePosRot(const AlgebraicMatrix &RotDet,
					 const AlgebraicMatrix &RotRot,
					 const AlgebraicVector &S) const
{

 AlgebraicVector dEulerA(3);
 AlgebraicVector dEulerB(3);
 AlgebraicVector dEulerC(3);
 AlgebraicMatrix RotDa(3,3);
 AlgebraicMatrix RotDb(3,3);
 AlgebraicMatrix RotDc(3,3);
 
 RotDa[1][2] =  1; RotDa[2][1] = -1;
 RotDb[0][2] = -1; RotDb[2][0] =  1; // New beta sign
 RotDc[0][1] =  1; RotDc[1][0] = -1;
 
 dEulerA = RotDet*( RotRot.T()*RotDa*RotRot*S );
 dEulerB = RotDet*( RotRot.T()*RotDb*RotRot*S );
 dEulerC = RotDet*( RotRot.T()*RotDc*RotRot*S );

 AlgebraicMatrix eulerDeriv(3,3);
 eulerDeriv[0][0] = dEulerA[0];
 eulerDeriv[1][0] = dEulerA[1];
 eulerDeriv[2][0] = dEulerA[2];
 eulerDeriv[0][1] = dEulerB[0];
 eulerDeriv[1][1] = dEulerB[1];
 eulerDeriv[2][1] = dEulerB[2];
 eulerDeriv[0][2] = dEulerC[0];
 eulerDeriv[1][2] = dEulerC[1];
 eulerDeriv[2][2] = dEulerC[2];

 return eulerDeriv;

}


//__________________________________________________________________________________________________
AlgebraicMatrix 
FrameToFrameDerivative::derivativeRotRot(const AlgebraicMatrix &RotDet,
					 const AlgebraicMatrix &RotRot) const
{

 AlgebraicVector dEulerA(3);
 AlgebraicVector dEulerB(3);
 AlgebraicVector dEulerC(3);
 AlgebraicMatrix RotDa(3,3);
 AlgebraicMatrix RotDb(3,3);
 AlgebraicMatrix RotDc(3,3);

 RotDa[1][2] =  1; RotDa[2][1] = -1;
 RotDb[0][2] = -1; RotDb[2][0] =  1; // New beta sign
 RotDc[0][1] =  1; RotDc[1][0] = -1;

 dEulerA = linearEulerAngles( RotDet*RotRot.T()*RotDa*RotRot*RotDet.T() );
 dEulerB = linearEulerAngles( RotDet*RotRot.T()*RotDb*RotRot*RotDet.T() );
 dEulerC = linearEulerAngles( RotDet*RotRot.T()*RotDc*RotRot*RotDet.T() );

 AlgebraicMatrix eulerDeriv(3,3);

 eulerDeriv[0][0] = dEulerA[0];
 eulerDeriv[1][0] = dEulerA[1];
 eulerDeriv[2][0] = dEulerA[2];
 eulerDeriv[0][1] = dEulerB[0];
 eulerDeriv[1][1] = dEulerB[1];
 eulerDeriv[2][1] = dEulerB[2];
 eulerDeriv[0][2] = dEulerC[0];
 eulerDeriv[1][2] = dEulerC[1];
 eulerDeriv[2][2] = dEulerC[2];

 return eulerDeriv;

}



//__________________________________________________________________________________________________
AlgebraicVector 
FrameToFrameDerivative::linearEulerAngles(const AlgebraicMatrix &rotDelta ) const
{
  
  AlgebraicMatrix eulerAB(3,3);
  AlgebraicVector aB(3);
  eulerAB[0][1] =  1; 
  eulerAB[1][0] = -1; // New beta sign
  aB[2] = 1;

  AlgebraicMatrix eulerC(3,3);
  AlgebraicVector C(3);
  eulerC[2][0] = 1;
  C[1] = 1;

  AlgebraicVector eulerAngles(3);
  eulerAngles = eulerAB*rotDelta*aB + eulerC*rotDelta*C;
  return eulerAngles;

}

