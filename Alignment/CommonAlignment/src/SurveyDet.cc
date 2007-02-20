#include "Alignment/CommonAlignment/interface/SurveyDet.h"

SurveyDet::SurveyDet(const AlignableSurface& surface):
  theSurface(surface)
{
  const float W3 = surface.width()  / 3.f;
  const float L3 = surface.length() / 3.f;

  thePoints.reserve(9); // 9 survey points

  thePoints.push_back( LocalPoint( 0.,  0., 0.) );
  thePoints.push_back( LocalPoint( W3,  0., 0.) );
  thePoints.push_back( LocalPoint( W3,  L3, 0.) );
  thePoints.push_back( LocalPoint( 0.,  L3, 0.) );
  thePoints.push_back( LocalPoint(-W3,  L3, 0.) );
  thePoints.push_back( LocalPoint(-W3,  0., 0.) );
  thePoints.push_back( LocalPoint(-W3, -L3, 0.) );
  thePoints.push_back( LocalPoint( 0., -L3, 0.) );
  thePoints.push_back( LocalPoint( W3, -L3, 0.) );
}

AlgebraicMatrix SurveyDet::derivatives(unsigned int index) const
{
  AlgebraicMatrix jac(6, 3, 0); // 6 by 3 Jacobian init to 0

//   jac(1, 1) = S11; jac(1, 2) = S12; jac(1, 3) = S13;
//   jac(2, 1) = S21; jac(2, 2) = S22; jac(2, 3) = S23;
//   jac(3, 1) = S31; jac(3, 2) = S32; jac(3, 3) = S33;

//   jac(4, 1) = u2 * S31;
//   jac(4, 2) = u2 * S32;
//   jac(4, 3) = u2 * S33;

//   jac(5, 1) = -u1 * S31;
//   jac(5, 2) = -u1 * S32;
//   jac(5, 3) = -u1 * S33;

//   jac(6, 1) = u1 * S21 - u2 * S11;
//   jac(6, 2) = u1 * S22 - u2 * S12;
//   jac(6, 3) = u1 * S23 - u2 * S13;

  jac(1, 1) = 1.; jac(2, 2) = 1.; jac(3, 3) = 1.;

  jac(5, 3) -= jac(6, 2) = thePoints[index].x();
  jac(6, 1) -= jac(4, 3) = thePoints[index].y();

  return jac;
}
