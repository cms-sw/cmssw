#include "Alignment/SurveyAnalysis/interface/SurveyPxbImage.h"
#include "Alignment/SurveyAnalysis/interface/SurveyPxbImageLocalFit.h"

#include <stdexcept>
#include <utility>
#include <sstream>
#include <vector>
#include <cmath>
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "Math/SMatrix.h"
#include "Math/SVector.h"
#include "DataFormats/Math/interface/Matrix.h"
#include "DataFormats/Math/interface/Vector.h"

#include <iostream>

void SurveyPxbImageLocalFit::doFit()
{
	fitValidFlag_ = false;

	// Creating vectors with the global parameters of the two modules
	math::VectorD<4>::type mod1, mod2;
	mod1(0)=u1_;
	mod1(1)=v1_;
	mod1(2)=cos(g1_);
	mod1(3)=sin(g1_);
	mod2(0)=u2_;
	mod2(1)=v2_;
	mod2(2)=cos(g2_);
	mod2(3)=sin(g2_);
	//std::cout << "mod1: " << mod1 << std::endl;
	//std::cout << "mod2: " << mod2 << std::endl;

	// Create a matrix for the transformed position of the fidpoints
	math::Matrix<4,4>::type M1, M2;
	M1(0,0)=1.; M1(0,1)=0.; M1(0,2)=+fidpoints_[0].x(); M1(0,3)=-fidpoints_[0].y();
	M1(1,0)=0.; M1(1,1)=1.; M1(1,2)=+fidpoints_[0].y(); M1(1,3)=+fidpoints_[0].x();
	M1(2,0)=1.; M1(2,1)=0.; M1(2,2)=+fidpoints_[1].x(); M1(2,3)=-fidpoints_[1].y();
	M1(3,0)=0.; M1(3,1)=1.; M1(3,2)=+fidpoints_[1].y(); M1(3,3)=+fidpoints_[1].x();
	M2(0,0)=1.; M2(0,1)=0.; M2(0,2)=+fidpoints_[2].x(); M2(0,3)=-fidpoints_[2].y();
	M2(1,0)=0.; M2(1,1)=1.; M2(1,2)=+fidpoints_[2].y(); M2(1,3)=+fidpoints_[2].x();
	M2(2,0)=1.; M2(2,1)=0.; M2(2,2)=+fidpoints_[3].x(); M2(2,3)=-fidpoints_[3].y();
	M2(3,0)=0.; M2(3,1)=1.; M2(3,2)=+fidpoints_[3].y(); M2(3,3)=+fidpoints_[3].x();

	//std::cout << "M1:\n" << M1 << std::endl;
	//std::cout << "M2:\n" << M2 << std::endl;

	math::VectorD<4>::type mod_tr1, mod_tr2;
	mod_tr1 = M1*mod2;
	mod_tr2 = M2*mod1;
	//std::cout << "mod_tr1: " << mod_tr1 << std::endl;
	//std::cout << "mod_tr2: " << mod_tr2 << std::endl;

	math::Matrix<8,4>::type A;
	A(0,0)=1.; A(0,1)=0; A(0,2)=+mod_tr1(0); A(0,3)=+mod_tr1(1);
	A(1,0)=0.; A(1,1)=1; A(1,2)=+mod_tr1(1); A(1,3)=-mod_tr1(0);
	A(2,0)=1.; A(2,1)=0; A(2,2)=+mod_tr1(2); A(2,3)=+mod_tr1(3);
	A(3,0)=0.; A(3,1)=1; A(3,2)=+mod_tr1(3); A(3,3)=-mod_tr1(2);
	A(4,0)=1.; A(4,1)=0; A(4,2)=+mod_tr2(0); A(4,3)=+mod_tr2(1);
	A(5,0)=0.; A(5,1)=1; A(5,2)=+mod_tr2(1); A(5,3)=-mod_tr2(0);
	A(6,0)=1.; A(6,1)=0; A(6,2)=+mod_tr2(2); A(6,3)=+mod_tr2(3);
	A(7,0)=0.; A(7,1)=1; A(7,2)=+mod_tr2(3); A(7,3)=-mod_tr2(2);
	//std::cout << "A: \n" << A << std::endl;

	// Covariance matrix
	math::Matrix<8,8>::type W;
	const value_t sigma_u2inv = 1./(sigma_u_*sigma_u_);
	const value_t sigma_v2inv = 1./(sigma_v_*sigma_v_);
	W(0,0) = sigma_u2inv;
	W(1,1) = sigma_v2inv;
	W(2,2) = sigma_u2inv;
	W(3,3) = sigma_v2inv;
	W(4,4) = sigma_u2inv;
	W(5,5) = sigma_v2inv;
	W(6,6) = sigma_u2inv;
	W(7,7) = sigma_v2inv;
	//std::cout << "W: \n" << W << std::endl;

	// Prepare for the fit
	math::Matrix<4,4>::type ATWA;
	ATWA = ROOT::Math::Transpose(A) * W * A;
	//std::cout << "ATWA: \n" << ATWA << std::endl;
	math::Matrix<4,4>::type ATWAi;
	int ifail = 0;
	ATWAi = ATWA.Inverse(ifail); // TODO: ifail pruefen
	//std::cout << "ATWA-1: \n" << ATWAi << ifail << std::endl;

	// Measurements
	math::VectorD<8>::type y;
	y(0) = measurementVec_[0].x();
	y(1) = measurementVec_[0].y();
	y(2) = measurementVec_[1].x();
	y(3) = measurementVec_[1].y();
	y(4) = measurementVec_[2].x();
	y(5) = measurementVec_[2].y();
	y(6) = measurementVec_[3].x();
	y(7) = measurementVec_[3].y();
	//std::cout << "y: " << y << std::endl;

	// do the fit
	math::VectorD<4>::type a;	
	a = ATWAi * ROOT::Math::Transpose(A) * W * y;
	const value_t chi2 = ROOT::Math::Dot(y,W*y)-ROOT::Math::Dot(a,ROOT::Math::Transpose(A)*W*y);
	std::cout << "a: " << a 
		<< " S= " << sqrt(a[2]*a[2]+a[3]*a[3]) 
		<< " phi= " << atan(a[3]/a[2]) 
		<< " chi2= " << chi2 << std::endl;
	//std::cout << "A*a: " << A*a << std::endl;

}


void SurveyPxbImageLocalFit::doFit(value_t u1, value_t v1, value_t g1, value_t u2, value_t v2, value_t g2)
{
	u1_ = u1;
	v1_ = v1;
	g1_ = g1;
	u2_ = u2;
	v2_ = v2;
	g2_ = g2;
	doFit();
}


