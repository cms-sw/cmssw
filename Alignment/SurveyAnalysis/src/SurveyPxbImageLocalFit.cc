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

#include <iostream>

void SurveyPxbImageLocalFit::doFit(const fidpoint_t &fidpointvec, const pede_label_t &label1, const pede_label_t &label2)
{
	labelVec1_.clear();
	labelVec1_.push_back(label1+0);
	labelVec1_.push_back(label1+1);
	labelVec1_.push_back(label1+5);
	labelVec2_.clear();
	labelVec2_.push_back(label2+0);
	labelVec2_.push_back(label2+1);
	labelVec2_.push_back(label2+5);
	doFit(fidpointvec);
}

void SurveyPxbImageLocalFit::doFit(const fidpoint_t &fidpointvec)
{
	fitValidFlag_ = false;

	// Calculate gamma of right module w.r.t left modules' fram
	const value_t dxr = fidpointvec[3].x()-fidpointvec[2].x();
	const value_t dyr = fidpointvec[3].y()-fidpointvec[2].y();
	const value_t gammar = atan(dyr/dxr);
	const value_t dxl = fidpointvec[1].x()-fidpointvec[0].x();
	const value_t dyl = fidpointvec[1].y()-fidpointvec[0].y();
	const value_t gammal = atan(dyl/dxl);
	const value_t gamma = gammar-gammal;
	//const value_t gamma = 0.; // Testhalber
	const value_t sing = sin(gamma);
	const value_t cosg = cos(gamma);
#ifdef DEBUG
	std::cout << "gamma: " << gamma << " gamma left: " << gammal << " gamma right: " << gammar << std::endl;
#endif


#ifdef DEBUG
	std::cout << "&fidpointvec: " << std::endl;
	for (count_t i=0; i!=fidpointvec.size(); i++)
	    std::cout << i << ": " << fidpointvec[i] << std::endl;
	std::cout << "&fidpoints_: " << std::endl;
	for (count_t i=0; i!=fidpoints_.size(); i++)
	    std::cout << i << ": " << fidpoints_[i] << std::endl;
#endif

	// Matrix of the local derivatives
	ROOT::Math::SMatrix<value_t,nMsrmts,nLcD> A; // 8x4
	A(0,0)=1.; A(0,1)=0; A(0,2)=+fidpointvec[0].x(); A(0,3)=+fidpointvec[0].y();
	A(1,0)=0.; A(1,1)=1; A(1,2)=+fidpointvec[0].y(); A(1,3)=-fidpointvec[0].x();
	A(2,0)=1.; A(2,1)=0; A(2,2)=+fidpointvec[1].x(); A(2,3)=+fidpointvec[1].y();
	A(3,0)=0.; A(3,1)=1; A(3,2)=+fidpointvec[1].y(); A(3,3)=-fidpointvec[1].x();
	A(4,0)=1.; A(4,1)=0; A(4,2)=+fidpointvec[2].x(); A(4,3)=+fidpointvec[2].y();
	A(5,0)=0.; A(5,1)=1; A(5,2)=+fidpointvec[2].y(); A(5,3)=-fidpointvec[2].x();
	A(6,0)=1.; A(6,1)=0; A(6,2)=+fidpointvec[3].x(); A(6,3)=+fidpointvec[3].y();
	A(7,0)=0.; A(7,1)=1; A(7,2)=+fidpointvec[3].y(); A(7,3)=-fidpointvec[3].x();
#ifdef DEBUG
	std::cout << "A: \n" << A << std::endl;
#endif

	// Covariance matrix
	ROOT::Math::SMatrix<value_t,nMsrmts,nMsrmts> W; // 8x8
	//ROOT::Math::MatRepSym<value_t,8> W;
	const value_t sigma_x2inv = 1./(sigma_x_*sigma_x_);
	const value_t sigma_y2inv = 1./(sigma_y_*sigma_y_);
	W(0,0) = sigma_x2inv;
	W(1,1) = sigma_y2inv;
	W(2,2) = sigma_x2inv;
	W(3,3) = sigma_y2inv;
	W(4,4) = sigma_x2inv;
	W(5,5) = sigma_y2inv;
	W(6,6) = sigma_x2inv;
	W(7,7) = sigma_y2inv;
#ifdef DEBUG
	std::cout << "W: \n" << W << std::endl;
#endif

	// Prepare for the fit
	ROOT::Math::SMatrix<value_t,nLcD,nLcD> ATWA; // 4x4
	ATWA = ROOT::Math::Transpose(A) * W * A;
	//ATWA = ROOT::Math::SimilarityT(A,W); // W muss symmterisch sein -> aendern.
	//std::cout << "ATWA: \n" << ATWA << std::endl;
	ROOT::Math::SMatrix<value_t,nLcD,nLcD> ATWAi; // 4x4
	int ifail = 0;
	ATWAi = ATWA.Inverse(ifail);
	if (ifail != 0)
	{ // TODO: ifail Pruefung auf message logger ausgeben statt cout
		std::cout << "Problem singular - fit impossible." << std::endl;
		fitValidFlag_ = false;
		return;
	}
#ifdef DEBUG
	std::cout << "ATWA-1: \n" << ATWAi << ifail << std::endl;
#endif

	// Measurements
	ROOT::Math::SVector<value_t,nMsrmts> y; // 8
	y(0) = measurementVec_[0].x();
	y(1) = measurementVec_[0].y();
	y(2) = measurementVec_[1].x();
	y(3) = measurementVec_[1].y();
	y(4) = measurementVec_[2].x();
	y(5) = measurementVec_[2].y();
	y(6) = measurementVec_[3].x();
	y(7) = measurementVec_[3].y();
#ifdef DEBUG
	std::cout << "y: " << y << std::endl;
#endif

	// do the fit
	ROOT::Math::SVector<value_t,nLcD> a; // 4
	a = ATWAi * ROOT::Math::Transpose(A) * W * y;
	chi2_ = ROOT::Math::Dot(y,W*y)-ROOT::Math::Dot(a,ROOT::Math::Transpose(A)*W*y);
#ifdef DEBUG
	std::cout << "a: " << a 
		<< " S= " << sqrt(a[2]*a[2]+a[3]*a[3]) 
		<< " phi= " << atan(a[3]/a[2]) 
		<< " chi2= " << chi2_ << std::endl;
	std::cout << "A*a: " << A*a << std::endl;
#endif
	a_.assign(a.begin(),a.end());

	// Calculate vector of residuals
	r = y - A*a;
#ifdef DEBUG
	std::cout << "r: " << r << std::endl;
#endif

	// Fill matrix for global fit with local derivatives
	localDerivsMatrix_(0,0)=1.; localDerivsMatrix_(0,1)=0; 
	localDerivsMatrix_(1,0)=0.; localDerivsMatrix_(1,1)=1; 
	localDerivsMatrix_(2,0)=1.; localDerivsMatrix_(2,1)=0; 
	localDerivsMatrix_(3,0)=0.; localDerivsMatrix_(3,1)=1; 
	localDerivsMatrix_(4,0)=1.; localDerivsMatrix_(4,1)=0; 
	localDerivsMatrix_(5,0)=0.; localDerivsMatrix_(5,1)=1; 
	localDerivsMatrix_(6,0)=1.; localDerivsMatrix_(6,1)=0; 
	localDerivsMatrix_(7,0)=0.; localDerivsMatrix_(7,1)=1; 
	localDerivsMatrix_(0,2)=+fidpointvec[0].x()+cosg*fidpoints_[0].x()-sing*fidpoints_[0].y();
	localDerivsMatrix_(0,3)=+fidpointvec[0].y()+cosg*fidpoints_[0].y()+sing*fidpoints_[0].x();
	localDerivsMatrix_(1,2)=+fidpointvec[0].y()+cosg*fidpoints_[0].y()+sing*fidpoints_[0].x();
	localDerivsMatrix_(1,3)=-fidpointvec[0].x()-cosg*fidpoints_[0].x()+sing*fidpoints_[0].y();
	localDerivsMatrix_(2,2)=+fidpointvec[1].x()+cosg*fidpoints_[1].x()-sing*fidpoints_[1].y();
	localDerivsMatrix_(2,3)=+fidpointvec[1].y()+cosg*fidpoints_[1].y()+sing*fidpoints_[1].x();
	localDerivsMatrix_(3,2)=+fidpointvec[1].y()+cosg*fidpoints_[1].y()+sing*fidpoints_[1].x();
	localDerivsMatrix_(3,3)=-fidpointvec[1].x()-cosg*fidpoints_[1].x()+sing*fidpoints_[1].y();
	localDerivsMatrix_(4,2)=+fidpointvec[2].x()+cosg*fidpoints_[2].x()-sing*fidpoints_[2].y();
	localDerivsMatrix_(4,3)=+fidpointvec[2].y()+cosg*fidpoints_[2].y()+sing*fidpoints_[2].x();
	localDerivsMatrix_(5,2)=+fidpointvec[2].y()+cosg*fidpoints_[2].y()+sing*fidpoints_[2].x();
	localDerivsMatrix_(5,3)=-fidpointvec[2].x()-cosg*fidpoints_[2].x()+sing*fidpoints_[2].y();
	localDerivsMatrix_(6,2)=+fidpointvec[3].x()+cosg*fidpoints_[3].x()-sing*fidpoints_[3].y();
	localDerivsMatrix_(6,3)=+fidpointvec[3].y()+cosg*fidpoints_[3].y()+sing*fidpoints_[3].x();
	localDerivsMatrix_(7,2)=+fidpointvec[3].y()+cosg*fidpoints_[3].y()+sing*fidpoints_[3].x();
	localDerivsMatrix_(7,3)=-fidpointvec[3].x()-cosg*fidpoints_[3].x()+sing*fidpoints_[3].y();

	// Fill vector with global derivatives and labels (8x3)
	globalDerivsMatrix_(0,0) = +a(2);
	globalDerivsMatrix_(0,1) = +a(3);
	globalDerivsMatrix_(0,2) = +cosg*(a(3)*fidpoints_[0].x()-a(2)*fidpoints_[0].y())
				   -sing*(a(2)*fidpoints_[0].x()+a(3)*fidpoints_[0].y());
	globalDerivsMatrix_(1,0) = -a(3);
	globalDerivsMatrix_(1,1) = +a(2);
	globalDerivsMatrix_(1,2) = +cosg*(a(2)*fidpoints_[0].x()+a(3)*fidpoints_[0].y())
				   -sing*(a(2)*fidpoints_[0].y()-a(3)*fidpoints_[0].x());
	globalDerivsMatrix_(2,0) = +a(2);
	globalDerivsMatrix_(2,1) = +a(3);
	globalDerivsMatrix_(2,2) = +cosg*(a(3)*fidpoints_[1].x()-a(2)*fidpoints_[1].y())
				   -sing*(a(2)*fidpoints_[1].x()+a(3)*fidpoints_[1].y());
	globalDerivsMatrix_(3,0) = -a(3);
	globalDerivsMatrix_(3,1) = +a(2);
	globalDerivsMatrix_(3,2) = +cosg*(a(2)*fidpoints_[1].x()+a(3)*fidpoints_[1].y())
				   -sing*(a(2)*fidpoints_[1].y()-a(3)*fidpoints_[1].x());

	globalDerivsMatrix_(4,0) = +a(2);
	globalDerivsMatrix_(4,1) = +a(3);
	globalDerivsMatrix_(4,2) = +cosg*(a(3)*fidpoints_[2].x()-a(2)*fidpoints_[2].y())
				   -sing*(a(2)*fidpoints_[2].x()+a(3)*fidpoints_[2].y());
	globalDerivsMatrix_(5,0) = -a(3);
	globalDerivsMatrix_(5,1) = +a(2);
	globalDerivsMatrix_(5,2) = +cosg*(a(2)*fidpoints_[2].x()+a(3)*fidpoints_[2].y())
				   -sing*(a(2)*fidpoints_[2].y()-a(3)*fidpoints_[2].x());
	globalDerivsMatrix_(6,0) = +a(2);
	globalDerivsMatrix_(6,1) = +a(3);
	globalDerivsMatrix_(6,2) = +cosg*(a(3)*fidpoints_[3].x()-a(2)*fidpoints_[3].y())
				   -sing*(a(2)*fidpoints_[3].x()+a(3)*fidpoints_[3].y());
	globalDerivsMatrix_(7,0) = -a(3);
	globalDerivsMatrix_(7,1) = +a(2);
	globalDerivsMatrix_(7,2) = +cosg*(a(2)*fidpoints_[3].x()+a(3)*fidpoints_[3].y())
				   -sing*(a(2)*fidpoints_[3].y()-a(3)*fidpoints_[3].x());

	fitValidFlag_ = true;
}


void SurveyPxbImageLocalFit::doFit(value_t u1, value_t v1, value_t g1, value_t u2, value_t v2, value_t g2)
{
	// Creating vectors with the global parameters of the two modules
	ROOT::Math::SVector<value_t,4> mod1, mod2;
	mod1(0)=u1;
	mod1(1)=v1;
	mod1(2)=cos(g1);
	mod1(3)=sin(g1);
	mod2(0)=u2;
	mod2(1)=v2;
	mod2(2)=cos(g2);
	mod2(3)=sin(g2);
	//std::cout << "mod1: " << mod1 << std::endl;
	//std::cout << "mod2: " << mod2 << std::endl;

	// Create a matrix for the transformed position of the fidpoints
	ROOT::Math::SMatrix<value_t,4,4> M1, M2;
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

	ROOT::Math::SVector<value_t,4> mod_tr1, mod_tr2;
	mod_tr1 = M1*mod2;
	mod_tr2 = M2*mod1;
	//std::cout << "mod_tr1: " << mod_tr1 << std::endl;
	//std::cout << "mod_tr2: " << mod_tr2 << std::endl;

	fidpoint_t fidpointvec;
	fidpointvec.push_back(coord_t(mod_tr1(0),mod_tr1(1)));
	fidpointvec.push_back(coord_t(mod_tr1(2),mod_tr1(3)));
	fidpointvec.push_back(coord_t(mod_tr2(0),mod_tr2(1)));
	fidpointvec.push_back(coord_t(mod_tr2(2),mod_tr2(3)));

	doFit(fidpointvec);
}

SurveyPxbImageLocalFit::localpars_t SurveyPxbImageLocalFit::getLocalParameters()
{
	if (!fitValidFlag_) throw std::logic_error("SurveyPxbImageLocalFit::getLocalParameters(): Fit is not valid. Call doFit(...) before calling this function.");
	return a_;
}

SurveyPxbImageLocalFit::value_t SurveyPxbImageLocalFit::getChi2()
{
	if (!fitValidFlag_) throw std::logic_error("SurveyPxbImageLocalFit::getChi2(): Fit is not valid. Call doFit(...) before calling this function.");
	return chi2_;
}

void SurveyPxbImageLocalFit::setLocalDerivsToZero(count_t j)
{
    if (!(j < nLcD)) throw std::range_error("SurveyPxbImageLocalFit::setLocalDerivsToZero(j): j out of range.");
    for(count_t i=0; i!=nMsrmts; i++)
	localDerivsMatrix_(i,j)=0;
}

void SurveyPxbImageLocalFit::setGlobalDerivsToZero(count_t j)
{
    if (!(j < nGlD)) throw std::range_error("SurveyPxbImageLocalFit::setLocalDerivsToZero(j): j out of range.");
    for(count_t i=0; i!=nMsrmts; i++)
	globalDerivsMatrix_(i,j)=0;
}


