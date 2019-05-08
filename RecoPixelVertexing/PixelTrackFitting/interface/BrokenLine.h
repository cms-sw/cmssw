#ifndef RECOPIXELVERTEXING_PIXELTRACKFITTING_BROKENLINE_H
#define RECOPIXELVERTEXING_PIXELTRACKFITTING_BROKENLINE_H

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <cuda_runtime.h>

namespace BrokenLine {
	
	using namespace Eigen;
	
	constexpr unsigned int max_nop = 4;  //!< In order to avoid use of dynamic memory
	
	// WARNING: USE STATIC DIMENSIONS ON GPUs. To do so, comment these definitions and uncomment the others following
	
	using MatrixNd = Eigen::Matrix<double, Dynamic, Dynamic, 0, max_nop, max_nop>;
	using MatrixNplusONEd = Eigen::Matrix<double, Dynamic, Dynamic, 0, max_nop + 1, max_nop + 1>;
	using Matrix3Nd = Eigen::Matrix<double, Dynamic, Dynamic, 0, 3 * max_nop, 3 * max_nop>;
	using Matrix2xNd = Eigen::Matrix<double, 2, Dynamic, 0, 2, max_nop>;
	using Matrix3xNd = Eigen::Matrix<double, 3, Dynamic, 0, 3, max_nop>;
	using VectorNd = Eigen::Matrix<double, Dynamic, 1, 0, max_nop, 1>;
	using VectorNplusONEd = Eigen::Matrix<double, Dynamic, 1, 0, max_nop + 1, 1>;
	using Matrix2x3d = Eigen::Matrix<double, 2, 3>;
	using Matrix5d = Eigen::Matrix<double, 5, 5>;
	using Vector5d = Eigen::Matrix<double, 5, 1>;
	using u_int    = unsigned int;
	
	/*
	 using MatrixNd = Eigen::Matrix<double, max_nop, max_nop, 0, max_nop, max_nop>;
	 using MatrixNplusONEd = Eigen::Matrix<double, max_nop + 1, max_nop + 1, 0, max_nop + 1, max_nop + 1>;
	 using Matrix3Nd = Eigen::Matrix<double, 3 * max_nop, 3 * max_nop, 0, 3 * max_nop, 3 * max_nop>;
	 using Matrix2xNd = Eigen::Matrix<double, 2, max_nop, 0, 2, max_nop>;
	 using Matrix3xNd = Eigen::Matrix<double, 3, max_nop, 0, 3, max_nop>;
	 using VectorNd = Eigen::Matrix<double, max_nop, 1, 0, max_nop, 1>;
	 using VectorNplusONEd = Eigen::Matrix<double, max_nop + 1, 1, 0, max_nop + 1, 1>;
	 using Matrix2x3d = Eigen::Matrix<double, 2, 3>;
	 using Matrix5d = Eigen::Matrix<double, 5, 5>;
	 using Vector5d = Eigen::Matrix<double, 5, 1>;
	 using u_int    = unsigned int;
	 */
	
	struct karimaki_circle_fit {
		Vector3d par;  //!< Karimäki's parameters: (phi, d, k=1/R)
		Matrix3d cov;
		/*!< covariance matrix: \n
		 |cov(phi,phi)|cov( d ,phi)|cov( k ,phi)| \n
		 |cov(phi, d )|cov( d , d )|cov( k , d )| \n
		 |cov(phi, k )|cov( d , k )|cov( k , k )|
		 */
		int q;  //!< particle charge
		double chi2;
	};
	
	struct line_fit {
		Vector2d par;  //!< parameters: (cotan(theta),Zip)
		Matrix2d cov;
		/*!< covariance matrix: \n
		 |cov(c_t,c_t)|cov(Zip,c_t)| \n
		 |cov(c_t,Zip)|cov(Zip,Zip)|
		 */
		double chi2;
	};
	
	struct helix_fit {
		Vector5d par;  //!< parameters: (phi,Tip,p_t,cotan(theta)),Zip)
		Matrix5d cov;
		/*!< covariance matrix: \n
		 |(phi,phi)|(Tip,phi)|(p_t,phi)|(c_t,phi)|(Zip,phi)| \n
		 |(phi,Tip)|(Tip,Tip)|(p_t,Tip)|(c_t,Tip)|(Zip,Tip)| \n
		 |(phi,p_t)|(Tip,p_t)|(p_t,p_t)|(c_t,p_t)|(Zip,p_t)| \n
		 |(phi,c_t)|(Tip,c_t)|(p_t,c_t)|(c_t,c_t)|(Zip,c_t)| \n
		 |(phi,Zip)|(Tip,Zip)|(p_t,Zip)|(c_t,Zip)|(Zip,Zip)|
		 */
		double chi2_circle;
		double chi2_line;
		Vector4d fast_fit;
		int q;  //!< particle charge
	} __attribute__ ((aligned(16)) );
	
	/*!
	 \brief data needed for the Broken Line fit procedure.
	 */
	struct PreparedBrokenLineData {
		int q; //!< particle charge
		Matrix2xNd radii; //!< xy data in the system in which the pre-fitted center is the origin
		VectorNd s; //!< total distance traveled in the transverse plane starting from the pre-fitted closest approach
		VectorNd S; //!< total distance traveled (three-dimensional)
		VectorNd Z; //!< orthogonal coordinate to the pre-fitted line in the sz plane
		VectorNd VarBeta; //!< kink angles in the SZ plane
	};
	
	/*!
	 \brief raise to square.
	 */
	__host__ __device__ inline double sqr(const double a) {
		return a*a;
	}
	
	/*!
	 \brief Computes the Coulomb multiple scattering variance of the planar angle.
	 
	 \param length length of the track in the material.
	 \param B magnetic field in Gev/cm/c.
	 \param R radius of curvature (needed to evaluate p).
	 \param Layer denotes which of the four layers of the detector is the endpoint of the multiple scattered track. For example, if Layer=3, then the particle has just gone through the material between the second and the third layer.
	 
	 \todo add another Layer variable to identify also the start point of the track, so if there are missing hits or multiple hits, the part of the detector that the particle has traversed can be exactly identified.
	 
	 \warning the formula used here assumes beta=1, and so neglects the dependence of theta_0 on the mass of the particle at fixed momentum.
	 
	 \return the variance of the planar angle ((theta_0)^2 /3).
	 */
	__host__ __device__ inline double MultScatt(const double& length, const double B, const double& R, int Layer, double slope) {
		double XX_0; //!< radiation length of the material in cm
		if(Layer==1) XX_0=16/0.06;
		else XX_0=16/0.06;
		XX_0*=1;
		double geometry_factor=0.7; //!< number between 1/3 (uniform material) and 1 (thin scatterer) to be manually tuned
		return geometry_factor*sqr((13.6/1000)/(1*B*R*sqrt(1+sqr(slope))))*(abs(length)/XX_0)*sqr(1+0.038*log(abs(length)/XX_0));
	}
	
	/*!
	 \brief Computes the 2D rotation matrix that transforms the line y=slope*x into the line y=0.
	 
	 \param slope tangent of the angle of rotation.
	 
	 \return 2D rotation matrix.
	 */
	__host__ __device__ inline Matrix2d RotationMatrix(const double& slope) {
		Matrix2d Rot;
		Rot(0,0)=1/sqrt(1+sqr(slope));
		Rot(0,1)=slope*Rot(0,0);
		Rot(1,0)=-Rot(0,1);
		Rot(1,1)=Rot(0,0);
		return Rot;
	}
	
	/*!
	 \brief Changes the Karimäki parameters (and consequently their covariance matrix) under a translation of the coordinate system, such that the old origin has coordinates (x0,y0) in the new coordinate system. The formulas are taken from Karimäki V., 1990, Effective circle fitting for particle trajectories, Nucl. Instr. and Meth. A305 (1991) 187.
	 
	 \param circle circle fit in the old coordinate system.
	 \param x0 x coordinate of the translation vector.
	 \param y0 y coordinate of the translation vector.
	 \param Jacob passed by reference in order to save stack.
	 */
	__host__ __device__ inline void TranslateKarimaki(karimaki_circle_fit& circle, const double& x0, const double& y0, Matrix3d& Jacob) {
		double A,U,BB,C,DO,DP,uu,xi,v,mu,lambda,zeta;
		DP=x0*cos(circle.par(0))+y0*sin(circle.par(0));
		DO=x0*sin(circle.par(0))-y0*cos(circle.par(0))+circle.par(1);
		uu=1+circle.par(2)*circle.par(1);
		C=-circle.par(2)*y0+uu*cos(circle.par(0));
		BB=circle.par(2)*x0+uu*sin(circle.par(0));
		A=2*DO+circle.par(2)*(sqr(DO)+sqr(DP));
		U=sqrt(1+circle.par(2)*A);
		xi=1/(sqr(BB)+sqr(C));
		v=1+circle.par(2)*DO;
		lambda=(A/2)/(U*sqr(1+U));
		mu=1/(U*(1+U))+circle.par(2)*lambda;
		zeta=sqr(DO)+sqr(DP);
		
		Jacob << xi*uu*v, -xi*sqr(circle.par(2))*DP, xi*DP,
		2*mu*uu*DP, 2*mu*v, mu*zeta-lambda*A,
		0, 0, 1;
		
		circle.par(0)=atan2(BB,C);
		circle.par(1)=A/(1+U);
		// circle.par(2)=circle.par(2);
		
		circle.cov=Jacob*circle.cov*Jacob.transpose();
	}
	
	/*!
	 \brief Compute cross product of two 2D vector (assuming z component 0), returning the z component of the result.
	 
	 \param a first 2D vector in the product.
	 \param b second 2D vector in the product.
	 
	 \return z component of the cross product.
	 */
	
	__host__ __device__ inline double cross2D(const Vector2d& a, const Vector2d& b) {
		return a.x()*b.y()-a.y()*b.x();
	}
	
	/*!
	 \brief Computes the data needed for the Broken Line fit procedure that are mainly common for the circle and the line fit.
	 
	 \param hits hits coordinates.
	 \param hits_cov hits covariance matrix.
	 \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
	 \param B magnetic field in Gev/cm/c.
	 \param results PreparedBrokenLineData to be filled (see description of PreparedBrokenLineData).
	 */
	__host__ __device__ inline void PrepareBrokenLineData(const Matrix3xNd& hits,
														  const Matrix3Nd& hits_cov,
														  const Vector4d& fast_fit,
														  const double B,
														  PreparedBrokenLineData & results) {
		u_int n=hits.cols();
		u_int i;
		Vector2d d;
		Vector2d e;
		results.radii=Matrix2xNd::Zero(2,n);
		results.s=VectorNd::Zero(n);
		results.S=VectorNd::Zero(n);
		results.Z=VectorNd::Zero(n);
		results.VarBeta=VectorNd::Zero(n);
		
		results.q=1;
		d=hits.block(0,1,2,1)-hits.block(0,0,2,1);
		e=hits.block(0,n-1,2,1)-hits.block(0,n-2,2,1);
		if(cross2D(d,e)>0) results.q=-1;
		
		const double slope=-results.q/fast_fit(3);
		
		Matrix2d R=RotationMatrix(slope);
		
		// calculate radii and s
		results.radii=hits.block(0,0,2,n)-fast_fit.head(2)*MatrixXd::Constant(1,n,1);
		e=-fast_fit(2)*fast_fit.head(2)/fast_fit.head(2).norm();
		for(i=0;i<n;i++) {
			d=results.radii.block(0,i,2,1);
			results.s(i)=results.q*fast_fit(2)*atan2(cross2D(d,e),d.dot(e)); // calculates the arc length
			//if(results.s(i)<=0);
		}
		VectorNd z=hits.block(2,0,1,n).transpose();
		
		//calculate S and Z
		Matrix2xNd pointsSZ=Matrix2xNd::Zero(2,n);
		for(i=0;i<n;i++) {
			pointsSZ(0,i)=results.s(i);
			pointsSZ(1,i)=z(i);
			pointsSZ.block(0,i,2,1)=R*pointsSZ.block(0,i,2,1);
		}
		results.S=pointsSZ.block(0,0,1,n).transpose();
		results.Z=pointsSZ.block(1,0,1,n).transpose();
		
		//calculate VarBeta
		for(i=1;i<n-1;i++) {
			results.VarBeta(i)=MultScatt(results.S(i+1)-results.S(i),B,fast_fit(2),i+2,slope)+MultScatt(results.S(i)-results.S(i-1),B,fast_fit(2),i+1,slope);
		}
	}
	
	/*!
	 \brief Computes the n-by-n band matrix obtained minimizing the Broken Line's cost function w.r.t u. This is the whole matrix in the case of the line fit and the main n-by-n block in the case of the circle fit.
	 
	 \param w weights of the first part of the cost function, the one with the measurements and not the angles (\sum_{i=1}^n w*(y_i-u_i)^2).
	 \param S total distance traveled by the particle from the pre-fitted closest approach.
	 \param VarBeta kink angles' variance.
	 
	 \return the n-by-n matrix of the linear system
	 */
	__host__ __device__ inline MatrixNd MatrixC_u(const VectorNd& w, const VectorNd& S, const VectorNd& VarBeta) {
		u_int n=S.rows();
		u_int i;
		
		MatrixNd C_U=MatrixNd::Zero(n,n);
		for(i=0;i<n;i++) {
			C_U(i,i)=w(i);
			if(i>1) C_U(i,i)+=1/(VarBeta(i-1)*sqr(S(i)-S(i-1)));
			if(i>0 && i<n-1) C_U(i,i)+=(1/VarBeta(i))*sqr((S(i+1)-S(i-1))/((S(i+1)-S(i))*(S(i)-S(i-1))));
			if(i<n-2) C_U(i,i)+=1/(VarBeta(i+1)*sqr(S(i+1)-S(i)));
			
			if(i>0 && i<n-1) C_U(i,i+1)=1/(VarBeta(i)*(S(i+1)-S(i)))*(-(S(i+1)-S(i-1))/((S(i+1)-S(i))*(S(i)-S(i-1))));
			if(i<n-2) C_U(i,i+1)+=1/(VarBeta(i+1)*(S(i+1)-S(i)))*(-(S(i+2)-S(i))/((S(i+2)-S(i+1))*(S(i+1)-S(i))));
			
			if(i<n-2) C_U(i,i+2)=1/(VarBeta(i+1)*(S(i+2)-S(i+1))*(S(i+1)-S(i)));
			
			C_U(i,i)=C_U(i,i)/2;
		}
		MatrixNd C_u;
		C_u=C_U+C_U.transpose();
		
		return C_u;
	}
	
	/*!
	 \brief A very fast helix fit.
	 
	 \param hits the measured hits.
	 
	 \return (X0,Y0,R,tan(theta)).
	 
	 \warning sign of theta is (intentionally, for now) mistaken for negative charges.
	 */
	
	__host__ __device__ inline Vector4d BL_Fast_fit(const Matrix3xNd& hits) {
		Vector4d result;
		u_int n=hits.cols();
		
		const Vector2d a=hits.block(0,n/2,2,1)-hits.block(0,0,2,1);
		const Vector2d b=hits.block(0,n-1,2,1)-hits.block(0,n/2,2,1);
		const Vector2d c=hits.block(0,0,2,1)-hits.block(0,n-1,2,1);
		
		result(0)=hits(0,0)-(a(1)*c.squaredNorm()+c(1)*a.squaredNorm())/(2*cross2D(c,a));
		result(1)=hits(1,0)+(a(0)*c.squaredNorm()+c(0)*a.squaredNorm())/(2*cross2D(c,a));
		// check Wikipedia for these formulas
		
		result(2)=(a.norm()*b.norm()*c.norm())/(2*abs(cross2D(b,a)));
		// Using Math Olympiad's formula R=abc/(4A)
		
		const Vector2d d=hits.block(0,0,2,1)-result.head(2);
		const Vector2d e=hits.block(0,n-1,2,1)-result.head(2);
		
		result(3)=result(2)*atan2(cross2D(d, e), d.dot(e))/(hits(2,n-1)-hits(2,0));
		// ds/dz slope between last and first point
		
		return result;
	}
	
	/*!
	 \brief Performs the Broken Line fit in the curved track case (that is, the fit parameters are the interceptions u and the curvature correction \Delta\kappa).
	 
	 \param hits hits coordinates.
	 \param hits_cov hits covariance matrix.
	 \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
	 \param B magnetic field in Gev/cm/c.
	 \param data PreparedBrokenLineData.
	 \param circle_results struct to be filled with the results in this form:
	 -par parameter of the line in this form: (phi, d, k); \n
	 -cov covariance matrix of the fitted parameter; \n
	 -chi2 value of the cost function in the minimum.
	 \param Jacob passed by reference in order to save stack.
	 \param C_U passed by reference in order to sake stack.
	 
	 \details The function implements the steps 2 and 3 of the Broken Line fit with the curvature correction.\n
	 The step 2 is the least square fit, done by imposing the minimum constraint on the cost function and solving the consequent linear system. It determines the fitted parameters u and \Delta\kappa and their covariance matrix.
	 The step 3 is the correction of the fast pre-fitted parameters for the innermost part of the track. It is first done in a comfortable coordinate system (the one in which the first hit is the origin) and then the parameters and their covariance matrix are transformed to the original coordinate system.
	 */
	
	__host__ __device__ inline void BL_Circle_fit(const Matrix3xNd& hits,
												  const Matrix3Nd& hits_cov,
												  const Vector4d& fast_fit,
												  const double B,
												  PreparedBrokenLineData& data,
												  karimaki_circle_fit & circle_results,
												  Matrix3d& Jacob,
												  MatrixNplusONEd& C_U) {
		u_int n=hits.cols();
		u_int i;
		
		circle_results.q=data.q;
		Matrix2xNd& radii=data.radii;
		const VectorNd& s=data.s;
		const VectorNd& S=data.S;
		VectorNd& Z=data.Z;
		VectorNd& VarBeta=data.VarBeta;
		const double slope=-circle_results.q/fast_fit(3);
		VarBeta*=1+sqr(slope); // the kink angles are projected!
		
		for(i=0;i<n;i++) {
			Z(i)=radii.block(0,i,2,1).norm()-fast_fit(2);
		}
		
		Matrix2d V; // covariance matrix
		VectorNd w=VectorNd::Zero(n); // weights
		Matrix2d RR; // rotation matrix point by point
		//double Slope; // slope of the circle point by point
		for(i=0;i<n;i++) {
			V(0,0)=hits_cov(i,i); // I could not find an easy access to sub-matrices in Eigen...
			V(0,1)=hits_cov(i,i+n);
			V(1,0)=hits_cov(i+n,i);
			V(1,1)=hits_cov(i+n,i+n);
			//Slope=-radii(0,i)/radii(1,i);
			RR=RotationMatrix(-radii(0,i)/radii(1,i));
			w(i)=1/((RR*V*RR.transpose())(1,1)); // compute the orthogonal weight point by point
		}
		
		VectorNplusONEd r_u=VectorNplusONEd::Zero(n+1);
		for(i=0;i<n;i++) {
			r_u(i)=w(i)*Z(i);
		} r_u(n)=0;
		
		C_U=MatrixNplusONEd::Zero(n+1,n+1);
		//add the border to the C_u matrix
		for(i=0;i<n;i++) {
			if(i>0 && i<n-1) {
				C_U(i,n)+=-(s(i+1)-s(i-1))/(2*VarBeta(i))*(s(i+1)-s(i-1))/((s(i+1)-s(i))*(s(i)-s(i-1)));
				C_U(n,i)=C_U(i,n);
			}
			if(i>1) {
				C_U(i,n)+=(s(i)-s(i-2))/(2*VarBeta(i-1)*(s(i)-s(i-1)));
				C_U(n,i)=C_U(i,n);
			}
			if(i<n-2) {
				C_U(i,n)+=(s(i+2)-s(i))/(2*VarBeta(i+1)*(s(i+1)-s(i)));
				C_U(n,i)=C_U(i,n);
			}
			
			if(i>0 && i<n-1) C_U(n,n)+=sqr(s(i+1)-s(i-1))/(4*VarBeta(i));
		}
		C_U.block(0,0,n,n)=MatrixC_u(w,s,VarBeta);
		MatrixNplusONEd& I=C_U;
		I=C_U.inverse();//MatrixNplusONEd I=C_U.inverse();
		
		VectorNplusONEd& u=r_u;
		u=I*r_u; // obtain the fitted parameters by solving the linear system
		
		// compute (phi, d_ca, k) in the system in which the midpoint of the first two corrected hits is the origin...
		
		radii.block(0,0,2,1)/=radii.block(0,0,2,1).norm();
		radii.block(0,1,2,1)/=radii.block(0,1,2,1).norm();
		
		Vector2d d=hits.block(0,0,2,1)+(-Z(0)+u(0))*radii.block(0,0,2,1);
		Vector2d e=hits.block(0,1,2,1)+(-Z(1)+u(1))*radii.block(0,1,2,1);
		
		circle_results.par << atan2((e-d)(1),(e-d)(0)),
		-circle_results.q*(fast_fit(2)-sqrt(sqr(fast_fit(2))-(e-d).squaredNorm()/4)),
		circle_results.q*(1/fast_fit(2)+u(n));
		
		assert(circle_results.q*circle_results.par(1)<=0);
		
		Vector2d eMinusd=e-d;
		double tmp1=eMinusd.squaredNorm();
		
		Jacob << (radii(1,0)*eMinusd(0)-eMinusd(1)*radii(0,0))/tmp1,(radii(1,1)*eMinusd(0)-eMinusd(1)*radii(0,1))/tmp1,0,
		(circle_results.q/2)*(eMinusd(0)*radii(0,0)+eMinusd(1)*radii(1,0))/sqrt(sqr(2*fast_fit(2))-tmp1),(circle_results.q/2)*(eMinusd(0)*radii(0,1)+eMinusd(1)*radii(1,1))/sqrt(sqr(2*fast_fit(2))-tmp1),0,
		0,0,circle_results.q;
		
		circle_results.cov << I(0,0), I(0,1), I(0,n),
		I(1,0), I(1,1), I(1,n),
		I(n,0), I(n,1), I(n,n);
		
		circle_results.cov=Jacob*circle_results.cov*Jacob.transpose();
		
		//...Translate in the system in which the first corrected hit is the origin, adding the m.s. correction...
		
		TranslateKarimaki(circle_results,(e-d)(0)/2,(e-d)(1)/2,Jacob);
		circle_results.cov(0,0)+=(1+sqr(slope))*MultScatt(S(1)-S(0),B,fast_fit(2),2,slope);
		
		//...And translate back to the original system
		
		TranslateKarimaki(circle_results,d(0),d(1),Jacob);
		
		// compute chi2
		circle_results.chi2=0;
		for(i=0;i<n;i++) {
			circle_results.chi2+=w(i)*sqr(Z(i)-u(i));
			if(i>0 && i<n-1) circle_results.chi2+=sqr(u(i-1)/(s(i)-s(i-1))-u(i)*(s(i+1)-s(i-1))/((s(i+1)-s(i))*(s(i)-s(i-1)))+u(i+1)/(s(i+1)-s(i))+(s(i+1)-s(i-1))*u(n)/2)/VarBeta(i);
		}
		
		assert(circle_results.chi2>=0);
	}
	
	/*!
	 \brief Performs the Broken Line fit in the straight track case (that is, the fit parameters are only the interceptions u).
	 
	 \param hits hits coordinates.
	 \param hits_cov hits covariance matrix.
	 \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
	 \param B magnetic field in Gev/cm/c.
	 \param data PreparedBrokenLineData.
	 \param line_results struct to be filled with the results in this form:
	 -par parameter of the line in this form: (cot(theta), Zip); \n
	 -cov covariance matrix of the fitted parameter; \n
	 -chi2 value of the cost function in the minimum.
	 
	 \details The function implements the steps 2 and 3 of the Broken Line fit without the curvature correction.\n
	 The step 2 is the least square fit, done by imposing the minimum constraint on the cost function and solving the consequent linear system. It determines the fitted parameters u and their covariance matrix.
	 The step 3 is the correction of the fast pre-fitted parameters for the innermost part of the track. It is first done in a comfortable coordinate system (the one in which the first hit is the origin) and then the parameters and their covariance matrix are transformed to the original coordinate system.
	 */
	
	__host__ __device__ inline void BL_Line_fit(const Matrix3xNd& hits,
												const Matrix3Nd& hits_cov,
												const Vector4d& fast_fit,
												const double B,
												const PreparedBrokenLineData& data,
												line_fit & line_results) {
		u_int n=hits.cols();
		u_int i;
		
		const Matrix2xNd& radii=data.radii;
		const VectorNd& S=data.S;
		const VectorNd& Z=data.Z;
		const VectorNd& VarBeta=data.VarBeta;
		
		const double slope=-data.q/fast_fit(3);
		Matrix2d R=RotationMatrix(slope);
		
		Matrix3d V=Matrix3d::Zero(); // covariance matrix XYZ
		Matrix2x3d JacobXYZtosZ=Matrix2x3d::Zero(); // jacobian for computation of the error on s (xyz -> sz)
		VectorNd w=VectorNd::Zero(n);
		for(i=0;i<n;i++) {
			V(0,0)=hits_cov(i,i); // I could not find an easy way to access the sub-matrices in Eigen...
			V(0,1)=hits_cov(i,i+n);
			V(0,2)=hits_cov(i,i+2*n);
			V(1,0)=hits_cov(i+n,i);
			V(1,1)=hits_cov(i+n,i+n);
			V(1,2)=hits_cov(i+n,i+2*n);
			V(2,0)=hits_cov(i+2*n,i);
			V(2,1)=hits_cov(i+2*n,i+n);
			V(2,2)=hits_cov(i+2*n,i+2*n);
			JacobXYZtosZ(0,0)=radii(1,i)/radii.block(0,i,2,1).norm();
			JacobXYZtosZ(0,1)=-radii(0,i)/radii.block(0,i,2,1).norm();
			JacobXYZtosZ(1,2)=1;
			w(i)=1/((R*JacobXYZtosZ*V*JacobXYZtosZ.transpose()*R.transpose())(1,1)); // compute the orthogonal weight point by point
		}
		
		VectorNd r_u=VectorNd::Zero(n);
		for(i=0;i<n;i++) {
			r_u(i)=w(i)*Z(i);
		}
		
		MatrixNd I=MatrixC_u(w,S,VarBeta).inverse();
		VectorNd u=I*r_u; // obtain the fitted parameters by solving the linear system
		
		// line parameters in the system in which the first hit is the origin and with axis along SZ
		line_results.par << (u(1)-u(0))/(S(1)-S(0)), u(0);
		line_results.cov << (I(0,0)-2*I(0,1)+I(1,1))/sqr(S(1)-S(0))+MultScatt(S(1)-S(0),B,fast_fit(2),2,slope), (I(0,1)-I(0,0))/(S(1)-S(0)),
		(I(0,1)-I(0,0))/(S(1)-S(0)), I(0,0);
		
		// translate to the original SZ system
		Matrix2d Jacob;
		Jacob(0,0)=1;
		Jacob(0,1)=0;
		Jacob(1,0)=-S(0);
		Jacob(1,1)=1;
		line_results.par(1)+=-line_results.par(0)*S(0);
		line_results.cov=Jacob*line_results.cov*Jacob.transpose();
		
		// rotate to the original sz system
		double tmp=R(0,0)-line_results.par(0)*R(0,1);
		Jacob(0,0)=1/sqr(tmp);
		Jacob(0,1)=0;
		Jacob(1,0)=line_results.par(1)*R(0,1)/sqr(tmp);
		Jacob(1,1)=1/tmp;
		line_results.par(1)=line_results.par(1)/tmp;
		line_results.par(0)=(R(0,1)+line_results.par(0)*R(0,0))/tmp;
		line_results.cov=Jacob*line_results.cov*Jacob.transpose();
		
		// compute chi2
		line_results.chi2=0;
		for(i=0;i<n;i++) {
			line_results.chi2+=w(i)*sqr(Z(i)-u(i));
			if(i>0 && i<n-1) line_results.chi2+=sqr(u(i-1)/(S(i)-S(i-1))-u(i)*(S(i+1)-S(i-1))/((S(i+1)-S(i))*(S(i)-S(i-1)))+u(i+1)/(S(i+1)-S(i)))/VarBeta(i);
		}
		
		assert(line_results.chi2>=0);
	}
	
	/*!
	 \brief Helix fit by three step:
	 -fast pre-fit (see Fast_fit() for further info); \n
	 -circle fit of the hits projected in the transverse plane by Broken Line algorithm (see BL_Circle_fit() for further info); \n
	 -line fit of the hits projected on the (pre-fitted) cilinder surface by Broken Line algorithm (see BL_Line_fit() for further info); \n
	 Points must be passed ordered (from inner to outer layer).
	 
	 \param hits Matrix3xNd hits coordinates in this form: \n
	 |x1|x2|x3|...|xn| \n
	 |y1|y2|y3|...|yn| \n
	 |z1|z2|z3|...|zn|
	 \param hits_cov Matrix3Nd covariance matrix in this form (()->cov()): \n
	 |(x1,x1)|(x2,x1)|(x3,x1)|(x4,x1)|.|(y1,x1)|(y2,x1)|(y3,x1)|(y4,x1)|.|(z1,x1)|(z2,x1)|(z3,x1)|(z4,x1)| \n
	 |(x1,x2)|(x2,x2)|(x3,x2)|(x4,x2)|.|(y1,x2)|(y2,x2)|(y3,x2)|(y4,x2)|.|(z1,x2)|(z2,x2)|(z3,x2)|(z4,x2)| \n
	 |(x1,x3)|(x2,x3)|(x3,x3)|(x4,x3)|.|(y1,x3)|(y2,x3)|(y3,x3)|(y4,x3)|.|(z1,x3)|(z2,x3)|(z3,x3)|(z4,x3)| \n
	 |(x1,x4)|(x2,x4)|(x3,x4)|(x4,x4)|.|(y1,x4)|(y2,x4)|(y3,x4)|(y4,x4)|.|(z1,x4)|(z2,x4)|(z3,x4)|(z4,x4)| \n
	 .       .       .       .       . .       .       .       .       . .       .       .       .       . \n
	 |(x1,y1)|(x2,y1)|(x3,y1)|(x4,y1)|.|(y1,y1)|(y2,y1)|(y3,x1)|(y4,y1)|.|(z1,y1)|(z2,y1)|(z3,y1)|(z4,y1)| \n
	 |(x1,y2)|(x2,y2)|(x3,y2)|(x4,y2)|.|(y1,y2)|(y2,y2)|(y3,x2)|(y4,y2)|.|(z1,y2)|(z2,y2)|(z3,y2)|(z4,y2)| \n
	 |(x1,y3)|(x2,y3)|(x3,y3)|(x4,y3)|.|(y1,y3)|(y2,y3)|(y3,x3)|(y4,y3)|.|(z1,y3)|(z2,y3)|(z3,y3)|(z4,y3)| \n
	 |(x1,y4)|(x2,y4)|(x3,y4)|(x4,y4)|.|(y1,y4)|(y2,y4)|(y3,x4)|(y4,y4)|.|(z1,y4)|(z2,y4)|(z3,y4)|(z4,y4)| \n
	 .       .       .    .          . .       .       .       .       . .       .       .       .       . \n
	 |(x1,z1)|(x2,z1)|(x3,z1)|(x4,z1)|.|(y1,z1)|(y2,z1)|(y3,z1)|(y4,z1)|.|(z1,z1)|(z2,z1)|(z3,z1)|(z4,z1)| \n
	 |(x1,z2)|(x2,z2)|(x3,z2)|(x4,z2)|.|(y1,z2)|(y2,z2)|(y3,z2)|(y4,z2)|.|(z1,z2)|(z2,z2)|(z3,z2)|(z4,z2)| \n
	 |(x1,z3)|(x2,z3)|(x3,z3)|(x4,z3)|.|(y1,z3)|(y2,z3)|(y3,z3)|(y4,z3)|.|(z1,z3)|(z2,z3)|(z3,z3)|(z4,z3)| \n
	 |(x1,z4)|(x2,z4)|(x3,z4)|(x4,z4)|.|(y1,z4)|(y2,z4)|(y3,z4)|(y4,z4)|.|(z1,z4)|(z2,z4)|(z3,z4)|(z4,z4)|
	 \param B magnetic field in the center of the detector in Gev/cm/c, in order to perform the p_t calculation.
	 
	 \warning see BL_Circle_fit(), BL_Line_fit() and Fast_fit() warnings.
	 
	 \bug see BL_Circle_fit(), BL_Line_fit() and Fast_fit() bugs.
	 
	 \return (phi,Tip,p_t,cot(theta)),Zip), their covariance matrix and the chi2's of the circle and line fits.
	 */
	
	__host__ __device__ inline helix_fit Helix_fit(const Matrix3xNd& hits,
												   const Matrix3Nd& hits_cov,
												   const double B) {
		helix_fit helix;
		
		helix.fast_fit=BL_Fast_fit(hits);
		
		PreparedBrokenLineData data;
		karimaki_circle_fit circle;
		line_fit line;
		Matrix3d Jacob;
		MatrixNplusONEd C_U;
		
		PrepareBrokenLineData(hits,hits_cov,helix.fast_fit,B,data);
		BL_Line_fit(hits,hits_cov,helix.fast_fit,B,data,line);
		BL_Circle_fit(hits,hits_cov,helix.fast_fit,B,data,circle,Jacob,C_U);
		
		// the circle fit gives k, but here we want p_t, so let's change the parameter and the covariance matrix
		Jacob << 1,0,0,
		0,1,0,
		0,0,-abs(circle.par(2))*B/(sqr(circle.par(2))*circle.par(2));
		circle.par(2)=B/abs(circle.par(2));
		circle.cov=Jacob*circle.cov*Jacob.transpose();
		
		helix.par << circle.par, line.par;
		helix.cov=MatrixXd::Zero(5, 5);
		helix.cov.block(0,0,3,3)=circle.cov;
		helix.cov.block(3,3,2,2)=line.cov;
		helix.q=circle.q;
		helix.chi2_circle=circle.chi2;
		helix.chi2_line=line.chi2;
		
		return helix;
	}
	
}  // namespace BrokenLine

#endif
