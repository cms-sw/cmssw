#ifndef BEAMSPOTOBJECTS_H
#define BEAMSPOTOBJECTS_H
/** \class BeamSpotObjects
 *
 * Reconstructed beam spot object. It provides position, error, and
 * width of the beam position.
 *
 * \author Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
 *
 * \version $Id: BeamSpotObjects.h,v 1.10 2009/03/26 19:43:30 yumiceva Exp $
 *
 */

#include <math.h>
#include <sstream>
#include <cstring>

class BeamSpotObjects {
	
  public:

	/// default constructor
	BeamSpotObjects(): sigmaZ_(0), beamwidthX_(0), beamwidthY_(0),
		dxdz_(0), dydz_(0), type_(-1) {

		beamwidthXError_ = 0;
		beamwidthYError_ = 0;
		emittanceX_ = 0;
		emittanceY_ = 0;
		betaStar_ = 0;
		std::memset(position_, 0, sizeof position_);
		std::memset(covariance_, 0, sizeof covariance_);
	}
	
	virtual ~BeamSpotObjects(){}

	/// set XYZ position
	void SetPosition( double x, double y, double z) { 
		position_[0] = x;
		position_[1] = y;
		position_[2] = z;
	};
	/// set sigma Z, RMS bunch length
	void SetSigmaZ(double val) { sigmaZ_ = val; }
	/// set dxdz slope, crossing angle
	void Setdxdz(double val) { dxdz_ = val; }
	/// set dydz slope, crossing angle in XZ
	void Setdydz(double val) { dydz_ = val; }
	/// set average transverse beam width X
	void SetBeamWidthX(double val) { beamwidthX_ = val; }
    /// set average transverse beam width Y
	void SetBeamWidthY(double val) { beamwidthY_ = val; }
	/// set beam width X error
	void SetBeamWidthXError(double val) { beamwidthXError_ = val; }
	/// set beam width Y error
	void SetBeamWidthYError(double val) { beamwidthYError_ = val; }
	/// set i,j element of the full covariance matrix 7x7
	void SetCovariance(int i, int j, double val) {
		covariance_[i][j] = val;
	}
	/// set beam type
	void SetType(int type) { type_ = type; }
	/// set emittance
	void SetEmittanceX(double val) { emittanceX_ = val;}
	/// set emittance
	void SetEmittanceY(double val) { emittanceY_ = val;}
	/// set beta star
	void SetBetaStar(double val) { betaStar_ = val;}
	
	/// get X beam position
	double GetX() const { return position_[0]; }
	/// get Y beam position
	double GetY() const { return position_[1]; }
	/// get Z beam position
	double GetZ() const { return position_[2]; }
	/// get sigma Z, RMS bunch length
	double GetSigmaZ() const { return sigmaZ_; }
	/// get average transverse beam width
	double GetBeamWidthX() const { return beamwidthX_; }
	/// get average transverse beam width
	double GetBeamWidthY() const { return beamwidthY_; }
	/// get dxdz slope, crossing angle in XZ
	double Getdxdz() const { return dxdz_; }
	/// get dydz slope, crossing angle in YZ
	double Getdydz() const { return dydz_; }
	/// get i,j element of the full covariance matrix 7x7
	double GetCovariance(int i, int j) const { return covariance_[i][j]; }
	/// get X beam position Error
	double GetXError() const { return sqrt(covariance_[0][0]); }
	/// get Y beam position Error
	double GetYError() const { return sqrt(covariance_[1][1]); }
	/// get Z beam position Error
	double GetZError() const { return sqrt(covariance_[2][2]); }
	/// get sigma Z, RMS bunch length Error
	double GetSigmaZError() const { return sqrt(covariance_[3][3]); }
	/// get average transverse beam width error ASSUME the same for X and Y
	double GetBeamWidthXError() const { return sqrt(covariance_[6][6]); }
    /// get average transverse beam width error X = Y
	double GetBeamWidthYError() const { return sqrt(covariance_[6][6]); }
	/// get dxdz slope, crossing angle in XZ Error
	double GetdxdzError() const { return sqrt(covariance_[4][4]); }
	/// get dydz slope, crossing angle in YZ Error
	double GetdydzError() const { return sqrt(covariance_[5][5]); }
	/// get beam type
	int GetBeamType() const { return type_; }
	/// get emittance
	double GetEmittanceX() const { return emittanceX_; }
	/// get emittance
	double GetEmittanceY() const { return emittanceY_; }
	/// get beta star
	double GetBetaStar() const { return betaStar_; }
	
	/// print beam spot parameters
	void print(std::stringstream& ss) const;

  private:

	double position_[3];
	double sigmaZ_;
	double beamwidthX_;
	double beamwidthY_;
	double beamwidthXError_;
	double beamwidthYError_;
	double dxdz_;
	double dydz_;
	double covariance_[7][7];
	int type_;
	double emittanceX_;
	double emittanceY_;
	double betaStar_;
	
};

std::ostream& operator<< ( std::ostream&, BeamSpotObjects beam );

#endif
