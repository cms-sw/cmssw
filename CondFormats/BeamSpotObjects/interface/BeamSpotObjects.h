#ifndef BEAMSPOTOBJECTS_H
#define BEAMSPOTOBJECTS_H


class BeamSpotObjects {
	
  public:
    
	BeamSpotObjects(){}
	virtual ~BeamSpotObjects(){}
    
	void SetPosition( double x, double y, double z) { 
		position_[0] = x;
		position_[1] = y;
		position_[2] = z;
	};
	void SetSigmaZ(double val) { sigmaZ_ = val; }
	void Setdxdz(double val) { dxdz_ = val; }
	void Setdydz(double val) { dydz_ = val; }
	void SetBeamWidth(double val) { beamwidth_ = val; }
	void SetCovariance(int i, int j, double val) {
		covariance_[i][j] = val;
	}

	double GetX() { return position_[0]; }
	double GetY() { return position_[1]; }
	double GetZ() { return position_[2]; }
	double GetSigmaZ() { return sigmaZ_; }
	double GetBeamWidth() { return beamwidth_; }
	double Getdxdz() { return dxdz_; }
	double Getdydz() { return dydz_; }
	double GetCovariance(int i, int j) { return covariance_[i][j]; }
	
  private:

	double position_[3];
	double sigmaZ_;
	double beamwidth_;
	double dxdz_;
	double dydz_;
	double covariance_[7][7];
	
};
#endif
