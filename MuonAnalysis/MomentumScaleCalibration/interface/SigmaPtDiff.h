#ifndef SigmaPtDiff_h
#define SigmaPtDiff_h

#include <vector>
#include <cmath>

class SigmaPt
{
 public:
  SigmaPt( const std::vector<double> & parameters_,
	   const std::vector<double> & errors_ )
    {
      setParErr(parameters_, errors_);
    }

  SigmaPt() {};
  
  void setParErr( const std::vector<double> & parameters,
		  const std::vector<double> & errors )
  {
    b_0 = parameters[0];
    b_1 = parameters[5];
    b_2 = parameters[1];
    b_3 = parameters[7];
    b_4 = parameters[8];
    sb_0 = errors[0];
    sb_1 = errors[5];
    sb_2 = errors[1];
    sb_3 = errors[7];
    sb_4 = errors[8];
    c = b_2 + b_3*(b_0 - b_4)*(b_0 - b_4) - b_1*b_0*b_0;
  }

  double operator()(const double & eta)
  {
    if( fabs(eta) <= b_0 ) {
      return( c + b_1*eta*eta );
    }
    return( b_2 + b_3*(fabs(eta) - b_4)*(fabs(eta) - b_4) );
  }
  double sigma(const double & eta)
  {
    if( fabs(eta) <= b_0 ) {
      return sqrt( (eta*eta - b_0*b_0)*(eta*eta - b_0*b_0)*sb_1*sb_1 +
		   sb_2*sb_2 +
		   pow(b_0 - b_4, 4)*sb_3*sb_3 +
		   pow(-2*b_3*pow(b_0-b_4,2), 2)*sb_4*sb_4 +
		   pow(2*b_3*(b_0 - b_4) - 2*b_1*b_0, 2)*sb_0*sb_0 );
    }
    return sqrt( sb_2*sb_2 +
		 pow(fabs(eta) - b_4, 4)*sb_3*sb_3 +
		 pow(-2*b_3*pow(fabs(eta)-b_4,2), 2)*sb_4*sb_4 );
  }

 protected:
  double b_0;
  double b_1;
  double b_2;
  double b_3;
  double b_4;
  double c;

  double sb_0;
  double sb_1;
  double sb_2;
  double sb_3;
  double sb_4;
};

/// Returns ( sigmaPt/Pt(data)^2 - sigmaPt/Pt(MC)^2 )
class SigmaPtDiff
{
 public:
  SigmaPtDiff()
  {
    std::vector<double> parameters;
    std::vector<double> errors;
    parameters.push_back(1.66);
    parameters.push_back(0.021);
    parameters.push_back(0.);
    parameters.push_back(0.);
    parameters.push_back(0.);
    parameters.push_back(0.0058);
    parameters.push_back(0.);
    parameters.push_back(0.03);
    parameters.push_back(1.8);
    parameters.push_back(0.);
    parameters.push_back(0.);
    parameters.push_back(0.);
    parameters.push_back(0.);
    parameters.push_back(0.);
    parameters.push_back(0.);
    errors.push_back(0.09);
    errors.push_back(0.002);
    errors.push_back(0.);
    errors.push_back(0.);
    errors.push_back(0.);
    errors.push_back(0.0009);
    errors.push_back(0.);
    errors.push_back(0.03);
    errors.push_back(0.3);
    errors.push_back(0.);
    errors.push_back(0.);
    errors.push_back(0.);
    errors.push_back(0.);
    errors.push_back(0.);
    errors.push_back(0.);

    sigmaPt.setParErr( parameters, errors );
  }
  double etaByPoints(const double & eta)
  {
    if(      eta <= -2.2 ) return 0.0233989;
    else if( eta <= -2.0 ) return 0.0197057;
    else if( eta <= -1.8 ) return 0.014693;
    else if( eta <= -1.6 ) return 0.0146727;
    else if( eta <= -1.4 ) return 0.0141323;
    else if( eta <= -1.2 ) return 0.0159712;
    else if( eta <= -1.0 ) return 0.0117224;
    else if( eta <= -0.8 ) return 0.010726;
    else if( eta <= -0.6 ) return 0.0104777;
    else if( eta <= -0.4 ) return 0.00814458;
    else if( eta <= -0.2 ) return 0.00632501;
    else if( eta <=  0.0 ) return 0.00644172;
    else if( eta <=  0.2 ) return 0.00772645;
    else if( eta <=  0.4 ) return 0.010103;
    else if( eta <=  0.6 ) return 0.0099275;
    else if( eta <=  0.8 ) return 0.0100309;
    else if( eta <=  1.0 ) return 0.0125116;
    else if( eta <=  1.2 ) return 0.0147211;
    else if( eta <=  1.4 ) return 0.0151623;
    else if( eta <=  1.6 ) return 0.015259;
    else if( eta <=  1.8 ) return 0.014499;
    else if( eta <=  2.0 ) return 0.0165215;
    else if( eta <=  2.2 ) return 0.0212348;
    return 0.0227285;
  }
  // double squaredDiff(const double & eta, SigmaPt & sigmaPt)
  double squaredDiff(const double & eta)
  {
    double sigmaPtPlus = sigmaPt(eta) + sigmaPt.sigma(eta);
    double sigmaPtMinus = sigmaPt(eta) - sigmaPt.sigma(eta);
    if( fabs(sigmaPtPlus*sigmaPtPlus - etaByPoints(eta)*etaByPoints(eta)) > fabs(sigmaPtMinus*sigmaPtMinus - etaByPoints(eta)*etaByPoints(eta)) ) {
      return( fabs(sigmaPtPlus*sigmaPtPlus - etaByPoints(eta)*etaByPoints(eta)) );
    }
    return( fabs(sigmaPtMinus*sigmaPtMinus - etaByPoints(eta)*etaByPoints(eta)) );
  }
  SigmaPt sigmaPt;
};

#endif
