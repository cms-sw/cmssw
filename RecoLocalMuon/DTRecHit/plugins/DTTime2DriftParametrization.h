#ifndef DTTime2DriftParametrization_H
#define DTTime2DriftParametrization_H

/** \class DTTime2DriftParametrization
 *  This class access the cell parametrization by Pablo Garcia-Abia and Jesus Puerta (CIEMAT).
 *  The two public methods act as an interface for the parametrization converting the reference frame
 *  and the units accordingly to CMSSW convention.
 *
 *  $Date: 2007/04/19 11:08:17 $
 *  $Revision: 1.1 $
 *  \author  Pablo Garcia-Abia and Jesus Puerta (CIEMAT, Madrid)
 */



#define N_alpha     11
#define N_By         5
#define N_Bz         5
#define N_Par_x     15
#define N_Sigma_t    7


class DTTime2DriftParametrization {
public:
  /// Constructor
  DTTime2DriftParametrization();

  /// Destructor
  virtual ~DTTime2DriftParametrization();

  /// Structure used to return output values
  typedef struct {
    double v_drift, x_drift, delta_x, x_width_m, x_width_p ;
  } drift_distance; //FIXME: Why this typedef?
  
  
  /// Compute the drift distance and relative errors (cm).
    /// The drift distance is computed as the MODE value of the distance distribution
    /// for this particular time.
    /// This method takes care of the conversions of units and reference frame used by CMSSW and the 
    /// parametrization.
    /// time is in ns, alpha (=atan(x/-z)) in rad, by and bz are the magnetic field in the layer RF.
    /// False is returned if the parametrization fails
  bool computeDriftDistance_mode(double time,
				 double alpha,
				 double by,
				 double bz,
				 short interpolate,
				 drift_distance* dx) const;

  /// Compute the drift distance and relative errors (cm).
  /// The drift distance is computed as the MEAN value of the distance distribution
  /// for this particular time.
  /// This method takes care of the conversions of units and reference frame used by CMSSW and the 
  /// parametrization.
  /// time is in ns, alpha (=atan(x/-z))  in rad, by and bz are the magnetic field in the layer RF.
  /// False is returned if the parametrization fails
  bool computeDriftDistance_mean(double time,
				 double alpha,
				 double by,
				 double bz,
				 short interpolate,
				 drift_distance* dx) const;


private:
  /// Calculate drift distance and spread
  unsigned short MB_DT_drift_distance (double time,
				       double alpha,
				       double by,
				       double bz,
				       drift_distance *DX,
				       short interpolate) const;


  double MB_DT_dist      (double , double *) const;
  double MB_DT_delta_x   (double , double *) const;
  double MB_DT_sigma_t_m (double , double *) const;
  double MB_DT_sigma_t_p (double , double *) const;

  unsigned short MB_DT_Check_boundaries (double distime,
					 double alpha,
					 double by,
					 double bz,
					 short ifl) const;

  void MB_DT_Get_grid_values (double Var,
			      unsigned short *pi,
			      unsigned short *pj,
			      short Initial,
			      unsigned short N,
			      const double *Values) const;
  
  void MB_DT_Get_grid_points (double alpha, 
			      double by,
			      double bz,
			      unsigned short *p_alpha,
			      unsigned short *p_By,
			      unsigned short *p_Bz,
			      unsigned short *q_alpha,
			      unsigned short *q_By,
			      unsigned short *q_Bz) const;
  
  double MB_DT_MLInterpolation (double *al,
				double *by,
				double *bz,
				double *f) const;
  
  /*** Points of the grid ***/
  static const double alpha_value[N_alpha];
  static const double    By_value[N_By];
  static const double    Bz_value[N_Bz];

  /*** Parameter values ***/
  static const double fun_x[N_alpha][N_By][N_Bz][N_Par_x];  
  static const double fun_sigma_t[N_alpha][N_By][N_Bz][N_Sigma_t];

  /*** skew factor ***/
  static const double velocitySkew;

};
#endif
