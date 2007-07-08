/*
  Performs a 3-dimensional parametrization of pixel hits
  as a funcion of the cluster size and the two projections
  of the impact track angle 
  !!! it is valid for 100 X 150 mu pixel size !!!
*/

#ifndef RecoLocalTracker_SiPixelRecHits_PixelErrorParametrization_H
#define RecoLocalTracker_SiPixelRecHits_PixelErrorParametrization_H 1

//--- For the type of DetUnit
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

//--- For the configuration:
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <utility>
#include <vector>
#include <string>
#include <iosfwd>


class PixelErrorParametrization
{
 public:

  PixelErrorParametrization(edm::ParameterSet const& conf);
  ~PixelErrorParametrization();
  
  /// Returns error (sigma) for local x and y coordinates
  std::pair<float,float>
    getError(GeomDetType::SubDetector pixelPart, 
	     int sizex, int sizey, 
	     float alpha, float beta,
	     bool bigInX = false, 
	     bool bigInY = false); // Gavril: add big pixel flags, 03/27/07
  
 private:
  float a_min; 
  float a_max;
  float a_bin; 

  float ys_bl[6];
  float ys_bh[6];
  
  std::vector<float> vec_error_YB;
  std::vector<float> vec_error_XB;
  std::vector<float> vec_error_YF;
  std::vector<float> vec_error_XF;


  std::string theParametrizationType;
  typedef std::vector<std::vector<std::vector<float> > > P3D; 
  typedef std::vector<std::vector<float> > P2D;

  // If this is set true, then use the parameterization from CalibTracker/SiPixelErrorEstimation
  // If it is set to false, use old parameterization by Susanna Cuciareli
  bool useNewParametrization;
  
  // If  useNewParametrization = true, then useSigma = true assigns gaussian errors. Otherwise asign RMS errors 
  bool useSigma;

  // MATRIX FOR Y BARREL ERRORS
  //
  // vector depending on cluster size:
  P3D ybarrel_3D;

  //vector containing beta range
  std::vector< std::pair<float,float> > brange_yb;
  //
  // MATRIX FOR X BARREL ERRORS
  //
  // vector depending cluster size
  P3D xbarrel_3D;

  // 2D vector containing beta bins 
  // for X-BARREL depending on x cluster size
  P2D bbins_xb;
  //
  // MATRIX FOR Y FORWARD ERRORS
  //
  // vector depending cluster size
  P3D yforward_3D;

  // pair containing abs(pi/2-beta) range 
  // for Y-FORWARD
  std::pair<float,float> brange_yf;

  // MATRIX FOR X FORWARD ERRORS
  //
  // vector depending cluster size
  P3D xforward_3D;

  // 1D vector containing beta bins 
  // for X-FORWARD
  std::vector<float> bbins_xf;

  // Gavril: for barrel x errors, add big pixel flag  
  float error_XB(int sizex, float alpha, float beta, bool bigInX); 
  float error_XF(int sizex, float alpha, float beta, bool bigInX);
  float error_YB(int sizey, float alpha, float beta, bool bigInY);
  float error_YF(int sizey, float alpha, float beta, bool bigInY);
  float interpolation(std::vector<float>&, float&, 
		      std::pair<float,float>&);
  int betaIndex(int&, std::vector<float>&, float&);
  float linParametrize(bool&, int&, int&, float&);  

  float quadParametrize(bool&, int&, int&, float&);  

  void readXB( P3D& vec3D, const std::string& prefix, 
	       const std::string& postfix1, const std::string& postfix2);

  void readYB( P3D& vec3D, const std::string& prefix, 
	       const std::string& postfix1, const std::string& postfix2);

  void readF( P3D& vec3D, const std::string& prefix, 
	      const std::string& postfix1, const std::string& postfix2);

  std::vector<float> readVec( const std::string& name);
};
#endif
