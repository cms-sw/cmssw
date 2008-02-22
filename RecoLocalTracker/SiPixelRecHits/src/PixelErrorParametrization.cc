
// G. Giurgiu (ggiurgiu@pha.jhu.edu): 01/23/07 - replaced #ifdef DEBUG statements with LogDebug("...")
//                                             - vector<float>& ybarrel_1D = (ybarrel_3D[i_size])[i_alpha];
//                                    03/27/07 - fixed index bug
//                                             - account for big pixels in barrel x errors
//                                    07/03/07 - adapt the code so that it can read the new error parameterization 
//                                               from ../data/residuals.dat produced by CalibTracker/SiPixelErrorEstimation
//                                             - boolean switch "UseNewParametrization" in ../data/PixelCPEParmError.cfi 
//                                               decides between new or old error parameterization  
//                                    07/31/07 - replace range boundary from "<" to "<=" for safety
//                                             - remove cout and assert statements 

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelErrorParametrization.h"

//#include "Utilities/GenUtil/interface/ioutils.h"
//#include "CommonDet/DetUtilities/interface/DetExceptions.h"

//#include "Utilities/General/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

// &&& Do we really need these two?  (Petar)
#include <iterator>
#include <cmath>

using namespace std;
using namespace edm;

const float math_pi = 3.14159265;
const float pitch_x = 0.0100;
const float pitch_y = 0.0150;

// errors for single big pixels
const float error_yb_big_pix = 0.0070;
const float error_xb_big_pix = 0.0030;
const float error_yf_big_pix = 0.0068;
const float error_xf_big_pix = 0.0040;

// alpha ranges for each of the three pixel X-sizes: 1, 2 and >2 
float aa_min[3] = {1.50, 1.45, 1.40};
float aa_max[3] = {1.75, 1.70, 1.65};

bool verbose = false;

//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
PixelErrorParametrization::PixelErrorParametrization(edm::ParameterSet const& conf)
{
  // static SimpleConfigurable<string> paramType("oscar", "CMSsimulation");
  theParametrizationType = 
    conf.getParameter<string>("PixelErrorParametrization");
  
  useNewParametrization =  conf.getParameter<bool>("UseNewParametrization");
  
  useSigma = conf.getParameter<bool>("UseSigma");

  if ( !useNewParametrization )
    {
      if (verbose)
	{
	  LogDebug ("PixelErrorParametrization::PixelErrorParametrization") 
	    << " Using OLD pixel hit error parameterization !!!" ;
	}
      
      ////////////////////////////////////////////////////////
      // define alpha and beta ranges-bins for Y BARREL errors 
      /////////////////////////////////////////////////////////
      
      // MAGIC NUMBERS: alpha bins 
      a_min = 1.37078;
      a_max = 1.77078;
      a_bin = 0.1;
      
      // MAGIC NUMBERS: beta ranges depending on y cluster size
      brange_yb.resize(6);
      brange_yb[0] = pair<float,float>(0., 0.6);   // ysize=1   // Gavril: this only defines 2 ranges 
      brange_yb[1] = pair<float,float>(0.1, 0.9);     // ysize = 2
      brange_yb[2] = pair<float,float>(0.6, 1.05);  // ysize = 3
      brange_yb[3] = pair<float,float>(0.9, 1.15);  // ysize = 4 
      brange_yb[4] = pair<float,float>(1.05, 1.22); // ysize = 5
      brange_yb[5] = pair<float,float>(1.15, 1.41); // ysize >= 6 
      
      // fill Y-BARREL matrix with sigma from gaussian fit 
      // of residuals
      // fill with the resolution points in order 
      // to make an error interpolation 
      
      readYB( ybarrel_3D, "yres_npix", "_alpha", "_b.vec");
      
      // define alpha and beta range/bins for X-BARREL errors
      
      // MAGIC NUMBERS:
      // abs(pi/2-beta) bins depending on X cluster size
      bbins_xb.resize(3);
      // xsize = 1 all beta range
      (bbins_xb[0]).resize(1); 
      (bbins_xb[0])[0] = 100.; 
      // xsize = 2 4 beta-bins
      (bbins_xb[1]).resize(3);
      (bbins_xb[1])[0] = 0.7; 
      (bbins_xb[1])[1] = 1.; 
      (bbins_xb[1])[2] = 1.2; 
      // xsize >= 3 same 4 beta-bins as for xsize=2 // Gavril: checked with Susanna and fixed index from "1" to "2", 03/16/07
      (bbins_xb[2]).resize(3);
      (bbins_xb[2])[0] = 0.7; 
      (bbins_xb[2])[1] = 1.; 
      (bbins_xb[2])[2] = 1.2; 
      
      // fill X-BARREL matrix with parameters to perform a 
      // linear parametrization of x erros
      // for each beta bin: p1 + p2*alpha
      
      readXB( xbarrel_3D, "xpar_npix", "_beta", "_b.vec");
      
      // define alpha and beta range/bins for Y-FORWARD errors
      
      // MAGIC NUMBERS:
      // abs(pi/2-beta) range independent on Y cluster size
      brange_yf = pair<float,float>(0.3, 0.4);     
      
      // fill Y-FORWARD matrix with parameters to perform  
      // a parametrization of Y erros
      // for npix=1 and all alpha range:
      // p1 + p2*beta + p3*beta**2 + p4*beta**3 + p5*beta**4
      // for npix>=2 and all alpha range:
      // p1 + p2*beta + p3*beta**2 
      
      readF( yforward_3D, "ypar_npix", "_alpha", "_f.vec");
      
      // fill X-FORWARD matrix with parameters to perform  
      // a linear parametrization on alpha for all beta range
      
      readF( xforward_3D, "xpar_npix", "_beta", "_f.vec");
    }     
  else
    {
      if (verbose)
	{
	  LogDebug ("PixelErrorParametrization::PixelErrorParametrization") 
	    << " Using NEW pixel hit error parameterization !!!" ;
	}
      
      a_min = 1.37078;
      a_max = 1.77078;
      a_bin = 0.1;
      
      ys_bl[0] = 0.05;
      ys_bh[0] = 0.50;
      
      ys_bl[1] = 0.15; 
      ys_bh[1] = 0.90;
      
      ys_bl[2] = 0.70; 
      ys_bh[2] = 1.05;
      
      ys_bl[3] = 0.95; 
      ys_bh[3] = 1.15;
      
      ys_bl[4] = 1.15; 
      ys_bh[4] = 1.20;
      
      ys_bl[5] = 1.20; 
      ys_bh[5] = 1.40;
      
      edm::FileInPath file( "RecoLocalTracker/SiPixelRecHits/data/residuals.dat" );
      const char* fname = (file.fullPath()).c_str();
      
      FILE* datfile;
      
      if ( (datfile=fopen(fname,"r")) == NULL ) 
	throw cms::Exception("FileNotFound")
	  << "PixelErrorParameterization::PixelErrorParameterization - Input file not found";
      
      vec_error_XB.clear();
      vec_error_YB.clear();
      vec_error_XF.clear();
      vec_error_YF.clear();
      
      while ( !feof(datfile) ) 
	{
	  int detid      = -999;
	  int size       = -999;
	  int angle_ind1 = -999;
	  int angle_ind2 = -999;
	  float sigma    = -999.9;
	  float rms      = -999.9;
	  
	  fscanf( datfile,
		  "%d %d %d %d %f %f \n", 
		  &detid, &size, &angle_ind1, &angle_ind2, &sigma, &rms );
	  
	  float error = -9999.9;
	  if ( useSigma )
	    {
	      if (verbose)
		LogDebug ("PixelErrorParametrization::PixelErrorParametrization") 
		  << " Use error = Gaussian sigma" ;
	      error = sigma;
	    }
	  else 
	    {
	      if (verbose)
		LogDebug ("PixelErrorParametrization::PixelErrorParametrization") 
		  << " Use error = RMS" ;
	      error = rms;
	    }
		  
	  if      ( detid == 1 )
	    vec_error_YB.push_back( error );
	  else if ( detid == 2 )
	    vec_error_XB.push_back( error );
	  else if ( detid == 3 )
	    vec_error_YF.push_back( error );
	  else if ( detid == 4 )
	    vec_error_XF.push_back( error );
	  else 
	    {
	      throw cms::Exception("PixelErrorParametrization::PixelErrorParametrization")
		<< " Wrong ID !!!";
	    }
	}
      
      int n_entries_yb = 240; // number of Y barrel constants to be read from the residuals.dat file
      if ( (int)vec_error_YB.size() != n_entries_yb )
	{
	  throw cms::Exception(" PixelErrorParametrization::PixelErrorParametrization: ") 
	    << " Number of Y barrel constants read different than expected !!!" 
	    << " Expected " << n_entries_yb << " and found " << (int)vec_error_YB.size();
	}
      int n_entries_xb = 120; // number of X barrel constants to be read from the residuals.dat file
      if ( (int)vec_error_XB.size() != n_entries_xb )
	{
	  throw cms::Exception(" PixelErrorParametrization::PixelErrorParametrization: ")
	    << " Number of X barrel constants read different than expected !!!"
	    << " Expected " << n_entries_xb << " and found " << (int)vec_error_XB.size();
	}
      int n_entries_yf = 20; // number of Y forward constants to be read from the residuals.dat file
      if ( (int)vec_error_YF.size() != n_entries_yf )
	{
	  throw cms::Exception(" PixelErrorParametrization::PixelErrorParametrization: ")
	    << " Number of Y forward constants read different than expected !!!"
	    << " Expected " << n_entries_yf << " and found " << (int)vec_error_YF.size();
	}
      int n_entries_xf = 20; // number of X barrel constants to be read from the residuals.dat file
      if ( (int)vec_error_XF.size() != n_entries_xf )
	{
	  throw cms::Exception(" PixelErrorParametrization::PixelErrorParametrization: ")
	    << " Number of X forward constants read different than expected !!!"
	    << " Expected " << n_entries_xf << " and found " << (int)vec_error_XF.size();
	}
  
    }
  
}


//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
PixelErrorParametrization::~PixelErrorParametrization(){}

 
//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
// Gavril: add big pixel info (at this time only bigIn X is used for barrel x errors), 03/27/07
pair<float,float> 
PixelErrorParametrization::getError( GeomDetType::SubDetector pixelPart, 
				     int sizex, int sizey, 
				     float alpha, float beta,
				     bool bigInX, bool bigInY)
{
  pair<float,float> element;
  
  ///
  /// Temporary patch for CMSSW_1_3_0. Handle NANs received from bad tracks
  /// to avoid job crash and return binary errors.
  ///
  if( isnan(alpha) || isnan(beta) ) 
    {
      LogError ("NANcatched") << "PixelErrorParametrization::getError: NAN catched in angles alpha or beta" ; 
      
      element = pair<float,float>(0.010/sqrt(12.), 0.015/sqrt(12.));
      return element;
    }
  
  switch (pixelPart) 
    {
    case GeomDetEnumerators::PixelBarrel:
      element = pair<float,float>(error_XB(sizex, alpha, beta, bigInX), // Gavril: add big pixel flag here. 03/27/07
				  error_YB(sizey, alpha, beta, bigInY));
      break;
    case GeomDetEnumerators::PixelEndcap:
      element =  pair<float,float>(error_XF(sizex, alpha, beta, bigInX),
				   error_YF(sizey, alpha, beta, bigInY));
      break;
    default:
      throw cms::Exception("PixelErrorParametrization::getError") 
	<< "Non-pixel detector type !!!" ;
    }
  
  LogDebug ("PixelErrorParametrization::getError") << " ErrorMatrix gives error: " 
						   << element.first << " , " << element.second;
  
  return element;
}

float PixelErrorParametrization::error_XB(int sizex, float alpha, float beta, bool bigInX)
{
  if ( !useNewParametrization )
    {
      LogDebug("PixelErrorParametrization::error_XB") << "I'M AT THE BEGIN IN ERROR XB METHOD";
      bool barrelPart = true;
      // find size index
      int i_size = min(sizex-1,2);
      
      // find beta index
      int i_beta = betaIndex(i_size, bbins_xb[i_size], beta);
      
      // if ( i_size==0 ) return linParametrize(barrelPart, i_size, i_beta, alpha);
      //else return quadParametrize(barrelPart, i_size, i_beta, alpha);
      
      // Gavril: fix for big pixels at the module center
      //double pitch_x = 0.0100;
      if ( bigInX && sizex == 1 )
	return pitch_x/sqrt(12.0);
      else
	return quadParametrize(barrelPart, i_size, i_beta, alpha);
    }
  else
    {
      float error_xb = -999.9;
      
      /*
	if ( verbose )
	{
	cout << " ---------- 2 ) error_XB: " << endl;
	cout << " sizex = "  << sizex  << endl;
	cout << " alpha = "  << alpha  << endl;
	cout << " beta  = "  << beta   << endl;
	cout << " bigInX = " << bigInX << endl;
	}
      */

      if ( bigInX && sizex == 1 )
	{
	  //error_xb = pitch_x/sqrt(12.0);
	  error_xb = error_xb_big_pix;
	}
      else
	{
	  float alpha_rad = fabs(alpha);
	  //float beta_rad  = fabs(beta);
	  float betap_rad = fabs( math_pi/2.0 - beta );
	  //float alphap_rad = fabs( math_pi/2.0 - alpha );
	  
	  if ( sizex > 3 ) sizex = 3;
	  
	  int ind_sizex = sizex - 1;
	  int ind_beta  = -999;
	  int ind_alpha = -999;
	  
	  if      (                     betap_rad <= 0.7 ) ind_beta = 0;
	  else if ( 0.7 <  betap_rad && betap_rad <= 1.0 ) ind_beta = 1;
	  else if ( 1.0 <  betap_rad && betap_rad <= 1.2 ) ind_beta = 2;
	  else if ( 1.2 <= betap_rad                     ) ind_beta = 3;
	  else 
	    {
	      throw cms::Exception("PixelErrorParametrization::error_XB") << " Wrong betap_rad = " << betap_rad;
	    }

	  if      ( alpha_rad <= aa_min[ind_sizex] ) ind_alpha = 0;
	  else if ( alpha_rad >= aa_max[ind_sizex] ) ind_alpha = 9;
	  else
	    ind_alpha = (int) ( ( alpha_rad - aa_min[ind_sizex] ) / ( ( aa_max[ind_sizex] - aa_min[ind_sizex] ) / 10.0 ) );  

	  /*
	    if ( verbose )
	    {
	    cout << "ind_sizex = " << ind_sizex << endl;
	    cout << "ind_alpha = " << ind_alpha << endl;
	    cout << "ind_beta  = " << ind_beta  << endl;
	    }
	  */

	  // There are 4 beta bins with 10 alpha bins  for each sizex
	  int index = 40*ind_sizex + 10*ind_beta + ind_alpha;
	  
	  if ( index < 0 || index >= 120  )
	    {
	      throw cms::Exception(" PixelErrorParametrization::error_XB") << " Wrong index !!!";
	    }
	  
	  error_xb = vec_error_XB[index];
	  
	}
      
      //if ( verbose )
      //cout << "error_xb = " << error_xb << endl;
	
      return error_xb;
    }
  
}


float PixelErrorParametrization::error_XF(int sizex, float alpha, float beta, bool bigInX)
{
  if ( !useNewParametrization )
    {
      LogDebug("PixelErrorParametrization::error_XF") << "I'M AT THE BEGIN IN ERROR XF METHOD";
      
      bool barrelPart = false;
      //symmetrization w.r.t. orthogonal direction
      float alpha_prime = fabs(3.14159/2.-alpha);
      // find x size index
      int i_size = min(sizex-1,2);
      // no beta parametrization!!!
      int i_beta = 0;
      // find beta index
      // int i_beta = betaIndex(i_size, bbins_xf, beta);
      LogDebug("PixelErrorParametrization::error_XF") << "size index = " << i_size
						      << "no beta index, "
						      << " alphaprime = " << alpha_prime;
      //double pitch_x = 0.0100;
      if ( bigInX && sizex == 1 )
	return pitch_x/sqrt(12.0);
      else
	return linParametrize(barrelPart, i_size, i_beta, alpha_prime);
    }
  else
    {
      float error_xf = -999.9;
      
      /*
	if ( verbose )
	{
	cout << " ---------- 4 ) error_XF:" << endl;
	cout << " sizex = "  << sizex  << endl;
	cout << " alpha = "  << alpha  << endl;
	cout << " beta  = "  << beta   << endl;
	cout << " bigInX = " << bigInX << endl;
	}
      */

      if ( bigInX && sizex == 1 )
	{
	  //error_xf = pitch_x/sqrt(12.0);
	  error_xf = error_xf_big_pix;
	}
      else
	{
	  //float alpha_rad = fabs(alpha);
	  //float beta_rad  = fabs(beta);
	  //float betap_rad = fabs( math_pi/2.0 - beta );
	  float alphap_rad = fabs( math_pi/2.0 - alpha );
	  
	  if ( sizex > 2 ) sizex = 2;
	  
	  int ind_sizex = sizex - 1;
	  int ind_alpha  = -9999; 
	  
	  if ( sizex == 1 )
	    {
	      if      ( alphap_rad <= 0.15 ) ind_alpha = 0;
	      else if ( alphap_rad >= 0.30 ) ind_alpha = 9;
	      else 
		ind_alpha = (int) ( ( alphap_rad - 0.15 ) / ( ( 0.30 - 0.15 ) / 10.0 ) );  
	    }
	  if ( sizex > 1 )
	    {
	      if      ( alphap_rad <= 0.15 ) ind_alpha = 0;
	      else if ( alphap_rad >= 0.50 ) ind_alpha = 9;
	      else 
		ind_alpha = (int) ( ( alphap_rad - 0.15 ) / ( ( 0.50 - 0.15 ) / 10.0 ) );  
	    }
	  
	  /*
	    if ( verbose )
	    {
	    cout << "ind_sizex = " << ind_sizex << endl;
	    cout << "ind_alpha = " << ind_alpha << endl;
	    }
	  */

	  int index = 10*ind_sizex + ind_alpha;
	  
	  if ( index < 0 || index >= 20  )
	    {
	      throw cms::Exception(" PixelErrorParametrization::error_XF") << " Wrong index !!!";
	    }
	  
	  error_xf = vec_error_XF[index];
	  
	}
      
      //if ( verbose )
      //cout << "error_xf = " << error_xf << endl;
      
      return error_xf;
    }
  
}


float PixelErrorParametrization::error_YB(int sizey, float alpha, float beta, bool bigInY)
{  
  if ( !useNewParametrization )
    {
      LogDebug("PixelErrorParametrization::error_YB") << "I'M AT THE BEGIN IN ERROR YB METHOD";
      
      //double pitch_y = 0.0150;
      
      if ( bigInY && sizey == 1 )
	{
	  return pitch_y/sqrt(12.0);
	}
      else
	{
	  int i_alpha;
	  int i_size = min(sizey-1,5);
	  
	  LogDebug("PixelErrorParametrization::error_YB") << "I found size index = " << i_size;
	  
	  if (sizey < 4) 
	    {      // 3 alpha bins
	      if (alpha <= a_min + a_bin) 
		{ 
		  i_alpha = 0;
		} 
	      else if (alpha < a_max-a_bin) 
		{
		  i_alpha = 1;
		}
	      else 
		{
		  i_alpha = 2; 
		}
	    }
	  else
	    { // 1 alpha bin 
	      i_alpha = 0;
	    }
	  
	  LogDebug("PixelErrorParametrization::error_YB") << "I found alpha index = " << i_alpha;
	  
	  // vector of beta parametrization
	  //vector<float> ybarrel_1D = (ybarrel_3D[i_size])[i_alpha];
	  vector<float>& ybarrel_1D = (ybarrel_3D[i_size])[i_alpha]; // suggestion to speed up the code by Patrick/Vincenzo
	  
	  LogDebug("PixelErrorParametrization::error_YB") << " beta vec has dimensions = " << ybarrel_1D.size()
							  << " beta = " << beta 
							  << " beta max = " << brange_yb[i_size].second 
							  << " beta min = " << brange_yb[i_size].first;
	  
	  // beta --> abs(pi/2-beta) to be symmetric w.r.t. pi/2 axis
	  float beta_prime = fabs(3.14159/2.-beta);
	  
	  if ( beta_prime <= brange_yb[i_size].first )// Gavril: brange_yb[0].first == 0.0; when i_size==0, beta_prime is never less than 0
	    { 
	      return ybarrel_1D[0];
	    }
	  else if ( beta_prime >= brange_yb[i_size].second )
	    {
	      //return ybarrel_1D[ybarrel_1D.size()-1];
	      return pitch_y / sqrt(12.0); // Gavril: we are in un-physical beta_prime range; return large error, 03/27/07 
	    } 
	  else 
	    {
	      return interpolation(ybarrel_1D, beta_prime, brange_yb[i_size] );
	    }  
	}
    }
  else
    {
      float error_yb = -999.9;
      
      /*
	if ( verbose )
	{
	cout << " ---------- 1 ) error_YB:" << endl;
	cout << " sizey = "  << sizey  << endl;
	cout << " alpha = "  << alpha  << endl;
	cout << " beta  = "  << beta   << endl;
	cout << " bigInY = " << bigInY << endl; 
	}
      */

      if ( bigInY && sizey == 1 )
	{
	  //error_yb = pitch_y/sqrt(12.0);
	  error_yb = error_yb_big_pix;
	}
      else
	{
	  float alpha_rad = fabs(alpha);
	  //float beta_rad  = fabs(beta);
	  float betap_rad = fabs( math_pi/2.0 - beta );
	  //float alphap_rad = fabs( math_pi/2.0 - alpha );
	  
	  if ( sizey > 6 ) sizey = 6;
	  
	  int ind_sizey = sizey - 1;
	  int ind_alpha = -9999;
	  int ind_beta  = -9999; 
	  
	  if      ( alpha_rad <= a_min ) ind_alpha = 0;
	  else if ( alpha_rad >= a_max ) ind_alpha = 3;
	  else if ( alpha_rad > a_min && 
		    alpha_rad < a_max ) 
	    {
	      double binw = ( a_max - a_min ) / 4.0;
	      ind_alpha = (int)( ( alpha_rad - a_min ) / binw );
	    }		
	  else
	    {
	      throw cms::Exception(" PixelErrorParametrization::error_YB") << " Wrong alpha_rad = " << alpha_rad;
	      
	    }

	  if      ( betap_rad <= ys_bl[sizey-1] ) ind_beta = 0;
	  else if ( betap_rad >= ys_bh[sizey-1] ) ind_beta = 9;
	  else if ( betap_rad >  ys_bl[sizey-1] && 
		    betap_rad <  ys_bh[sizey-1] ) 
	    {
	      double binw = ( ys_bh[sizey-1] - ys_bl[sizey-1] ) / 8.0;
	      ind_beta = 1 + (int)( ( betap_rad - ys_bl[sizey-1] ) / binw );
	    }		
	  else 
	    {
	      throw cms::Exception(" PixelErrorParametrization::error_YB") << " Wrong betap_rad = " << betap_rad;
	    }

	  /*
	    if ( verbose )
	    {
	    cout << "ind_sizey = " << ind_sizey << endl;
	    cout << "ind_alpha = " << ind_alpha << endl;
	    cout << "ind_beta  = " << ind_beta  << endl;
	    }
	  */

	  int index = 40*ind_sizey + 10*ind_alpha + ind_beta;
	  
	  if ( index < 0 || index >= 240  )
	    {
	      throw cms::Exception(" PixelErrorParametrization::error_YB") << " Wrong index !!!";
	    }
	  
	  error_yb = vec_error_YB[index];
	  
	}
      
      //if ( verbose )
      //cout << "error_yb = " << error_yb << endl;
      
      return error_yb; 
    }

}


float PixelErrorParametrization::error_YF(int sizey, float alpha, float beta, bool bigInY)
{
  if ( !useNewParametrization )
    {
      LogDebug("PixelErrorParametrization::error_YF") << "I'M AT THE BEGIN IN ERROR YF METHOD";
      
      float err_par = 0.0;
      //double pitch_y = 0.0150;
      
      if ( bigInY && sizey == 1 )
	{
	  err_par = pitch_y/sqrt(12.0);
	}
      else
	{
	  // find y size index
	  int i_size = min(sizey-1,1);
	  // no parametrization in alpha
	  int i_alpha = 0;
	  // beta --> abs(pi/2-beta) to be symmetric w.r.t. pi/2 axis
	  float beta_prime = fabs(3.14159/2.-beta);
	  if (beta_prime < brange_yf.first) beta_prime = brange_yf.first;
	  if (beta_prime > brange_yf.second) beta_prime = brange_yf.second;
	  err_par = 0.0;
	  for(int ii=0; ii < (int)( (yforward_3D[i_size])[i_alpha] ).size(); ii++){
	    err_par += ( (yforward_3D[i_size])[i_alpha] )[ii] * pow(beta_prime,ii);
	  }
	}
      
      return err_par; 
    }
  else
    {
      float error_yf = -999.9;
      
      /*
	if ( verbose )
	{
	cout << " ---------- 3 ) error_YF:" << endl;
	cout << " sizey = "  << sizey  << endl;
	cout << " alpha = "  << alpha  << endl;
	cout << " beta  = "  << beta   << endl;
	cout << " bigInY = " << bigInY << endl;
	}
      */

      if ( bigInY && sizey == 1 )
	{
	  //error_yf = pitch_y/sqrt(12.0);
	  error_yf = error_yf_big_pix;
	}
      else
	{
	  //float alpha_rad = fabs(alpha);
	  //float beta_rad  = fabs(beta);
	  float betap_rad = fabs( math_pi/2.0 - beta );
	  //float alphap_rad = fabs( math_pi/2.0 - alpha );
	  
	  if ( sizey > 2 ) sizey = 2;
	  
	  int ind_sizey = sizey - 1;
	  int ind_beta  = -9999; 
	  
	  if      ( betap_rad <= 0.3 ) ind_beta = 0;
	  else if ( betap_rad >= 0.4 ) ind_beta = 9;
	  else 
	    ind_beta = (int) ( ( betap_rad - 0.3 ) / ( ( 0.4 - 0.3 ) / 10.0 ) );  
	  
	  /*
	    if ( verbose )
	    {
	    cout << "ind_sizey = " << ind_sizey << endl;
	    cout << "ind_beta  = " << ind_beta  << endl;
	    }
	  */

	  int index = 10*ind_sizey + ind_beta;
	  
	  if ( index < 0 || index >= 20  )
	    {
	      throw cms::Exception(" PixelErrorParametrization::error_YF") << " Wrong index !!!";
	    }
	  
	  error_yf = vec_error_YF[index];
	  
	}
      
      //if ( verbose )
      //cout << "error_yf = " << error_yf << endl;
      
      return error_yf; 
    }
    
}


//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
float PixelErrorParametrization::interpolation(vector<float>& vector_1D, 
					       float& angle, pair<float,float>& range)
{

  if (angle < range.first) 
    LogDebug("PixelErrorParametrization::interpolation") << " IT IS NOT NECESSARY TO DO AN INTERPOLATION";

  float bin = (range.second-range.first)/float(vector_1D.size());
  float curr = range.first;
  int i_bin = -1;
  for(int i = 0; i< (int)vector_1D.size(); i++){
    //cout<< "index = "<< i << endl;
    curr += bin;
    if (angle <= curr){
      i_bin = i;
      break;
    }
  }
  // cout << " ibin= " << i_bin << " value= " << curr <<endl;
  float y1;
  float y2;
  float x1;
  float x2;
  // test_1
  if(i_bin == 0){
    x1 = range.first + bin/2;
    x2 = x1 + bin;
    y1 = vector_1D[0];
    y2 = vector_1D[1];
  }else if ( i_bin == (int)(vector_1D.size()-1) || i_bin == -1){
    x2 = range.second - bin/2;
    x1 = x2 - bin;
    y2 = vector_1D[(vector_1D.size()-1)]; 
    y1 = vector_1D[(vector_1D.size()-2)]; 
  } else if ( (curr-angle) < bin/2){
    x1 = curr - bin/2;
    x2 = x1 + bin;
    y1 = vector_1D[i_bin];
    y2 = vector_1D[i_bin+1];
  } else {
    x2 = curr - bin/2;
    x1 = x2 - bin;
    y2 = vector_1D[i_bin];
    y1 = vector_1D[i_bin-1];    
  }
  float y = ((y2-y1)*angle + (y1*x2-y2*x1))/(x2-x1);
  //end test1

  LogDebug("PixelErrorParametrization::interpolation") << "INTERPOLATION GIVES" 
						       << "  (x1,y1) = " << x1    << " , " << y1
						       << "  (x2,y2) = " << x2    << " , " << y2
						       << "(angle,y) = " << angle << " , " << y;
  
  return y;
}



//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
int PixelErrorParametrization::betaIndex(int& i_size, vector<float>& betabins, 
					 float& beta)
{
  int i_beta = -1;
  // beta --> abs(pi/2-beta) to be symmetric w.r.t. pi/2 axis
  float beta_prime = fabs(3.14159/2.-beta);
  for(int i = 0; i < (int)betabins.size(); i++){
    if (beta_prime < betabins[i]) {
      i_beta = i;
      break;
    } 
  }
  if (i_beta < 0) i_beta = betabins.size();
  return i_beta;
}





//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
float PixelErrorParametrization::linParametrize(bool& barrelPart, int& i_size, 
						int& i_angle1, float& angle2)
{
  LogDebug("PixelErrorParametrization::linParametrize") << "We are in linParametrize metod"
							<< "barrel? " << barrelPart
							<< "size index = " << i_size
							<< "angle index = " << i_angle1
							<< "angle variable = " << angle2;
  
  float par0 = 0;
  float par1 = 0;
  if (barrelPart) {
    par0 = ((xbarrel_3D[i_size])[i_angle1])[0]; 
    par1 = ((xbarrel_3D[i_size])[i_angle1])[1]; 
  } else {
    par0 = ((xforward_3D[i_size])[i_angle1])[0]; 
    par1 = ((xforward_3D[i_size])[i_angle1])[1]; 
  }

  LogDebug("PixelErrorParametrization::linParametrize") << "PAR0 = " << par0
							<< "PAR1 = " << par1
							<< "PAR1 = " << angle2
							<< "X error = " << (par0 + par1*angle2);
  
  return par0 + par1*angle2;
}





//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
float PixelErrorParametrization::quadParametrize(bool& barrelPart, int& i_size, 
						 int& i_angle1, float& angle2)
{
  LogDebug("PixelErrorParametrization::quadParametrize") << "barrel? " << barrelPart
							 << "size index = " << i_size
							 << "angle index = " << i_angle1
							 << "angle variable = " << angle2;
  
  float par0 = 0;
  float par1 = 0;
  float par2 = 0;
  if (barrelPart) {
    par0 = ((xbarrel_3D[i_size])[i_angle1])[0]; 
    par1 = ((xbarrel_3D[i_size])[i_angle1])[1]; 
    par2 = ((xbarrel_3D[i_size])[i_angle1])[2]; 
  } else {
    par0 = ((xforward_3D[i_size])[i_angle1])[0]; 
    par1 = ((xforward_3D[i_size])[i_angle1])[1]; 
    par2 = ((xforward_3D[i_size])[i_angle1])[2]; 
  }

  LogDebug("PixelErrorParametrization::quadParametrize") << "PAR0 = " << par0
							 << "PAR1 = " << par1
							 << "PAR2 = " << par1
							 << "ANGLE = " << angle2
							 << "X error = " << (par0 + par1*angle2 + par2*angle2*angle2);
  
  return par0 + par1*angle2 + par2*angle2*angle2;
}




//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
void PixelErrorParametrization::readYB( P3D& vec3D, const string& prefix, 
					const string& postfix1, const string& postfix2)
{
  vec3D.clear();

  // for npixy= 1 to 3 => 3 alpha bins
  // for npixy = 4 to 6 => 1 alpha bin
  for(int i_pix = 1; i_pix <7; i_pix++ ) {

    P2D tmp2D;
    if(i_pix < 4){
      for (int i_abin = 1; i_abin < 4; i_abin++) {
	ostringstream filename;
	filename << prefix << i_pix << postfix1 << i_abin << postfix2;
	//&&& string filename = 
	//&&&   prefix + toa()(i_pix) + postfix1 + toa()(i_abin) + postfix2;
	tmp2D.push_back( readVec( filename.str() ));
      }// loop alpha bins
    } else { // only 1 alpha bin
      ostringstream filename;
      filename << prefix << i_pix << postfix1 << '0' << postfix2;
      //&&& string filename = prefix + toa()(i_pix) + postfix1 + "0" + postfix2;
      tmp2D.push_back( readVec( filename.str() ));
    }// if npixel < 4
    vec3D.push_back(tmp2D);
  }//loop npixy
}  



//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
void PixelErrorParametrization::readXB( P3D& vec3D, const string& prefix, 
					const string& postfix1, const string& postfix2)
{
  vec3D.clear();

  for(int i_pix = 1; i_pix <4; i_pix++ ) {
    P2D tmp2D;
    if ( i_pix == 1 ){ //all beta range
      //      int i_bbin = 0;
      //&&& string filename = 
      //&&& 	prefix + toa()(i_pix) + postfix1 + "0" + postfix2;
      ostringstream filename;
      filename << prefix << i_pix << postfix1 << '0' << postfix2;
      tmp2D.push_back( readVec( filename.str() ));
    } else {
      for (int i_bbin = 1; i_bbin < 5; i_bbin++) {
	//&&& string filename = 
	//&&&   prefix + toa()(i_pix) + postfix1 + toa()(i_bbin) + postfix2;
	ostringstream filename;
	filename << prefix << i_pix << postfix1 << i_bbin << postfix2;
	tmp2D.push_back( readVec( filename.str() ));
      }
    }
    vec3D.push_back(tmp2D);
  }
}



//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
void PixelErrorParametrization::readF( P3D& vec3D, const string& prefix, 
				       const string& postfix1, const string& postfix2)
{
  vec3D.clear();
  int maxPix = 3;
  if ( prefix == "xpar_npix" ) maxPix=4;
  for(int i_pix = 1; i_pix <maxPix; i_pix++ ) {
    P2D tmp2D;

    //&&& string filename = prefix + toa()(i_pix) + postfix1 + "0" + postfix2;
    ostringstream filename;
    filename << prefix << i_pix << postfix1 << '0' << postfix2;

    tmp2D.push_back(readVec( filename.str() ));
    vec3D.push_back(tmp2D);
  }
}





//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
vector<float> PixelErrorParametrization::readVec( const string& name) 
{
  string partialName = "RecoLocalTracker/SiPixelRecHits/data/";
  if (theParametrizationType == "cmsim") {
    partialName += theParametrizationType + "/";
  }
  string fullName =  partialName + name;
  edm::FileInPath f1( fullName );
  ifstream invec( (f1.fullPath()).c_str() );

  vector<float> result;
  copy(istream_iterator<float>(invec), 
	    istream_iterator<float>(), back_inserter(result));

  return result;
}

