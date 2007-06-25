
// G. Giurgiu (ggiurgiu@pha.jhu.edu): 01/23/07 - replaced #ifdef DEBUG statements with LogDebug("...")
//                                             - vector<float>& ybarrel_1D = (ybarrel_3D[i_size])[i_alpha];

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelErrorParametrization.h"

//#include "Utilities/GenUtil/interface/ioutils.h"
//#include "CommonDet/DetUtilities/interface/DetExceptions.h"

//#include "Utilities/General/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

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

//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
PixelErrorParametrization::PixelErrorParametrization(edm::ParameterSet const& conf)
{
  // static SimpleConfigurable<string> paramType("oscar", "CMSsimulation");
  theParametrizationType = 
    conf.getParameter<string>("PixelErrorParametrization");

  ////////////////////////////////////////////////////////
  // define alpha and beta ranges-bins for Y BARREL errors 
  /////////////////////////////////////////////////////////
  
  // MAGIC NUMBERS: alpha bins 
  a_min = 1.37078;
  a_max = 1.77078;
  a_bin = 0.1;

  // MAGIC NUMBERS: beta ranges depending on y cluster size
  brange_yb.resize(6);
  brange_yb[0] = pair<float,float>(0., 0.6);   // ysize=1
  brange_yb[1] = pair<float,float>(0.1, 0.9);     // ysize = 2
  brange_yb[2] = pair<float,float>(0.6, 1.05);  // ysize = 3
  brange_yb[3] = pair<float,float>(0.9, 1.15);  // ysize = 4 
  brange_yb[4] = pair<float,float>(1.05, 1.22); // ysize = 5
  brange_yb[5] = pair<float,float>(1.15, 1.41); // ysize >= 6 

  /////////////////////////////////////////////////////
  // fill Y-BARREL matrix with sigma from gaussian fit 
  // of residuals
  /////////////////////////////////////////////////////
  // fill with the resolution points in order 
  // to make an error interpolation 

  readYB( ybarrel_3D, "yres_npix", "_alpha", "_b.vec");

  ///////////////////////////////////////////////////////
  // define alpha and beta range/bins for X-BARREL errors
  ///////////////////////////////////////////////////////
  
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
  // xsize >= 3 same 4 beta-bins as for xsize=2
  (bbins_xb[1]).resize(3);
  (bbins_xb[1])[0] = 0.7; 
  (bbins_xb[1])[1] = 1.; 
  (bbins_xb[1])[2] = 1.2; 

  ///////////////////////////////////////////////////////
  // fill X-BARREL matrix with parameters to perform a 
  // linear parametrization of x erros
  ///////////////////////////////////////////////////////
  // for each beta bin: p1 + p2*alpha

  readXB( xbarrel_3D, "xpar_npix", "_beta", "_b.vec");

  ///////////////////////////////////////////////////////
  // define alpha and beta range/bins for Y-FORWARD errors
  ///////////////////////////////////////////////////////

  // MAGIC NUMBERS:
  // abs(pi/2-beta) range independent on Y cluster size
  brange_yf = pair<float,float>(0.3, 0.4);     

  //////////////////////////////////////////////////////
  // fill Y-FORWARD matrix with parameters to perform  
  // a parametrization of Y erros
  //////////////////////////////////////////////////////
  // for npix=1 and all alpha range:
  // p1 + p2*beta + p3*beta**2 + p4*beta**3 + p5*beta**4
  // for npix>=2 and all alpha range:
  // p1 + p2*beta + p3*beta**2 

  readF( yforward_3D, "ypar_npix", "_alpha", "_f.vec");

  //////////////////////////////////////////////////////
  // fill X-FORWARD matrix with parameters to perform  
  // a linear parametrization on alpha for all beta range
  //////////////////////////////////////////////////////

  readF( xforward_3D, "xpar_npix", "_beta", "_f.vec");
}     


//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
PixelErrorParametrization::~PixelErrorParametrization(){}

 
//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
pair<float,float> 
PixelErrorParametrization::getError(GeomDetType::SubDetector pixelPart, 
				    int sizex, int sizey, 
				    float alpha, float beta)
{
  pair<float,float> element;

  ///
  /// Temporary patch for CMSSW_1_3_0. Handle NANs received from bad tracks
  /// to avoid job crash and return binary errors.
  ///
  if( isnan(alpha) || isnan(beta) ) {

    LogError ("NANcatched") << "PixelErrorParametrization::getError: NAN catched in angles alpha or beta" ; 
 
    element = pair<float,float>(0.010/sqrt(12.), 0.015/sqrt(12.));
    return element;

  }
  
  switch (pixelPart) {
  case GeomDetEnumerators::PixelBarrel:
    element = pair<float,float>(error_XB(sizex, alpha, beta), 
				error_YB(sizey, alpha, beta));
    break;
  case GeomDetEnumerators::PixelEndcap:
    element =  pair<float,float>(error_XF(sizex, alpha, beta),
				 error_YF(sizey, alpha, beta));
    break;
  default:
    LogDebug ("PixelErrorParametrization::getError") 
      << "PixelErrorParametrization:: a non-pixel detector type in here?" ;
    //  &&& Should throw an exception here!
    assert(0);
  }

  LogDebug ("PixelErrorParametrization::getError") << " ErrorMatrix gives error: " 
						  << element.first << " , " << element.second;
  
  return element;
}




//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
float PixelErrorParametrization::error_XB(int sizex, float alpha, float beta)
{
  LogDebug("PixelErrorParametrization::error_XB") << "I'M AT THE BEGIN IN ERROR XB METHOD";
  bool barrelPart = true;
 // find size index
  int i_size = min(sizex-1,2);

  // find beta index
  int i_beta = betaIndex(i_size, bbins_xb[i_size], beta);

  // if ( i_size==0 ) return linParametrize(barrelPart, i_size, i_beta, alpha);
  //else return quadParametrize(barrelPart, i_size, i_beta, alpha);
  return quadParametrize(barrelPart, i_size, i_beta, alpha);
}



//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
float PixelErrorParametrization::error_XF(int sizex, float alpha, float beta)
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
  
  return linParametrize(barrelPart, i_size, i_beta, alpha_prime);
}



//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
float PixelErrorParametrization::error_YB(int sizey, float alpha, float beta)
{  
  LogDebug("PixelErrorParametrization::error_YB") << "I'M AT THE BEGIN IN ERROR YB METHOD";
  
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
  if ( beta_prime <= brange_yb[i_size].first )
    { 
      return ybarrel_1D[0];
    }
  else if ( beta_prime >= brange_yb[i_size].second )
    {
      return ybarrel_1D[ybarrel_1D.size()-1];
    } 
  else 
    {
      return interpolation(ybarrel_1D, beta_prime, brange_yb[i_size] );
    }  
}

//-----------------------------------------------------------------------------
//  
//-----------------------------------------------------------------------------
float PixelErrorParametrization::error_YF(int sizey, float alpha, float beta)
{
  LogDebug("PixelErrorParametrization::error_YF") << "I'M AT THE BEGIN IN ERROR YF METHOD";

  // find y size index
  int i_size = min(sizey-1,1);
  // no parametrization in alpha
  int i_alpha = 0;
  // beta --> abs(pi/2-beta) to be symmetric w.r.t. pi/2 axis
  float beta_prime = fabs(3.14159/2.-beta);
  if (beta_prime < brange_yf.first) beta_prime = brange_yf.first;
  if (beta_prime > brange_yf.second) beta_prime = brange_yf.second;
  float err_par = 0;
  for(int ii=0; ii < (int)( (yforward_3D[i_size])[i_alpha] ).size(); ii++){
    err_par += ( (yforward_3D[i_size])[i_alpha] )[ii] * pow(beta_prime,ii);
  }
  return err_par; 
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

