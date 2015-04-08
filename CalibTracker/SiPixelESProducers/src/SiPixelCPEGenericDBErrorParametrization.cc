#include "CalibTracker/SiPixelESProducers/interface/SiPixelCPEGenericDBErrorParametrization.h"
#include "CondFormats/DataRecord/interface/SiPixelCPEGenericErrorParmRcd.h"
#include <iostream>
#include <cmath>

const float math_pi = 3.14159265;

//These are the bin parameters -- they determine the width of the bins
//998, 999 refer to bins where the 0 should always be returned
const float SiPixelCPEGenericDBErrorParametrization::bx_a_min[3] = {1.525, 1.475, 1.425};
const float SiPixelCPEGenericDBErrorParametrization::bx_a_max[3] = {1.725, 1.675, 1.625};

const float SiPixelCPEGenericDBErrorParametrization::fx_a_min[2] = {0.165, 0.185};
const float SiPixelCPEGenericDBErrorParametrization::fx_a_max[2] = {0.285, 0.465};
const float SiPixelCPEGenericDBErrorParametrization::fx_b_min[2] = {998, 998};
const float SiPixelCPEGenericDBErrorParametrization::fx_b_max[2] = {999, 999};

const float SiPixelCPEGenericDBErrorParametrization::by_a_min[6] = {1.47078, 1.47078, 1.47078, 1.47078, 1.47078, 1.47078};
const float SiPixelCPEGenericDBErrorParametrization::by_a_max[6] = {1.67078, 1.67078, 1.67078, 1.67078, 1.67078, 1.67078};
const float SiPixelCPEGenericDBErrorParametrization::by_b_min[6] = {0.05, 0.15, 0.70, 0.95, 1.15, 1.20};
const float SiPixelCPEGenericDBErrorParametrization::by_b_max[6] = {0.50, 0.90, 1.05, 1.15, 1.20, 1.40};
	
const float SiPixelCPEGenericDBErrorParametrization::fy_a_min[2] = {998, 998};
const float SiPixelCPEGenericDBErrorParametrization::fy_a_max[2] = {999, 999};
const float SiPixelCPEGenericDBErrorParametrization::fy_b_min[2] = {0.31, 0.31};
const float SiPixelCPEGenericDBErrorParametrization::fy_b_max[2] = {0.39, 0.39};

//Constants based on subpart 
const float SiPixelCPEGenericDBErrorParametrization::errors_big_pix[4] = {0.0070, 0.0030, 0.0068, 0.0040};
const int   SiPixelCPEGenericDBErrorParametrization::size_max[4]       = {5, 2, 0, 0};

//Garbage is set to hold a place for bx_b, though we don't parametrize it the same way
const float garbage[1] = {-9999.99};

const float* SiPixelCPEGenericDBErrorParametrization::a_min[4] = {by_a_min, bx_a_min, fy_a_min, fx_a_min};
const float* SiPixelCPEGenericDBErrorParametrization::a_max[4] = {by_a_max, bx_a_max, fy_a_max, fx_a_max};
const float* SiPixelCPEGenericDBErrorParametrization::b_min[4] = {by_b_min, garbage,  fy_b_min, fx_b_min};
const float* SiPixelCPEGenericDBErrorParametrization::b_max[4] = {by_b_max, garbage,  fy_b_max, fx_b_max};

//Bin Sizes
const int SiPixelCPEGenericDBErrorParametrization::part_bin_size[4]  = { 0, 240, 360, 380};
const int SiPixelCPEGenericDBErrorParametrization::size_bin_size[4]  = {40,  40,  40,  40};
const int SiPixelCPEGenericDBErrorParametrization::alpha_bin_size[4] = {10,   1,  10,   1};
const int SiPixelCPEGenericDBErrorParametrization::beta_bin_size[4]  = { 1,  10,   1,  10};

SiPixelCPEGenericDBErrorParametrization::SiPixelCPEGenericDBErrorParametrization(){}

SiPixelCPEGenericDBErrorParametrization::~SiPixelCPEGenericDBErrorParametrization(){}

void SiPixelCPEGenericDBErrorParametrization::setDBAccess(const edm::EventSetup& es)
{
	es.get<SiPixelCPEGenericErrorParmRcd>().get(errorsH);
}

//The function which is called to return errX and errY. Used in CPEs.
std::pair<float,float> SiPixelCPEGenericDBErrorParametrization::getError(const SiPixelCPEGenericErrorParm* parmErrors,
	                                                      GeomDetType::SubDetector pixelPart,
	                                                      int sizex, int sizey,
	                                                      float alpha, float beta,
	                                                      bool bigInX, bool bigInY)
{
	std::pair<float,float> element;
  std::pair<float,float> errors;
	
  switch (GeomDetEnumerators::subDetGeom[pixelPart])  // the _real_ subdetector enumerator is projected into the geometry subdetector enumerator
        {
		case GeomDetEnumerators::PixelBarrel:
			element = std::pair<float,float>(index(1, sizex, alpha, beta, bigInX),  //1 -- Bx
                          			       index(0, sizey, alpha, beta, bigInY)); //0 -- By
			break;
		case GeomDetEnumerators::PixelEndcap:
			element = std::pair<float,float>(index(3, sizex, alpha, beta, bigInX),  //3 -- Fx
                          				     index(2, sizey, alpha, beta, bigInY)); //2 -- Fy
			break;
		default:
			throw cms::Exception("PixelCPEGenericDBErrorParametrization::getError")
				<< "Non-pixel detector type !!!" ;
	}
	
	if (bigInX && sizex == 1) errors.first  = element.first;
	else                      errors.first  = parmErrors->errors()[(int)element.first].sigma;
	if (bigInY && sizey == 1) errors.second = element.second;
	else                      errors.second = parmErrors->errors()[(int)element.second].sigma;

	return errors;
}

//The function which is called to return errX and errY. Used outside CPEs with access to ES.
std::pair<float,float> SiPixelCPEGenericDBErrorParametrization::getError(GeomDetType::SubDetector pixelPart,
	                                                      int sizex, int sizey,
	                                                      float alpha, float beta,
	                                                      bool bigInX, bool bigInY)
{
	std::pair<float,float> element;
  std::pair<float,float> errors;
	
  switch (GeomDetEnumerators::subDetGeom[pixelPart])  // the _real_ subdetector enumerator is projected into the geometry subdetector enumerator
	{
		case GeomDetEnumerators::PixelBarrel:
			element = std::pair<float,float>(index(1, sizex, alpha, beta, bigInX),  //1 -- Bx
                          			       index(0, sizey, alpha, beta, bigInY)); //0 -- By
			break;
		case GeomDetEnumerators::PixelEndcap:
			element = std::pair<float,float>(index(3, sizex, alpha, beta, bigInX),  //3 -- Fx
                          				     index(2, sizey, alpha, beta, bigInY)); //2 -- Fy
			break;
		default:
			throw cms::Exception("PixelCPEGenericDBErrorParametrization::getError")
				<< "Non-pixel detector type !!!" ;
	}
	
	if (bigInX && sizex == 1) errors.first  = element.first;
	else                      errors.first  = errorsH->errors()[(int)element.first].sigma;
	if (bigInY && sizey == 1) errors.second = element.second;
	else                      errors.second = errorsH->errors()[(int)element.second].sigma;

	return errors;
}



float SiPixelCPEGenericDBErrorParametrization::index(int ind_subpart, int size, float alpha, float beta, bool big)
{
	//This is a check for big pixels. If it passes, the code returns a given error and the function ends.
	if ( big && size == 1) return errors_big_pix[ind_subpart];
	
	int ind_size = std::min(size - 1, size_max[ind_subpart]);

	float alpha_rad = -999.9;
	float betap_rad = -999.9;

	int ind_alpha = -99;
	int ind_beta  = -99;

	float binw_a = -999.9;
	float binw_b = -999.9;
	int maxbin_a = -99;
	int maxbin_b = -99;

	betap_rad = fabs(math_pi/2.0 - beta);
	//We must take into account that Fx(subpart=3) has different alpha parametrization
	if(ind_subpart == 3) alpha_rad = fabs(math_pi/2.0 - alpha);
	else                 alpha_rad = fabs(alpha);

	//Sets the correct binning for alpha and beta based on whether in x or y
	if(ind_subpart == 0||ind_subpart == 2)
	{
		binw_a = (a_max[ind_subpart][ind_size] - a_min[ind_subpart][ind_size])/2.0;
		binw_b = (b_max[ind_subpart][ind_size] - b_min[ind_subpart][ind_size])/8.0;
		maxbin_a = 3;
		maxbin_b = 9;
	}
	else
	{
		binw_a = (a_max[ind_subpart][ind_size] - a_min[ind_subpart][ind_size])/8.0;
		binw_b = (b_max[ind_subpart][ind_size] - b_min[ind_subpart][ind_size])/2.0;
		maxbin_a = 3;
		maxbin_b = 9;
	}

	//Binning for alpha
	if      ( alpha_rad <  a_min[ind_subpart][ind_size]) ind_alpha = 0;
	else if ( alpha_rad >= a_max[ind_subpart][ind_size]) ind_alpha = maxbin_a;
	else      ind_alpha  = 1 + (int)((alpha_rad - a_min[ind_subpart][ind_size])/binw_a);

	//Binning for beta -- we need to account for Bx(subpart=1) having uneven binning
	if(ind_subpart == 1)
	{
		if      (                     betap_rad <= 0.7 ) ind_beta = 0;
		else if ( 0.7 <  betap_rad && betap_rad <= 1.0 ) ind_beta = 1;
		else if ( 1.0 <  betap_rad && betap_rad <= 1.2 ) ind_beta = 2;
		else if ( 1.2 <= betap_rad                     ) ind_beta = 3;
	}
	else if ( betap_rad <  b_min[ind_subpart][ind_size]) ind_beta = 0;
	else if ( betap_rad >= b_max[ind_subpart][ind_size]) ind_beta = maxbin_b;
	else      ind_beta   = 1 + (int)((betap_rad - b_min[ind_subpart][ind_size])/binw_b);

	//Index to be used to find error in database
	int index = part_bin_size[ind_subpart] + size_bin_size[ind_subpart] * ind_size + alpha_bin_size[ind_subpart] * ind_alpha + beta_bin_size[ind_subpart] * ind_beta;
	
	return index;
}
