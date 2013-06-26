#ifndef CalibTracker_SiPixelESProducers_SiPixelDBErrorParametrization_h
#define CalibTracker_SiPixelESProducers_SiPixelDBErrorParametrization_h

#include <memory>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"

class SiPixelCPEGenericDBErrorParametrization
{
  public:
	
	SiPixelCPEGenericDBErrorParametrization();
	 ~SiPixelCPEGenericDBErrorParametrization();

	 void setDBAccess(const edm::EventSetup& es);
	 
	 std::pair<float,float>
		 getError(const SiPixelCPEGenericErrorParm* parmErrors,
			        GeomDetType::SubDetector pixelPart,
			        int sizex, int sizey,
			        float alpha, float beta,
			        bool bigInX = false,
			        bool bigInY = false);

	 std::pair<float,float>
		 getError(GeomDetType::SubDetector pixelPart,
			        int sizex, int sizey,
			        float alpha, float beta,
			        bool bigInX = false,
			        bool bigInY = false);

	 float index(int ind_subpart, int size, float alpha, float beta, bool big);
	 
  private:

	 edm::ESHandle<SiPixelCPEGenericErrorParm> errorsH;
	 
	 static const float bx_a_min[3];
	 static const float bx_a_max[3];
	 
	 static const float fx_a_min[2];
	 static const float fx_a_max[2];
	 static const float fx_b_min[2];
	 static const float fx_b_max[2];
	 
	 static const float by_a_min[6];
	 static const float by_a_max[6];
	 static const float by_b_min[6];
	 static const float by_b_max[6];
	 
	 static const float fy_a_min[2];
	 static const float fy_a_max[2];
	 static const float fy_b_min[2];
	 static const float fy_b_max[2];

	 static const float errors_big_pix[4];
	 static const int   size_max[4];
	 
	 static const int part_bin_size[4];
	 static const int size_bin_size[4];
	 static const int alpha_bin_size[4];
	 static const int beta_bin_size[4];

	 static const float* a_min[4];
	 static const float* a_max[4];
	 static const float* b_min[4];
	 static const float* b_max[4];

};
#endif
