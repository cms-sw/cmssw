#pragma once

#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelCPEGenericDBErrorParametrization.h"


// The template header files
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelGenError.h"


#include <utility>
#include <vector>


class MagneticField;
class PixelCPEFast final : public PixelCPEBase
{
public:
   struct ClusterParamGeneric : ClusterParam
   {
      ClusterParamGeneric(const SiPixelCluster & cl) : ClusterParam(cl){}
      // The truncation value pix_maximum is an angle-dependent cutoff on the
      // individual pixel signals. It should be applied to all pixels in the
      // cluster [signal_i = fminf(signal_i, pixmax)] before the column and row
      // sums are made. Morris
      int pixmx;
      
      // These are errors predicted by PIXELAV
      float sigmay; // CPE Generic y-error for multi-pixel cluster
      float sigmax; // CPE Generic x-error for multi-pixel cluster
      float sy1   ; // CPE Generic y-error for single single-pixel
      float sy2   ; // CPE Generic y-error for single double-pixel cluster
      float sx1   ; // CPE Generic x-error for single single-pixel cluster
      float sx2   ; // CPE Generic x-error for single double-pixel cluster
      
   };
   
   PixelCPEFast(edm::ParameterSet const& conf, const MagneticField *,
                   const TrackerGeometry&, const TrackerTopology&, const SiPixelLorentzAngle *,
                   const SiPixelGenErrorDBObject *, const SiPixelLorentzAngle *);
   
   
   ~PixelCPEFast();
private:
   ClusterParam * createClusterParam(const SiPixelCluster & cl) const override;
   
   LocalPoint localPosition (DetParam const & theDetParam, ClusterParam & theClusterParam) const override;
   LocalError localError   (DetParam const & theDetParam, ClusterParam & theClusterParam) const override;
   
   //--------------------------------------------------------------------
   //  Methods.
   //------------------------------------------------------------------
   static float
   generic_position_formula( int size,                //!< Size of this projection.
                            int Q_f,              //!< Charge in the first pixel.
                            int Q_l,              //!< Charge in the last pixel.
                            uint16_t upper_edge_first_pix, //!< As the name says.
                            uint16_t lower_edge_last_pix,  //!< As the name says.
                            float lorentz_shift,   //!< L-width
                            float theThickness,   //detector thickness
                            float cot_angle,            //!< cot of alpha_ or beta_
                            float pitch,            //!< thePitchX or thePitchY
                            bool first_is_big,       //!< true if the first is big
                            bool last_is_big        //!< true if the last is big
   );
   
   static void
   collect_edge_charges(ClusterParam & theClusterParam,  //!< input, the cluster
                        int & Q_f_X,              //!< output, Q first  in X
                        int & Q_l_X,              //!< output, Q last   in X
                        int & Q_f_Y,              //!< output, Q first  in Y
                        int & Q_l_Y,              //!< output, Q last   in Y
                        bool truncate
   );
   
   
   bool UseErrorsFromTemplates_;
   bool TruncatePixelCharge_;
   
   float EdgeClusterErrorX_;
   float EdgeClusterErrorY_;
   
   std::vector<float> xerr_barrel_l1_,yerr_barrel_l1_,xerr_barrel_ln_;
   std::vector<float> yerr_barrel_ln_,xerr_endcap_,yerr_endcap_;
   float xerr_barrel_l1_def_, yerr_barrel_l1_def_,xerr_barrel_ln_def_;
   float yerr_barrel_ln_def_, xerr_endcap_def_, yerr_endcap_def_;
   
   //--- DB Error Parametrization object, new light templates 
   std::vector< SiPixelGenErrorStore > thePixelGenError_;


public :

   void fillParamsForGpu();

   // not needed if not used on CPU...
   std::vector<pixelCPEforGPU::DetParams> m_detParamsGPU;
   pixelCPEforGPU::CommonParams m_commonParamsGPU;     

   pixelCPEforGPU::ParamsOnGPU h_paramsOnGPU;

   pixelCPEforGPU::ParamsOnGPU * d_paramsOnGPU;  // copy of the above on the Device  


};





