/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 ****************************************************************************/

#ifndef RecoCTPPS_ProtonReconstruction_ProtonReconstructionAlgorithm_h
#define RecoCTPPS_ProtonReconstruction_ProtonReconstructionAlgorithm_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsSet.h"

#include "TSpline.h"
#include "Fit/Fitter.h"

#include <unordered_map>

//----------------------------------------------------------------------------------------------------

class ProtonReconstructionAlgorithm
{
  public:
    ProtonReconstructionAlgorithm(bool fit_vtx_y, bool improved_estimate, unsigned int verbosity);
    ~ProtonReconstructionAlgorithm() = default;

    void init(const std::unordered_map<unsigned int, LHCOpticalFunctionsSet> &opticalFunctions);
    void release();

    /// run proton reconstruction using single-RP strategy
    void reconstructFromSingleRP(const CTPPSLocalTrackLiteRefVector& input,
      reco::ForwardProtonCollection& output,
      const LHCInfo& lhcInfo, std::ostream& os) const;

    /// run proton reconstruction using multiple-RP strategy
    void reconstructFromMultiRP(const CTPPSLocalTrackLiteRefVector& input,
      reco::ForwardProtonCollection& output,
      const LHCInfo& lhcInfo, std::ostream& os) const;

  private:
    unsigned int verbosity_;
    bool fitVtxY_;
    bool useImprovedInitialEstimate_;
    bool initialized_;

    /// optics data associated with 1 RP
    struct RPOpticsData
    {
      const LHCOpticalFunctionsSet *optics;
      std::shared_ptr<const TSpline3> s_xi_vs_x_d, s_y_d_vs_xi, s_v_y_vs_xi, s_L_y_vs_xi;
      double x0; ///< beam horizontal position, cm
      double y0; ///< beam vertical position, cm
      double ch0; ///< intercept for linear approximation of \f$x(\xi)\f$
      double ch1; ///< slope for linear approximation of \f$x(\xi)\f$
      double la0; ///< intercept for linear approximation of \f$L_x(\xi)\f$
      double la1; ///< slope for linear approximation of \f$L_x(\xi)\f$
    };

    /// map: RP id --> optics data
    std::map<unsigned int, RPOpticsData> m_rp_optics_;

    /// class for calculation of chi^2
    class ChiSquareCalculator
    {
      public:
        ChiSquareCalculator() = default;

        double operator() (const double *parameters) const;

        const CTPPSLocalTrackLiteRefVector* tracks;
        const std::map<unsigned int, RPOpticsData>* m_rp_optics;
    };

    /// fitter object
    std::unique_ptr<ROOT::Fit::Fitter> fitter_;

    /// object to calculate chi^2
    std::unique_ptr<ChiSquareCalculator> chiSquareCalculator_;

    static void doLinearFit(const std::vector<double> &vx, const std::vector<double> &vy, double &b, double &a);
};

#endif

