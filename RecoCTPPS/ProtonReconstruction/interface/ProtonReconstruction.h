/****************************************************************************
 *
 * This is a part of CTPPS offline software
 * Authors:
 *   Leszek Grzanka
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#ifndef SimRomanPot_CTPPSOpticsParameterisation_ProtonReconstructionAlgorithm_h
#define SimRomanPot_CTPPSOpticsParameterisation_ProtonReconstructionAlgorithm_h

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "SimDataFormats/CTPPS/interface/LHCOpticsApproximator.h"
//#include "SimDataFormats/CTPPS/interface/LHCApertureApproximator.h"
#include "SimDataFormats/CTPPS/interface/CTPPSSimProtonTrack.h"

#include "TFile.h"
#include "TSpline.h"
#include "Fit/Fitter.h"

#include <map>
#include <unordered_map>
#include <string>
#include <cmath>

enum LHCSector { unknownSector, sector45, sector56 };

class ProtonReconstructionAlgorithm
{
  public:
    ProtonReconstructionAlgorithm( const edm::ParameterSet&, std::unordered_map<unsigned int, std::string>, const std::string&, bool, bool );
    ~ProtonReconstructionAlgorithm();

    void reconstruct( const std::vector< edm::Ptr<CTPPSLocalTrackLite> >& tracks, std::vector<CTPPSSimProtonTrack>& reco ) const;

  private:
    /// optics data associated with 1 RP
    struct RPOpticsData {
      std::shared_ptr<LHCOpticsApproximator> optics;
      std::shared_ptr<TSpline3> s_xi_vs_x, s_y0_vs_xi, s_v_y_vs_xi, s_L_y_vs_xi;
    };

    edm::ParameterSet beamConditions_;
    double halfCrossingAngleSector45_, halfCrossingAngleSector56_;
    double yOffsetSector45_, yOffsetSector56_;
    /// map: RP id --> optics data
    std::map<TotemRPDetId,RPOpticsData> m_rp_optics_;

    /// class for calculation of chi^2
    class ChiSquareCalculator {
      public:
        ChiSquareCalculator( const edm::ParameterSet& bc, bool aper, bool invert ) :
          beamConditions_( bc ),
          halfCrossingAngleSector45_( bc.getParameter<double>( "halfCrossingAngleSector45" ) ),
          halfCrossingAngleSector56_( bc.getParameter<double>( "halfCrossingAngleSector56" ) ),
          yOffsetSector45_( bc.getParameter<double>( "yOffsetSector45" ) ),
          yOffsetSector56_( bc.getParameter<double>( "yOffsetSector56" ) ),
          check_apertures( aper ), invert_beam_coord_systems( invert ) {}
        double operator() ( const double* ) const;

        const std::vector< edm::Ptr<CTPPSLocalTrackLite> >* tracks;
        const std::map<TotemRPDetId,RPOpticsData>* m_rp_optics;

      private:
        edm::ParameterSet beamConditions_;
        double halfCrossingAngleSector45_, halfCrossingAngleSector56_;
        double yOffsetSector45_, yOffsetSector56_;
        const bool check_apertures;
        const bool invert_beam_coord_systems;
    };

    // fitter object
    std::unique_ptr<ROOT::Fit::Fitter> fitter_;

    bool checkApertures_;
    bool invertBeamCoordinatesSystem_;

    std::unique_ptr<ChiSquareCalculator> chiSquareCalculator_;
};

#endif

//------------------------------------------------------------------------------
// JAN'S VERSION
//------------------------------------------------------------------------------

/****************************************************************************
*
* This is a part of CTPPS offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef _proton_reconstruction_h_
#define _proton_reconstruction_h_

#include "TFile.h"
#include "TSpline.h"
#include "Fit/Fitter.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "SimDataFormats/CTPPS/interface/LHCOpticsApproximator.h"

#include "RecoCTPPS/ProtonReconstruction/interface/BeamConditions.h"
#include "RecoCTPPS/ProtonReconstruction/interface/ProtonData.h"

#include <map>
#include <string>
#include <cmath>

//----------------------------------------------------------------------------------------------------

class ProtonReconstruction
{
	public:

		ProtonReconstruction() {}

		int Init(const std::string &optics_file_beam1, const std::string &optics_file_beam2);

		ProtonData Reconstruct(LHCSector sector, const std::vector<CTPPSLocalTrackLite> &tracks) const;

		~ProtonReconstruction()
		{
			for (auto &it : m_rp_optics)
			{
				delete it.second.optics;
				delete it.second.s_xi_vs_x;
				delete it.second.s_y0_vs_xi;
				delete it.second.s_v_y_vs_xi;
				delete it.second.s_L_y_vs_xi;
			}

			if (chiSquareCalculator)
				delete chiSquareCalculator;

			if (fitter)
				delete fitter;
		}

	private:
		/// optics data associated with 1 RP
		struct RPOpticsData
		{
			LHCOpticsApproximator *optics;
			TSpline3 *s_xi_vs_x, *s_y0_vs_xi, *s_v_y_vs_xi, *s_L_y_vs_xi;
		};

		/// map: RP id --> optics data
		std::map<unsigned int, RPOpticsData> m_rp_optics;

		/// class for calculation of chi^2
		class ChiSquareCalculator
		{
			public:
				const std::vector<CTPPSLocalTrackLite> *tracks;
				const std::map<unsigned int, RPOpticsData> *m_rp_optics;

				ChiSquareCalculator() {}

				double operator() (const double *parameters) const;
		};

		/// instance of ChiSquareCalculator used by the fitter
		ChiSquareCalculator *chiSquareCalculator = NULL;

		/// fitter object
		ROOT::Fit::Fitter *fitter = NULL;
};

#endif
