#ifndef CalibCalorimetry_HBHERecalibration_h
#define CalibCalorimetry_HBHERecalibration_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HcalCalibObjects/interface/HBHEDarkening.h"

#include <vector>

// Simple recalibration algorithm for radiation damage to HB and HE
// produces response correction for a depth based on average of darkening per layer, weighted by mean energy per layer
// (a depth can contain several layers)
// (mean energy per layer derived from 50 GeV single pion scan in MC)

class HBHERecalibration {
	public:
		HBHERecalibration(double intlumi_, double cutoff_, const edm::ParameterSet & p);
		~HBHERecalibration();
		
		//accessors
		double getCorr(int ieta, int depth) const;
		void setDsegm(const std::vector<std::vector<int>>& m_segmentation);
		int maxDepth() const { return max_depth; }

	private:
		//helper
		void initialize();
		
		//members
		double intlumi;
		double cutoff;
		int ieta_shift;
		int max_depth;
		HBHEDarkening darkening;
		std::vector<std::vector<int>> dsegm;
		std::vector<std::vector<double>> meanenergies;
		std::vector<std::vector<double>> corr;
};

#endif // HBHERecalibration_h
