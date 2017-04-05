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
		HBHERecalibration(double intlumi, double cutoff, const edm::ParameterSet & p);
		~HBHERecalibration();
		
		//accessors
		double getCorr(int ieta, int depth) const;
		void setDsegm(const std::vector<std::vector<int>>& m_segmentation);
		int maxDepth() const { return max_depth_; }

	private:
		//helper
		void initialize();
		
		//members
		double intlumi_;
		double cutoff_;
		int ieta_shift_;
		int max_depth_;
		HBHEDarkening darkening_;
		std::vector<std::vector<int>> dsegm_;
		std::vector<std::vector<double>> meanenergies_;
		std::vector<std::vector<double>> corr_;
};

#endif // HBHERecalibration_h
