#include "CalibCalorimetry/HcalAlgos/interface/HBHERecalibration.h"

#include <vector>
#include <cmath>

//reuse parsing function to read mean energy table
HBHERecalibration::HBHERecalibration(float intlumi, float cutoff, std::string meanenergies) :
	intlumi_(intlumi), cutoff_(cutoff), ieta_shift_(0), max_depth_(0),
	meanenergies_(HBHEDarkening::readDoseMap(meanenergies)), darkening_(nullptr)
{}

HBHERecalibration::~HBHERecalibration() {}

void HBHERecalibration::setup(const std::vector<std::vector<int>>& m_segmentation, const HBHEDarkening* darkening)
{
	darkening_ = darkening;
	ieta_shift_ = darkening_->get_ieta_shift();

	//infer eta bounds
	int min_ieta = ieta_shift_ - 1;
	int max_ieta = min_ieta + meanenergies_.size();
	dsegm_.reserve(max_ieta - min_ieta);
	for(int ieta = min_ieta; ieta < max_ieta; ++ieta){
		dsegm_.push_back(m_segmentation[ieta]);
		//find maximum
		for(unsigned lay = 0; lay < dsegm_.back().size(); ++lay){
			if(lay>=meanenergies_[0].size()) break;
			int depth = dsegm_.back()[lay];
			if(depth>max_depth_) max_depth_ = depth;
		}
	}
	
	initialize();
}

float HBHERecalibration::getCorr(int ieta, int depth) const {
	ieta = abs(ieta);
	//shift ieta tower index to act as array index
	ieta -= ieta_shift_;

	//shift depth index to act as array index (depth = 0 - not used!)
	depth -= 1;
	
	//bounds check
	if(ieta<0 or ieta>=int(corr_.size())) return 1.0;
	if(depth<0 or depth>=int(corr_[ieta].size())) return 1.0;
	
	if(cutoff_ > 1 and corr_[ieta][depth] > cutoff_) return cutoff_;
	else return corr_[ieta][depth];
}

void HBHERecalibration::initialize() {
	std::vector<std::vector<float>> vtmp(dsegm_.size(),std::vector<float>(max_depth_,0.0));
	auto dval = vtmp; //conversion of meanenergies into depths-averaged values - denominator (including degradation for intlumi) 
	auto nval = vtmp; // conversion of meanenergies into depths-averaged values - numerator (no degradation)
	corr_ = vtmp;
  
	//converting energy values from layers into depths 
	for (unsigned int ieta = 0; ieta < dsegm_.size(); ++ieta) {
		//fill sum(means(layer,0)) and sum(means(layer,lumi)) for each depth
		for(unsigned int ilay = 0; ilay < std::min(meanenergies_[ieta].size(),dsegm_[ieta].size()); ++ilay) {
			int depth = dsegm_[ieta][ilay] - 1; // depth = 0 - not used!
			nval[ieta][depth] += meanenergies_[ieta][ilay];
			dval[ieta][depth] += meanenergies_[ieta][ilay]*darkening_->degradation(intlumi_,ieta+ieta_shift_,ilay+1); //be careful of eta and layer numbering
		}

		//compute factors, w/ safety checks
		for(int depth = 0; depth < max_depth_; ++depth){
			if(dval[ieta][depth] > 0) corr_[ieta][depth] = nval[ieta][depth]/dval[ieta][depth];
			else corr_[ieta][depth] = 1.0;
			
			if(corr_[ieta][depth] < 1.0) corr_[ieta][depth] = 1.0;
		}
  }
}
