#include "CalibCalorimetry/HcalAlgos/interface/HBHERecalibration.h"

#include <vector>
#include <cmath>

//reuse parsing function to read mean energy table
HBHERecalibration::HBHERecalibration(double intlumi_, double cutoff_, const edm::ParameterSet & p) :
	intlumi(intlumi_), cutoff(cutoff_), ieta_shift(p.getParameter<int>("ieta_shift")), max_depth(0), darkening(p), 
	meanenergies(HBHEDarkening::readDoseMap(p.getParameter<edm::FileInPath>("meanenergies").fullPath()))
{}

HBHERecalibration::~HBHERecalibration() {}

void HBHERecalibration::setDsegm(const std::vector<std::vector<int>>& m_segmentation) 
{
	//infer eta bounds
	int min_ieta = ieta_shift - 1;
	int max_ieta = min_ieta + meanenergies.size();
	dsegm.reserve(max_ieta - min_ieta);
	for(int ieta = min_ieta; ieta < max_ieta; ++ieta){
		dsegm.push_back(m_segmentation[ieta]);
		//find maximum
		for(unsigned lay = 0; lay < dsegm.back().size(); ++lay){
			if(lay>=meanenergies[0].size()) break;
			int depth = dsegm.back()[lay];
			if(depth>max_depth) max_depth = depth;
		}
	}
	
	initialize();
}

double HBHERecalibration::getCorr(int ieta, int depth) const {
	ieta = abs(ieta);
	//shift ieta tower index to act as array index
	ieta -= ieta_shift;

	//shift depth index to act as array index (depth = 0 - not used!)
	depth -= 1;
	
	//bounds check
	if(ieta<0 or ieta>=int(corr.size())) return 1.0;
	if(depth<0 or depth>=int(corr[ieta].size())) return 1.0;
	
	if(cutoff > 1 and corr[ieta][depth] > cutoff) return cutoff;
	else return corr[ieta][depth];
}

void HBHERecalibration::initialize() {
	std::vector<std::vector<double>> vtmp(dsegm.size(),std::vector<double>(max_depth,0.0));
	auto dval = vtmp; //conversion of meanenergies into depths-averaged values - denominator (including degradation for intlumi) 
	auto nval = vtmp; // conversion of meanenergies into depths-averaged values - numerator (no degradation)
	corr = vtmp;
  
	//converting energy values from layers into depths 
	for (unsigned int ieta = 0; ieta < dsegm.size(); ++ieta) {
		//fill sum(means(layer,0)) and sum(means(layer,lumi)) for each depth
		for(unsigned int ilay = 0; ilay < std::min(meanenergies[ieta].size(),dsegm[ieta].size()); ++ilay) {
			int depth = dsegm[ieta][ilay] - 1; // depth = 0 - not used!
			nval[ieta][depth] += meanenergies[ieta][ilay];
			dval[ieta][depth] += meanenergies[ieta][ilay]*darkening.degradation(intlumi,ieta+ieta_shift,ilay+1); //be careful of eta and layer numbering
		}

		//compute factors, w/ safety checks
		for(int depth = 0; depth < max_depth; ++depth){
			if(dval[ieta][depth] > 0) corr[ieta][depth] = nval[ieta][depth]/dval[ieta][depth];
			else corr[ieta][depth] = 1.0;
			
			if(corr[ieta][depth] < 1.0) corr[ieta][depth] = 1.0;
		}
  }
}
