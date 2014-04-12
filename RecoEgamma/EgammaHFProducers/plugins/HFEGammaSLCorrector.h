#ifndef HFEGamma_SL_CORRECTOR
#define HFEGamma_SL_CORRECTOR 1

namespace hf_egamma {

	double eSeLCorrected(double es, double el, double m, double b);

	//enum CorrectionEra { ce_Fall10, ce_Spring11, ce_Summer11, ce_Data41 };

	double eSeLCorrected(double es, double el, int era);
}

#endif // HFEGamma_SL_CORRECTOR

