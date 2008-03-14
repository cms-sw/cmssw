#ifndef SiPixelCPEParmErrors_h
#define SiPixelCPEParmErrors_H

#include <vector>
class SiPixelCPEParmErrors {
public:
	struct siPixelCPEParmErrorsEntry {
		float sigma;
		float rms;
		float bias;
		float pix_height;
		float ave_qclu;
	};
	SiPixelCPEParmErrors(){}
	virtual ~SiPixelCPEParmErrors(){}
	std::vector<siPixelCPEParmErrorsEntry> siPixelCPEParmErrors_Bx;
	std::vector<siPixelCPEParmErrorsEntry> siPixelCPEParmErrors_By;
	std::vector<siPixelCPEParmErrorsEntry> siPixelCPEParmErrors_Fx;
	std::vector<siPixelCPEParmErrorsEntry> siPixelCPEParmErrors_Fy;
};

#endif
