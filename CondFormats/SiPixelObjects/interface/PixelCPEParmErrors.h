#ifndef PixelCPEParmErrors_h
#define PixelCPEParmErrors_H

#include <vector>
class PixelCPEParmErrors {
public:
	struct pixelCPEParmErrorsEntry {
		int part;
		int size;
		int alpha;
		int beta;
		double sigma;
	};
	PixelCPEParmErrors(){}
	virtual ~PixelCPEParmErrors(){}
	std::vector<pixelCPEParmErrorsEntry> pixelCPEParmErrors;
};

#endif
