#include "DataFormats/Math/interface/FastMath.h"
namespace fastmath_details {
  float atanbuf_[257 * 2];
  double datanbuf_[513 * 2];

  namespace {
    // ====================================================================
    // arctan initialization
    // =====================================================================
    struct Initatan {
      Initatan() {
	unsigned int ind;
	for (ind = 0; ind <= 256; ind++) {
	  double v = ind / 256.0;
	  double asinv = ::asin(v);
	  atanbuf_[ind * 2    ] = ::cos(asinv);
	  atanbuf_[ind * 2 + 1] = asinv;
	}
	for (ind = 0; ind <= 512; ind++) {
	  double v = ind / 512.0;
	  double asinv = ::asin(v);
	  datanbuf_[ind * 2    ] = ::cos(asinv);
	  datanbuf_[ind * 2 + 1] = asinv;
	}
      }
    };
    Initatan initAtan;
  }
}
