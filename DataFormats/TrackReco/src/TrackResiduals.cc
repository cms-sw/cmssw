#include <math.h>
#include "DataFormats/TrackReco/interface/TrackResiduals.h"
using namespace reco;

TrackResiduals::TrackResiduals () : residualType(X_Y_RESIDUALS)
{
     memset(residuals_, 0, sizeof(residuals_));
}

TrackResiduals::TrackResiduals (enum ResidualType type) : residualType(type)
{
     memset(residuals_, 0, sizeof(residuals_));
}

void TrackResiduals::setResidualType (enum ResidualType type)
{
     residualType = type;
}

void TrackResiduals::setResidualXY (int idx, double residualX, double residualY)
{
     assert(residualType == X_Y_RESIDUALS);
     residuals_[idx] = (pack_residual(residualX) << 4) | pack_residual(residualY);
}

double TrackResiduals::residualX (int i) const
{
     return unpack_residual(residuals_[i] >> 4);
}

double TrackResiduals::residualY (int i) const
{
     return unpack_residual(residuals_[i] & 0x0f);
}

void TrackResiduals::setPullXY (int idx, double pullX, double pullY)
{
     assert(residualType == X_Y_PULLS);
     residuals_[idx] = (pack_pull(pullX) << 4) | pack_pull(pullY);
}

double TrackResiduals::pullX (int i) const
{
     return unpack_pull(residuals_[i] >> 4);
}

double TrackResiduals::pullY (int i) const
{
     return unpack_pull(residuals_[i] & 0x0f);
}

static const double pull_char_to_double[8][2] = { 
     { 0,   0.5 },
     { 0.5, 1   },
     { 1,   1.5 },
     { 1.5, 2   },
     { 2,   2.5 },
     { 2.5, 3.5 },
     { 3.5, 4.5 },
     { 4.5, 5.5 },
};

double TrackResiduals::unpack_pull (unsigned char pull)
{
     int sgn = 2 * (pull & 0x08) - 1;
     unsigned char mag = pull & 0x07;
     return sgn * 
	  (pull_char_to_double[mag][0] + pull_char_to_double[mag][1]) / 2;
}

unsigned char TrackResiduals::pack_pull (double pull)
{
     unsigned char sgn = (pull < 0) * 0x08; // 1xxx is -abs(0xxx)
     int mag = -1;
     while (++mag < 8 && pull_char_to_double[mag][1] < fabs(pull));
     return sgn + mag;
}

static const double residual_char_to_double[8][2] = { 
     { 0,   0.5 },
     { 0.5, 1   },
     { 1,   1.5 },
     { 1.5, 2   },
     { 2,   2.5 },
     { 2.5, 3.5 },
     { 3.5, 4.5 },
     { 4.5, 5.5 },
};

double TrackResiduals::unpack_residual (unsigned char pull)
{
     signed char sgn = 2 * (pull & 0x08) - 1;
     unsigned char mag = pull & 0x07;
     return sgn * 
	  (pull_char_to_double[mag][0] + pull_char_to_double[mag][1]) / 2;
}

unsigned char TrackResiduals::pack_residual (double pull)
{
     unsigned char sgn = (pull < 0) * 0x08; // 1xxx is -abs(0xxx)
     int mag = -1;
     while (++mag < 8 && pull_char_to_double[mag][1] < fabs(pull));
     return sgn + mag;
}

void TrackResiduals::print (std::ostream &stream) const
{
     stream << "TrackResiduals" << std::endl;
     std::ios_base::fmtflags flags = stream.flags();
     stream.setf ( std::ios_base::hex, std::ios_base::basefield );  
     stream.setf ( std::ios_base::showbase );               
     for (int i = 0; i < numResiduals; i++) {
	  unsigned char residual = residuals_[i];
 	  printf("0x%x\n", residual);
//  	  stream << residual << std::endl;
     }
     stream.flags(flags);
}
