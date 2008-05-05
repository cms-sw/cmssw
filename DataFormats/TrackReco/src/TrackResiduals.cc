#include <math.h>
#include "DataFormats/TrackReco/interface/HitPattern.h"
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
     switch (residualType) {
     case X_Y_RESIDUALS:
	  return unpack_residual(residuals_[i] >> 4);
     case X_Y_PULLS:
	  return unpack_pull(residuals_[i] >> 4);
     default:
	  assert(0);
     }
     return 0;
}

double TrackResiduals::residualY (int i) const
{
     switch (residualType) {
     case X_Y_RESIDUALS:
	  return unpack_residual(residuals_[i] & 0x0f);
     case X_Y_PULLS:
	  return unpack_pull(residuals_[i] & 0x0f);
     default:
	  assert(0);
     }
     return 0;
}

static int index_to_hitpattern (int i_hitpattern, const HitPattern &h)
{
     int i_residuals = 0;
     assert(i_hitpattern < h.numberOfHits());
     if (!h.validHitFilter(h.getHitPattern(i_hitpattern))) 
	  // asking for residual of invalid hit...
	  return -999;
     for (int i = 0; i <= i_hitpattern; i++) {
	  if (h.validHitFilter(h.getHitPattern(i)))
	       i_residuals++;
     }
     return i_residuals;
}

double TrackResiduals::residualX (int i, const HitPattern &h) const
{
     int idx = index_to_hitpattern(i, h);
     if (idx == -999)
	  return -999;
     return residualX(idx);
}

double TrackResiduals::residualY (int i, const HitPattern &h) const
{
     int idx = index_to_hitpattern(i, h);
     if (idx == -999)
	  return -999;
     return residualY(idx);
}

void TrackResiduals::setPullXY (int idx, double pullX, double pullY)
{
     assert(residualType == X_Y_PULLS);
     residuals_[idx] = (pack_pull(pullX) << 4) | pack_pull(pullY);
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

void TrackResiduals::print (const HitPattern &h, std::ostream &stream) const
{
     stream << "TrackResiduals" << std::endl;
     for (int i = 0; i < h.numberOfHits(); i++) {
	  stream << "( " << residualX(i, h) << " , " << residualY(i, h) << " )" 
		 << std::endl;
     }
}
