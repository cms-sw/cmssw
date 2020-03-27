#ifndef __KFpragmaOpts__
#define __KFpragmaOpts__

/**
 * Define options specified by pragma statements.
 *
 * Author: Ian Tomalin
 */

// OPTION 1:
// If defined, HLS KF will cope with tracking down to 2 GeV Pt instead of 3 GeV.
#define PT_2GEV

// If defined, HLS assumes hybrid (=tracklet) input format & digitisation multipliers.
#define HYBRID_FORMAT

#endif
