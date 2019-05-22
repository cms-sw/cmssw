/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#ifndef DataFormats_CTPPSReco_CTPPSLocalTrackLiteFwd_h
#define DataFormats_CTPPSReco_CTPPSLocalTrackLiteFwd_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

#include <vector>

class CTPPSLocalTrackLite;
/// Collection of CTPPSLocalTrackLite objects
typedef std::vector<CTPPSLocalTrackLite> CTPPSLocalTrackLiteCollection;
/// Persistent reference to a CTPPSLocalTrackLite
typedef edm::Ref<CTPPSLocalTrackLiteCollection> CTPPSLocalTrackLiteRef;
/// Reference to a CTPPSLocalTrackLite collection
typedef edm::RefProd<CTPPSLocalTrackLiteCollection> CTPPSLocalTrackLiteRefProd;
/// Vector of references to CTPPSLocalTrackLite in the same collection
typedef edm::RefVector<CTPPSLocalTrackLiteCollection> CTPPSLocalTrackLiteRefVector;

#endif

