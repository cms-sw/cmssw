#ifndef Muon_HLTMuonPtFilter_h
#define Muon_HLTMuonPtFilter_h

/** \class HLTMuonPtFilter
 *
 * HLTFilter to select muons above certain Pt
 *
 * $Date: 2009/02/13 15:37:48 $
 * $Revision: 1.1 $
 * \author Silvia Goy Lopez - CERN <silvia.goy.lopez@cern.ch>
 *
 */

/* Base Class Headers */
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

/* Collaborating Class Declarations */
class Propagator;

/* C++ Headers */
#include <string>

/* ====================================================================== */

/* Class HLTMuonPtFilter Interface */

class HLTMuonPtFilter : public HLTFilter {

  public:

/// Constructor
    HLTMuonPtFilter(const edm::ParameterSet&) ;

/// Destructorquer
    ~HLTMuonPtFilter() override ;

/* Operations */
    bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

  private:
    std::string theSTAMuonLabel; // label of muons
    double theMinPt;    // minimum pt required


};
#endif // Muon_HLTMuonPtFilter_h

