#ifndef Muon_HLTMuonPtFilter_h
#define Muon_HLTMuonPtFilter_h

/** \class HLTMuonPtFilter
 *
 * HLTFilter to select muons above certain Pt
 *
 * $Date: 2012/01/21 14:56:53 $
 * $Revision: 1.2 $
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
    ~HLTMuonPtFilter() ;

/* Operations */ 
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

  private:
    std::string theSTAMuonLabel; // label of muons 
    double theMinPt;    // minimum pt required 


};
#endif // Muon_HLTMuonPtFilter_h

