#ifndef Muon_MuonPtFilter_h
#define Muon_MuonPtFilter_h

/** \class MuonPtFilter
 *
 * HLTFilter to select muons above certain Pt
 *
 * $Date: 2009/02/18 12:36:38 $
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

/* Class MuonPtFilter Interface */

class MuonPtFilter : public HLTFilter {

  public:

/// Constructor
    MuonPtFilter(const edm::ParameterSet&) ;

/// Destructorquer
    ~MuonPtFilter() ;

/* Operations */ 
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

  private:
    std::string theSTAMuonLabel; // label of muons 
    double theMinPt;    // minimum pt required 


};
#endif // Muon_MuonPtFilter_h

