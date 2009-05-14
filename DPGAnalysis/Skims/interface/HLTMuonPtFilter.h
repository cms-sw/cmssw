#ifndef Muon_HLTMuonPtFilter_h
#define Muon_HLTMuonPtFilter_h

/** \class HLTMuonPtFilter
 *
 * HLTFilter to select muons above certain Pt
 *
 * $Date: 2009/13/02 16:21:14 $
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
    ~HLTMuonPtFilter() ;

/* Operations */ 
    virtual bool filter(edm::Event&, const edm::EventSetup&);

  private:
    std::string theSTAMuonLabel; // label of muons 
    double theMinPt;    // minimum pt required 


};
#endif // Muon_HLTMuonPtFilter_h

