#ifndef Muon_MuonPtFilter_h
#define Muon_MuonPtFilter_h

/** \class MuonPtFilter
 *
 * EDFilter to select muons above certain Pt
 *
 * $Date: 2013/02/27 20:17:13 $
 * $Revision: 1.4 $
 * \author Silvia Goy Lopez - CERN <silvia.goy.lopez@cern.ch>
 *
 */

/* Base Class Headers */
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/* Collaborating Class Declarations */
class Propagator;

/* C++ Headers */
#include <string>

/* ====================================================================== */

/* Class MuonPtFilter Interface */

class MuonPtFilter : public edm::EDFilter {

  public:

/// Constructor
    MuonPtFilter(const edm::ParameterSet&) ;

/// Destructorquer
    ~MuonPtFilter() ;

/* Operations */ 
    virtual bool filter(edm::Event &, const edm::EventSetup&) override;

  private:
    std::string theSTAMuonLabel; // label of muons 
    double theMinPt;    // minimum pt required 


};
#endif // Muon_MuonPtFilter_h

