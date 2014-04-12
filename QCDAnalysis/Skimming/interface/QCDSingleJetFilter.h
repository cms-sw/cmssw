#ifndef __QCDSingleJetFilter__H__
#define __QCDSingleJetFilter__H__

/* \class QCDSingleJetFilter for CSA07 Excercise
 *  by Andreas Oehler (andreas.oehler@cern.ch)
 *
 Filter to select on pt of the jet with hightest Pt
 can do a random prescale (requires RandomNumber Service)[deactivated]

 */

// user include files
#include <FWCore/Framework/interface/Frameworkfwd.h>

#include <FWCore/Framework/interface/EDFilter.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Utilities/interface/InputTag.h"

#include <DataFormats/JetReco/interface/CaloJet.h>


//RandomGenerator Service taken out
//namespace CLHEP {
//  class RandFlat;
//}

class QCDSingleJetFilter : public edm::EDFilter {

 public:
  // Constructor
  QCDSingleJetFilter(const edm::ParameterSet&);

  // Destructor
  virtual ~QCDSingleJetFilter();

  /// Get event properties to send to builder to fill seed collection
  virtual bool filter(edm::Event&, const edm::EventSetup&);



 private:

  bool debug;
  //CLHEP::RandFlat *theFlatDistrib;
  double theMinPt;
    //,thePreScale;
  edm::EDGetTokenT<reco::CaloJetCollection> theTriggerJetCollectionAToken;
  edm::EDGetTokenT<reco::CaloJetCollection> theTrigCollBToken;
};

#endif
