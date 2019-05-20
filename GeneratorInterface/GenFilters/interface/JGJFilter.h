#ifndef JGJFilter_h
#define JGJFilter_h

/*******************************************************
*
*   Original Author:  Alexander Proskuryakov
*           Created:  Sat Aug  1 10:42:50 CEST 2009
*
*       Modified by:  Sheila Amaral
* Last modification:  Thu Aug 13 09:46:26 CEST 2009
* 
* Allows events which have at least 2 highest ET jets,
* at generator level, with deltaEta between jets higher
* than 3.5
*
*******************************************************/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

class JGJFilter : public edm::EDFilter {
public:
  explicit JGJFilter(const edm::ParameterSet&);
  ~JGJFilter() override;

private:
  void beginJob() override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  unsigned int nEvents;
  unsigned int nAccepted;
};

#endif
