#ifndef ECALTRIGPRIMRECPRODUCERTEST_H
#define ECALTRIGPRIMRECPRODUCERTEST_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

/** \class EcalCompactTrigPrimProducerTest
 *
 *  $Id:
 *  $Date:
 *  $Revision:
 *  \author Ph. Gras CEA/IRFU Saclay
 *
 **/

class EcalCompactTrigPrimProducerTest : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  EcalCompactTrigPrimProducerTest(const edm::ParameterSet& ps)
      : tpDigiToken_(consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("tpDigiColl"))),
        tpRecToken_(consumes<EcalTrigPrimCompactColl>(ps.getParameter<edm::InputTag>("tpRecColl"))),
        tpSkimToken_(consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("tpSkimColl"))),
        nCompressEt_(0),
        nFineGrain_(0),
        nTTF_(0),
        nL1aSpike_(0),
        err_(false) {}

  /// Destructor
  ~EcalCompactTrigPrimProducerTest();

protected:
  /// Analyzes the event.
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;

private:
  std::ostream& err(const char* mess);

private:
  const edm::EDGetTokenT<EcalTrigPrimDigiCollection> tpDigiToken_;
  const edm::EDGetTokenT<EcalTrigPrimCompactColl> tpRecToken_;
  const edm::EDGetTokenT<EcalTrigPrimDigiCollection> tpSkimToken_;
  int nCompressEt_;
  int nFineGrain_;
  int nTTF_;
  int nL1aSpike_;
  bool err_;
};
#endif  //ECALTRIGPRIMRECPRODUCERTEST_H not defined
