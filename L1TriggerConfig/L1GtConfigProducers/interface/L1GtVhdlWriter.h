#ifndef L1GtConfigProducers_L1GtVhdlWriter_h
#define L1GtConfigProducers_L1GtVhdlWriter_h

/**
 * \class L1GtVhdlWriter
 *
 *
 * Description: write VHDL templates for the L1 GT.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Philipp Wagner
 *
 *
 */

// system include files
#include <string>

// base class

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

class Event;
class EventSetup;
class ParameterSet;

// forward declarations

// class declaration
class L1GtVhdlWriter : public edm::one::EDAnalyzer<> {
public:
  /// constructor
  explicit L1GtVhdlWriter(const edm::ParameterSet&);

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  /// templates directory
  std::string vhdlDir_;

  /// output directory
  std::string outputDir_;

  edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> menuToken_;
};
#endif /*L1GtConfigProducers_L1GtVhdlWriter_h*/
