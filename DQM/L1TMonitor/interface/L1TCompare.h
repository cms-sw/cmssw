// -*-C++-*-
#ifndef L1TCOMPARE_H
#define L1TCOMPARE_H

/*
 * \file L1TCompare.h
 *
 * \author P. Wittich
 *
 * Revision 1.2  2007/06/08 08:37:42  wittich
 * Add ECAL TP - RCT comparisons. Lingering problems with
 * mismatches right now - still needs work.
 *
 *
 *
 *
*/

// system include files
#include <memory>
#include <functional>
#include <unistd.h>


#include <iostream>
#include <fstream>
#include <vector>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// GCT and RCT data formats
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

// L1Extra
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

// Ecal
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

// DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
// Trigger Headers



//
// class declaration
//

class L1TCompare : public DQMEDAnalyzer {

public:

// Constructor
  L1TCompare(const edm::ParameterSet& ps);

// Destructor
 virtual ~L1TCompare();

protected:
// Analyze
 void analyze(const edm::Event& e, const edm::EventSetup& c) override;

// BeginRun
  virtual void bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&) override;
  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

private:
  // ----------member data ---------------------------

  // ++ RCT-GCT
  // - iso
  MonitorElement* rctGctLeadingIsoEmEta_;
  MonitorElement* rctGctLeadingIsoEmPhi_;
  MonitorElement* rctGctLeadingIsoEmRank_;
  // - non-iso
  MonitorElement* rctGctLeadingNonIsoEmEta_;
  MonitorElement* rctGctLeadingNonIsoEmPhi_;
  MonitorElement* rctGctLeadingNonIsoEmRank_;

  // ++ ECAL TPG - RCT
  MonitorElement* ecalTpgRctLeadingEmEta_;
  MonitorElement* ecalTpgRctLeadingEmEta2_;
  MonitorElement* ecalTpgRctLeadingEmPhi_;
  MonitorElement* ecalTpgRctLeadingEmRank_;



  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool verbose() const { return verbose_; };
  bool monitorDaemon_;
  std::ofstream logFile_;

  edm::EDGetTokenT<L1CaloEmCollection> rctSourceEm_token_;
  edm::EDGetTokenT<L1CaloRegionCollection> rctSourceRctEmRgn_token_;
  edm::InputTag rctSource_;
  edm::InputTag gctSource_;
  edm::InputTag ecalTpgSource_;
  edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalTpgSource_token_;

  //define Token(-s)
  edm::EDGetTokenT<L1GctJetCandCollection> gctCenJetsToken_;
  edm::EDGetTokenT<L1GctEmCandCollection> gctIsoEmCandsToken_;
  edm::EDGetTokenT<L1GctEmCandCollection> gctNonIsoEmCandsToken_;
  
  class RctObject {
  public:
    RctObject(int eta, int phi, int rank):
      eta_(eta), phi_(phi), rank_(rank)
    {}
    virtual ~RctObject() {}
    int eta_, phi_;
    int rank_;
    
  };
  typedef std::vector<L1TCompare::RctObject> RctObjectCollection;

  // functor for sorting the above collection based on rank.
  // note it's then reverse-sorted (low to high) so you have to use
  // the rbegin() and rend() and reverse_iterators.
  class RctObjectComp: public std::binary_function<L1TCompare::RctObject, 
						   L1TCompare::RctObject, bool>
  {
  public:
    bool operator()(const RctObject &a, const RctObject &b) const
    {
      // for equal rank I don't know what the appropriate sorting is.
      if ( a.rank_ == b.rank_ ) {
	if ( a.eta_ == b.eta_ ) {
	  return a.phi_ < b.phi_;
	}
	else {
	  return a.eta_ < b.eta_;
	}
      }
      else {
	return a.rank_ < b.rank_;
      }
    }
  };


};

#endif // L1TCOMPARE_H
