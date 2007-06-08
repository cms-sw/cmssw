// -*-C++-*-
#ifndef L1TCOMPARE_H
#define L1TCOMPARE_H

/*
 * \file L1TCompare.h
 *
 * $Date: 2007/06/06 14:55:50 $
 * $Revision: 1.1 $
 * \author P. Wittich
 * $Id: L1TCompare.h,v 1.1 2007/06/06 14:55:50 wittich Exp $
 * $Log$
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

// DQM
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"


// Trigger Headers



//
// class declaration
//

class L1TCompare : public edm::EDAnalyzer {

public:

// Constructor
  L1TCompare(const edm::ParameterSet& ps);

// Destructor
 virtual ~L1TCompare();

protected:
// Analyze
 void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
 void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:
  // ----------member data ---------------------------
  DaqMonitorBEInterface * dbe;

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
  ofstream logFile_;

  edm::InputTag rctSource_;
  edm::InputTag gctSource_;
  edm::InputTag ecalTpgSource_;
  
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
