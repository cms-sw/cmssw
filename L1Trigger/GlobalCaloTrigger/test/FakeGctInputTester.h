
// -*- C++ -*-
//
// Package:    FakeGctInputTester
// Class:      FakeGctInputTester
//
/**\class FakeGctInputTester FakeGctInputTester.h L1Trigger/GlobalCalotrigger/src/FakeGctInputTester.h

 \brief EDAnalyzer to check GCT output using fake input

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Tue Nov 21 14:49:14 CET 2006
//
//

#ifndef FAKEGCTINPUTTESTER_H
#define FAKEGCTINPUTTESTER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include <string>

class TH1F;
class TFile;

class FakeGctInputTester : public edm::one::EDAnalyzer<> {
public:
  explicit FakeGctInputTester(const edm::ParameterSet&);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  std::string hFileName_;
  TFile* hFile_;

  TH1F* isoEmDEta_;
  TH1F* isoEmDPhi_;

  TH1F* nonIsoEmDEta_;
  TH1F* nonIsoEmDPhi_;

  TH1F* jetDEta_;
  TH1F* jetDPhi_;
};

#endif
