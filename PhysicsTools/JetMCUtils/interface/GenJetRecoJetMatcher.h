#ifndef GENJETRECOJETMATCHER_H
#define GENJETRECOJETMATCHER_H

// class to match GenJet to RecoJet

/* \class GenJetRecoJetMatcher
 *
 * Producer for simple match map
 * based on DeltaR
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include<vector>

class GenJetRecoJetMatcher : public edm::EDProducer {
 public:
  GenJetRecoJetMatcher( const edm::ParameterSet & );
  ~GenJetRecoJetMatcher();
 private:
  void produce( edm::Event&, const edm::EventSetup& );
  double lenght(char* my_str);
  
  edm::InputTag source_;
  edm::InputTag matched_;
  bool printdebug_;
  std::vector < std::vector<float> > AllDist;
};

#endif

