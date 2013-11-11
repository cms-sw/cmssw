#include "RecoMuon/MuonIsolation/interface/MuIsoByTrackPt.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
#include "RecoMuon/MuonIsolation/interface/IsolatorByDeposit.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <vector>
#include <iostream>

using std::vector;
using std::string;
using reco::IsoDeposit;
using namespace muonisolation;


MuIsoByTrackPt::MuIsoByTrackPt(const edm::ParameterSet& conf) 
  : theExtractor(0), theIsolator(0)
{
  edm::ParameterSet extractorPSet = conf.getParameter<edm::ParameterSet>("ExtractorPSet");
  string extractorName = extractorPSet.getParameter<string>("ComponentName"); 
  theExtractor = IsoDepositExtractorFactory::get()->create(extractorName, extractorPSet);

  theCut = conf.getUntrackedParameter<double>("Threshold", 0.);
  float coneSize =  conf.getUntrackedParameter<double>("ConeSize", 0.);
  vector<double> weights(1,1.);
  theIsolator = new IsolatorByDeposit(coneSize, weights); 
}

MuIsoByTrackPt::~MuIsoByTrackPt()
{
  delete theExtractor;
  delete theIsolator;
}

void MuIsoByTrackPt::setConeSize(float dr)
{
  theIsolator->setConeSize(dr);
}

float MuIsoByTrackPt::isolation(const edm::Event& ev, const edm::EventSetup& es, const reco::Track & muon)
{
  IsoDeposit dep = extractor()->deposit(ev,es,muon); 
  MuIsoBaseIsolator::DepositContainer deposits;
  deposits.push_back(&dep);
  if (isolator()->resultType() == MuIsoBaseIsolator::ISOL_FLOAT_TYPE){
    return isolator()->result(deposits).valFloat; 
  }

  return -999.;
}

bool MuIsoByTrackPt::isIsolated(const edm::Event& ev, const edm::EventSetup& es, const reco::Track& muon)
{
  return (isolation(ev,es,muon) > theCut);
}
