//-------------------------------------------------
//
/**  \class L1TMuonBarrelTrackProducer
 *
 *   L1 BM Track Finder EDProducer
 *
 *
 *
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef BMTrackFinder_h
#define BMTrackFinder_h

#include <FWCore/Framework/interface/one/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <string>

class L1MuBMTFSetup;


class L1TMuonBarrelTrackProducer: public edm::one::EDProducer<edm::one::SharedResources> {
 public:
  /// Constructor
  L1TMuonBarrelTrackProducer(const edm::ParameterSet & pset);

  /// Destructor
  ~L1TMuonBarrelTrackProducer() override;

  /// Produce digis out of raw data
  void produce(edm::Event & e, const edm::EventSetup& c) override;

 private:

  L1MuBMTFSetup* setup1;
  const edm::ParameterSet* m_ps;


};

#endif
