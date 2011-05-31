#ifndef RecoMuon_MuonIdentification_MuonProducer_H
#define RecoMuon_MuonIdentification_MuonProducer_H

/** \class MuonProducer
 *  Producer meant for the Post PF reconstruction.
 *
 * This class takes the muon collection produced before the PF is run (muons1Step) and the information calculated after that 
 * the entire event has been reconstructed. The collections produced here are meant to be used for the final analysis (or as PAT input).
 * The previous muon collection is meant to be transient.
 *
 *  $Date: 2010/02/11 00:14:29 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - UCSB <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"


namespace reco {class Track;}
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"


class MuonProducer : public edm::EDProducer {
public:

  /// Constructor
  MuonProducer(const edm::ParameterSet&);

  /// Destructor
  virtual ~MuonProducer();

  /// reconstruct muons
  virtual void produce(edm::Event&, const edm::EventSetup&);

protected:

private:

  std::string theAlias;

  void setAlias( std::string alias ){
    alias.erase( alias.size() - 1, alias.size() );
    theAlias=alias;
  }

private:
  bool debug_;

  edm::InputTag theMuonsCollectionLabel;
  edm::InputTag thePFCandLabel;
};
#endif

