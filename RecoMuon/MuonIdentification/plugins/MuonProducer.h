#ifndef RecoMuon_MuonIdentification_MuonProducer_H
#define RecoMuon_MuonIdentification_MuonProducer_H

/** \class MuonProducer
 *  No description available.
 *
 *  $Date: 2007/05/12 22:14:39 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/EDProducer.h"
//#include "FWCore/Framework/interface/Frameworkfwd.h"
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

  // tmp
  void printTrackRecHits(const reco::Track &track, 
			 edm::ESHandle<GlobalTrackingGeometry> trackingGeometry) const;


private:

  edm::InputTag theLinksCollectionLabel;
};
#endif

