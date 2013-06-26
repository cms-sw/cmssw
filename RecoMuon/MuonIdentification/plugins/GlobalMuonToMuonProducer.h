#ifndef RecoMuon_MuonIdentification_GlobalMuonToMuonProducer_H
#define RecoMuon_MuonIdentification_GlobalMuonToMuonProducer_H

/** \class GlobalMuonToMuonProducer
 *  No description available.
 *
 *  $Date: 2011/05/31 14:44:35 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/EDProducer.h"
//#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"


namespace reco {class Track;}
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"


class GlobalMuonToMuonProducer : public edm::EDProducer {
public:

  /// Constructor
  GlobalMuonToMuonProducer(const edm::ParameterSet&);

  /// Destructor
  virtual ~GlobalMuonToMuonProducer();

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

