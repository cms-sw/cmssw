#ifndef RecoMuon_MuonIdentification_GlobalMuonToMuonProducer_H
#define RecoMuon_MuonIdentification_GlobalMuonToMuonProducer_H

/** \class GlobalMuonToMuonProducer
 *  No description available.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/global/EDProducer.h"
//#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"


namespace reco {class Track;}
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"


class GlobalMuonToMuonProducer : public edm::global::EDProducer<> {
public:

  /// Constructor
  GlobalMuonToMuonProducer(const edm::ParameterSet&);

  /// Destructor
  virtual ~GlobalMuonToMuonProducer();

  /// reconstruct muons
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

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
  edm::EDGetTokenT<reco::MuonTrackLinksCollection> trackLinkToken_;
};
#endif

