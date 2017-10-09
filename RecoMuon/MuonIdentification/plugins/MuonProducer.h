#ifndef RecoMuon_MuonIdentification_MuonProducer_H
#define RecoMuon_MuonIdentification_MuonProducer_H

/** \class MuonProducer
 *  Producer meant for the Post PF reconstruction.
 *
 * This class takes the muon collection produced before the PF is run (muons1Step) and the information calculated after that 
 * the entire event has been reconstructed. The collections produced here are meant to be used for the final analysis (or as PAT input).
 * The previous muon collection is meant to be transient.
 *
 *  \author R. Bellan - UCSB <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"

namespace reco {class Track;}
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"
#include "DataFormats/MuonReco/interface/MuonShower.h"
#include "DataFormats/MuonReco/interface/MuonCosmicCompatibility.h"
#include "DataFormats/MuonReco/interface/MuonToMuonMap.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"


class MuPFIsoHelper;


class MuonProducer : public edm::stream::EDProducer<> {
public:

  /// Constructor
  MuonProducer(const edm::ParameterSet&);

  /// Destructor
  virtual ~MuonProducer();

  /// reconstruct muons
  virtual void produce(edm::Event&, const edm::EventSetup&) override;


  typedef std::vector<edm::InputTag> InputTags;

protected:

private:
  template<typename TYPE>
    void fillMuonMap(edm::Event& event,
		     const edm::OrphanHandle<reco::MuonCollection>& muonHandle,
		     const std::vector<TYPE>& muonExtra,
		     const std::string& label);
  
  std::string theAlias;

  void setAlias( std::string alias ){
    alias.erase( alias.size() - 1, alias.size() );
    theAlias=alias;
  }

  std::string labelOrInstance(const edm::InputTag &) const;

private:
  bool debug_;
  bool fastLabelling_;
  
  edm::InputTag theMuonsCollectionLabel;
  edm::EDGetTokenT<reco::MuonCollection> theMuonsCollectionToken_;
 
  edm::InputTag thePFCandLabel;
  edm::EDGetTokenT<reco::PFCandidateCollection> thePFCandToken_;
 


  bool fillIsolation_;
  bool writeIsoDeposits_;
  bool fillSelectors_;
  bool fillCosmicsIdMap_;
  bool fillPFMomentum_;
  bool fillPFIsolation_;
  bool fillDetectorBasedIsolation_;
  bool fillShoweringInfo_;
  bool fillTimingInfo_;

  edm::InputTag theTrackDepositName;
  edm::InputTag theEcalDepositName;
  edm::InputTag theHcalDepositName;
  edm::InputTag theHoDepositName;
  edm::InputTag theJetDepositName;

  edm::EDGetTokenT<reco::IsoDepositMap> theTrackDepositToken_;
  edm::EDGetTokenT<reco::IsoDepositMap> theEcalDepositToken_;
  edm::EDGetTokenT<reco::IsoDepositMap> theHcalDepositToken_;
  edm::EDGetTokenT<reco::IsoDepositMap> theHoDepositToken_;
  edm::EDGetTokenT<reco::IsoDepositMap> theJetDepositToken_;


  InputTags theSelectorMapNames;
  std::vector<edm::EDGetTokenT<edm::ValueMap<bool> > >  theSelectorMapTokens_;


  edm::InputTag theShowerMapName;
  edm::EDGetTokenT<edm::ValueMap<reco::MuonShower> > theShowerMapToken_;

  edm::InputTag theCosmicCompMapName;
  edm::EDGetTokenT<edm::ValueMap<unsigned int> > theCosmicIdMapToken_;
  edm::EDGetTokenT<edm::ValueMap<reco::MuonCosmicCompatibility> > theCosmicCompMapToken_;
  std::string theMuToMuMapName;

  MuPFIsoHelper *thePFIsoHelper;

  edm::EDGetTokenT<reco::MuonTimeExtraMap> timeMapCmbToken_;
  edm::EDGetTokenT<reco::MuonTimeExtraMap> timeMapDTToken_;
  edm::EDGetTokenT<reco::MuonTimeExtraMap> timeMapCSCToken_;




  std::vector<std::map<std::string,edm::InputTag> > pfIsoMapNames;
  std::vector<std::map<std::string,edm::EDGetTokenT<edm::ValueMap<double> > > > pfIsoMapTokens_;
  
};
#endif


 
