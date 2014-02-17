#ifndef RecoMuon_MuonIdentification_MuonProducer_H
#define RecoMuon_MuonIdentification_MuonProducer_H

/** \class MuonProducer
 *  Producer meant for the Post PF reconstruction.
 *
 * This class takes the muon collection produced before the PF is run (muons1Step) and the information calculated after that 
 * the entire event has been reconstructed. The collections produced here are meant to be used for the final analysis (or as PAT input).
 * The previous muon collection is meant to be transient.
 *
 *  $Date: 2012/07/21 00:38:17 $
 *  $Revision: 1.15 $
 *  \author R. Bellan - UCSB <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"

namespace reco {class Track;}
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

class MuPFIsoHelper;


class MuonProducer : public edm::EDProducer {
public:

  /// Constructor
  MuonProducer(const edm::ParameterSet&);

  /// Destructor
  virtual ~MuonProducer();

  /// reconstruct muons
  virtual void produce(edm::Event&, const edm::EventSetup&);


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
  edm::InputTag thePFCandLabel;

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

  InputTags theSelectorMapNames;

  edm::InputTag theShowerMapName;
  edm::InputTag theCosmicCompMapName;

  std::string theMuToMuMapName;

  MuPFIsoHelper *thePFIsoHelper;

  std::vector<std::map<std::string,edm::InputTag> > pfIsoMapNames;
  
};
#endif


 
