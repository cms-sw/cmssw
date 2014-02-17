#ifndef RecoMuon_StandAloneMuonProducer_StandAloneMuonProducer_H
#define RecoMuon_StandAloneMuonProducer_StandAloneMuonProducer_H

/**  \class StandAloneMuonProducer
 * 
 *   StandAlone muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from internal seeds (L2 muon track segments).
 *
 *
 *   $Date: 2010/02/11 00:14:31 $
 *   $Revision: 1.9 $
 *
 *   \author  R.Bellan - INFN TO
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrackFinder;
class MuonServiceProxy;

class StandAloneMuonProducer : public edm::EDProducer {

 public:

  /// constructor with config
  StandAloneMuonProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~StandAloneMuonProducer(); 
  
  /// reconstruct muons
  virtual void produce(edm::Event&, const edm::EventSetup&);
    
 private:
  
  /// MuonSeed Collection Label
  edm::InputTag theSeedCollectionLabel;
 
  /// the track finder
  MuonTrackFinder* theTrackFinder; //It isn't the same as in ORCA

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;

  std::string theAlias;

  void setAlias( std::string alias ){
    alias.erase( alias.size() - 1, alias.size() );
    theAlias=alias;
  }
};

#endif
