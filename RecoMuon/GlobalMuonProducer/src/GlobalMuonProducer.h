#ifndef RecoMuon_GlobalMuonProducer_H
#define RecoMuon_GlobalMuonProducer_H

/**  \class GlobalMuonProducer
 * 
 *   Global muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a standalone reonstructed muon.
 *
 *
 *   $Date: 2006/05/19 15:23:20 $
 *   $Revision: 1.2 $
 *
 *   \author  R.Bellan - INFN TO
 */

#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrackFinder;

class GlobalMuonProducer : public edm::EDProducer {

  public:

    /// constructor with config
    GlobalMuonProducer(const edm::ParameterSet&);
  
    /// destructor
    virtual ~GlobalMuonProducer(); 
  
    /// reconstruct muons
    virtual void produce(edm::Event&, const edm::EventSetup&);

  
  private:
    
    /// Seed STA Label
    std::string theSTACollectionLabel;
  
    MuonTrackFinder* theTrackFinder;
 
};

#endif
