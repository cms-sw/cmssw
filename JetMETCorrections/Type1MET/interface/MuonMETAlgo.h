#ifndef MuonMETAlgo_h
#define MuonMETAlgo_h

/** \class MuonMETAlgo
 *
 * Correct MET for muons in the events.
 *
 * \version   1st Version August 30, 2007
 ************************************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

class MuonMETAlgo 
{
 public:
  MuonMETAlgo();
  virtual ~MuonMETAlgo();
  virtual void run(edm::Event&, const edm::EventSetup&,
		   const reco::METCollection&, 
		   const edm::View<reco::Muon>&, 
		   double, double, double, double,
		   int, double, double,
		   bool, TrackDetectorAssociator&, TrackAssociatorParameters&,
		   reco::METCollection *);
  virtual void run(edm::Event&, const edm::EventSetup&,
		   const reco::CaloMETCollection&, 
		   const edm::View<reco::Muon>&, 
		   double, double, double, double,
		   int, double, double,
		   bool, TrackDetectorAssociator&, TrackAssociatorParameters&,
		   reco::CaloMETCollection*);
};

#endif // MuonMETAlgo_h

/*  LocalWords:  MuonMETAlgo
 */
