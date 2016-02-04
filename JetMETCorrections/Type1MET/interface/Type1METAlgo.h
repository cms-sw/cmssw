#ifndef Type1METAlgo_h
#define Type1METAlgo_h

/** \class Type1METAlgo
 *
 * Calculates MET for given input CaloTower collection.
 * Does corrections based on supplied parameters.
 *
 * \author M. Schmitt, R. Cavanaugh, The University of Florida
 *
 * \version   1st Version May 14, 2005
 ************************************************************/

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoMET/METAlgorithms/interface/MuonMETInfo.h"

#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"

class JetCorrector;

class Type1METAlgo 
{
 public:
  Type1METAlgo();
  virtual ~Type1METAlgo();
  virtual void run(const reco::PFMETCollection&, 
		   const JetCorrector&,
		   const reco::PFJetCollection&, 
		   double, double, double, double, double, bool, bool,
                   const edm::View<reco::Muon>& ,
                   const edm::ValueMap<reco::MuonMETCorrectionData>& ,
		   reco::METCollection *);
  virtual void run(const reco::PFMETCollection&, 
		   const JetCorrector&,
		   const pat::JetCollection&, 
		   double, double, double, double, double, bool, bool,
                   const edm::View<reco::Muon>& ,
                   const edm::ValueMap<reco::MuonMETCorrectionData>& ,
		   reco::METCollection *);
  virtual void run(const reco::CaloMETCollection&, 
		   const JetCorrector&,
		   const reco::CaloJetCollection&, 
		   double, double, double, double, double, bool, bool,
                   const edm::View<reco::Muon>& ,
                   const edm::ValueMap<reco::MuonMETCorrectionData>& ,
		   reco::CaloMETCollection*);
};

#endif // Type1METAlgo_h

/*  LocalWords:  Type1METAlgo
 */
