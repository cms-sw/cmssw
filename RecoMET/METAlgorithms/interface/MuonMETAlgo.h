#ifndef RecoMET_MuonMETAlgo_h
#define RecoMET_MuonMETAlgo_h

/** \class MuonMETAlgo
 *
 * Correct MET for muons in the events.
 *
 * \version   1st Version August 30, 2007
 ************************************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "DataFormats/Common/interface/ValueMap.h" 
#include "RecoMET/METAlgorithms/interface/MuonMETInfo.h"


#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
class MuonMETAlgo 
{
 public:
  MuonMETAlgo();
  virtual ~MuonMETAlgo();
   
  reco::CaloMET makeMET(const reco::CaloMET& fMet, double fSumEt,
			const std::vector<CorrMETData>& fCorrections, 
			const reco::MET::LorentzVector&);
  reco::MET     makeMET(const reco::MET&, double fSumEt,
			const std::vector<CorrMETData>& fCorrections, 
			const reco::MET::LorentzVector& fP4);
  
  virtual void run(const edm::View<reco::Muon>& inputMuons,
		   const edm::ValueMap<reco::MuonMETCorrectionData>& vm_muCorrData,
		   const edm::View<reco::MET>& uncorMET,
		   reco::METCollection *corMET);
		   
  virtual void run(const edm::View<reco::Muon>& inputMuons,
		   const edm::ValueMap<reco::MuonMETCorrectionData>& vm_muCorrData,
		   const edm::View<reco::CaloMET>& uncorMET,
		   reco::CaloMETCollection *corMET);
		     
  
  void GetMuDepDeltas(const reco::Muon* inputMuon,
		      TrackDetMatchInfo& info,
		      bool useTrackAssociatorPositions,
		      bool useRecHits,
		      bool useHO,
		      double towerEtThreshold,
		      double& deltax, double& deltay, double Bfield);
  
  template <class T> void MuonMETAlgo_run(const edm::View<reco::Muon>& inputMuons,
					  const edm::ValueMap<reco::MuonMETCorrectionData>& vm_muCorrData,
					  const edm::View<T>& v_uncorMET,
					  std::vector<T>* v_corMET);
					  
  static void  correctMETforMuon(double& deltax, double& deltay,
				 double bfield, int muonCharge,
				 const math::XYZTLorentzVector& muonP4,
				 const math::XYZPoint& muonVertex,
				 MuonMETInfo&);
    
};

#endif // Type1MET_MuonMETAlgo_h

/*  LocalWords:  MuonMETAlgo
 */
