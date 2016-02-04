#ifndef __L1Analysis_L1AnalysisRecoMuon2_H__
#define __L1Analysis_L1AnalysisRecoMuon2_H__

//-------------------------------------------------------------------------------
// Created 05/03/2010 - A.C. Le Bihan
// 
//
// Original code : L1Trigger/L1TNtuples/L1RecoJetNtupleProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/JetID.h"
#include "L1AnalysisRecoMuon2DataFormat.h"

//muons
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/MuonReco/interface/MuonTime.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"


namespace L1Analysis
{
  class L1AnalysisRecoMuon2
  {
  public:
    L1AnalysisRecoMuon2();
    ~L1AnalysisRecoMuon2();
    
    //void Print(std::ostream &os = std::cout) const;
    void SetMuon(const edm::Event& event,
					   const edm::EventSetup& setup,
		const edm::Handle<reco::MuonCollection> muons, unsigned maxMuon);

    /* bool isMediumMuon(const reco::Muon & recoMu) ; */
    /* bool isLooseMuon (const reco::Muon & recoMu); */

    L1AnalysisRecoMuon2DataFormat * getData() {return &recoMuon_;}
    void Reset() {recoMuon_.Reset();}

  private :
    L1AnalysisRecoMuon2DataFormat recoMuon_;
  }; 
}
#endif


