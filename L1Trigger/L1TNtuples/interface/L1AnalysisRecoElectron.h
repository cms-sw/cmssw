#ifndef __L1Analysis_L1AnalysisRecoElectron_H__
#define __L1Analysis_L1AnalysisRecoElectron_H__

//-------------------------------------------------------------------------------
// Created 05/03/2010 - A.C. Le Bihan
// 
//
// Original code : L1Trigger/L1TNtuples/L1RecoJetNtupleProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include "L1AnalysisRecoElectronDataFormat.h"

//electrons
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"



namespace L1Analysis
{
  class L1AnalysisRecoElectron
  {
  public:
    L1AnalysisRecoElectron();
    ~L1AnalysisRecoElectron();
    
    //void Print(std::ostream &os = std::cout) const;
    void SetElectron(const edm::Event& event,
		     const edm::EventSetup& setup,
		     //const edm::Handle<edm::View<reco::GsfElectron>>& electrons,
		     const edm::Handle<reco::GsfElectronCollection> electrons,
		     const std::vector<edm::Handle<edm::ValueMap<bool> > > eleVIDDecisionHandles,
		     const unsigned& maxElectron);

      /*(const edm::Event& event,
		     const edm::EventSetup& setup,
		     const edm::Handle<reco::GsfElectronCollection> electrons,
		     const edm::Handle<reco::VertexCollection> vertices,
		     const edm::Handle<reco::BeamSpot>,
		     double Rho,
		     unsigned maxElectron);*/

    L1AnalysisRecoElectronDataFormat * getData() {return &recoElectron_;}
    void Reset() {recoElectron_.Reset();}

  private :
    L1AnalysisRecoElectronDataFormat recoElectron_;
  }; 
}
#endif


