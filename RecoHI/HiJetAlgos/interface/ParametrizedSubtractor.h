#ifndef __ParametrizedSubtractor_h_
#define __ParametrizedSubtractor_h_

#include <vector>

#include "RecoJets/JetProducers/interface/PileUpSubtractor.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "TH1D.h"

#include "TF1.h"

class CentralityBins;

class ParametrizedSubtractor : public PileUpSubtractor {
 public:
  ParametrizedSubtractor(const edm::ParameterSet& iConfig, edm::ConsumesCollector && iC);
   void setupGeometryMap(edm::Event& iEvent,const edm::EventSetup& iSetup) override;
   void calculatePedestal( std::vector<fastjet::PseudoJet> const & coll ) override;
   void subtractPedestal(std::vector<fastjet::PseudoJet> & coll) override;
   void calculateOrphanInput(std::vector<fastjet::PseudoJet> & orphanInput) override;
    void offsetCorrectJets() override;
    double getMeanAtTower(const reco::CandidatePtr & in) const override;
    double getSigmaAtTower(const reco::CandidatePtr & in) const override;
    double getPileUpAtTower(const reco::CandidatePtr & in) const override;
    double getEt(const reco::CandidatePtr & in) const;
    double getEta(const reco::CandidatePtr & in) const;

    void rescaleRMS(double s);
    double getPU(int ieta, bool addMean, bool addSigma) const;
    ~ParametrizedSubtractor() override{;}

    bool sumRecHits_;
    bool interpolate_;
    bool dropZeroTowers_;
    int bin_;
    double centrality_;
    const CentralityBins * cbins_;
    edm::EDGetTokenT<reco::Centrality> centTag_;
    std::vector<TH1D*> hEta;
    std::vector<TH1D*> hEtaMean;
    std::vector<TH1D*> hEtaRMS;

    TF1* fPU;    
    TF1* fMean;
    TF1* fRMS;
    TH1D* hC;
};

#endif
