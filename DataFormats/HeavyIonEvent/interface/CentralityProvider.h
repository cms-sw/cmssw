#ifndef __CentralityProvider_h__
#define __CentralityProvider_h__
#include <iostream>

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

class CentralityProvider : public CentralityBins {

 public:
  CentralityProvider(const edm::EventSetup& iSetup);
  ~CentralityProvider(){;}

  enum VariableType {HFtowers, HFtowersPlus, HFtowersMinus, HFtowersTrunc, HFtowersPlusTrunc, HFtowersMinusTrunc, HFhits, PixelHits, PixelTracks, Tracks, EB, EE, Missing};

  int getNbins() const {return table_.size();}
  double centralityValue() const;
  int getBin() const {return CentralityBins::getBin(centralityValue());}
  float lowEdge() const { return lowEdgeOfBin(getBin());}
  float NpartMean() const { return NpartMeanOfBin(getBin());}
  float NpartSigma() const { return NpartSigmaOfBin(getBin());}
  float NcollMean() const { return NcollMeanOfBin(getBin());}
  float NcollSigma()const { return NcollSigmaOfBin(getBin());}
  float NhardMean() const { return NhardMeanOfBin(getBin());}
  float NhardSigma() const { return NhardSigmaOfBin(getBin());}
  float bMean() const { return bMeanOfBin(getBin());}
  float bSigma() const { return bSigmaOfBin(getBin());}
  void newRun(const edm::EventSetup& iSetup);
  void newEvent(const edm::Event& ev,const edm::EventSetup& iSetup);
  void print();
  const CentralityBins* table() const {return this;}
  const reco::Centrality* raw() const {return chandle_.product();}

 private:
  edm::InputTag tag_;
  std::string centralityVariable_;
  std::string centralityLabel_;
  std::string centralityMC_;
  unsigned int prevRun_;
  mutable edm::Handle<reco::Centrality> chandle_;
  VariableType varType_;
};

#endif











