/*HLTTauRefInfo
Producer that creates LorentzVector Collections
from offline reconstructed quantities to be used
in Offline Trigger DQM etc
*/

#ifndef HLTTauRefInfo_h
#define HLTTauRefInfo_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>
#include <string>

class HLTTauRefInfo : public edm::EDProducer {

public:
  explicit HLTTauRefInfo(const edm::ParameterSet&);
  ~HLTTauRefInfo();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  typedef math::XYZTLorentzVectorD LorentzVector;
  typedef std::vector<LorentzVector> LorentzVectorCollection;

  edm::InputTag PFTaus_;
  edm::InputTag PFTauDis_;
  bool doPFTaus_;
  double ptMinPFTau_;

  edm::InputTag CaloTaus_;
  edm::InputTag CaloTauDis_;
  bool doCaloTaus_;
  double ptMinCaloTau_;

  edm::InputTag Electrons_;
  bool doElectrons_;
  double ptMinElectron_;

  edm::InputTag Muons_;
  bool doMuons_;
  double ptMinMuon_;


  edm::InputTag Jets_;
  bool doJets_;
  double ptMinJet_;


  edm::InputTag MC_;
  bool doMC_;
  double ptMinMCTau_;
  double ptMinMCElectron_;
  double ptMinMCMuon_;
  int m_PDG_;

  double etaMax;


  void doPFTaus(edm::Event&,const edm::EventSetup&);
  void doCaloTaus(edm::Event&,const edm::EventSetup&);
  void doMuons(edm::Event&,const edm::EventSetup&);
  void doElectrons(edm::Event&,const edm::EventSetup&);
  void doJets(edm::Event&,const edm::EventSetup&);
  void doMC(edm::Event&,const edm::EventSetup&);





};

#endif
