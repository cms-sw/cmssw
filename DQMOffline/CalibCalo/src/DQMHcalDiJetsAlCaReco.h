#ifndef DQMHcalDiJetsAlCaReco_H
#define DQMHcalDiJetsAlCaReco_H

/** \class DQMHcalPhiSymAlCaReco
 * *
 *  DQM Source for phi symmetry stream
 *
 *  \author Stefano Argiro'
 *          Andrea Gozzelino - Universita  e INFN Torino
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"


class DQMStore;
class MonitorElement;

class DQMHcalDiJetsAlCaReco : public edm::EDAnalyzer {

public:

  DQMHcalDiJetsAlCaReco( const edm::ParameterSet& );
  ~DQMHcalDiJetsAlCaReco();

protected:
   
  void beginJob();

  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& context) ;

  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  void endRun(const edm::Run& r, const edm::EventSetup& c);

  void endJob();

private:
 

  DQMStore*   dbe_;  
  int eventCounter_;  
      
//                        
// Monitor elements
//
  MonitorElement * hiDistrRecHitEnergyEBEE_;
  MonitorElement * hiDistrRecHitEnergyHBHE_;
  MonitorElement * hiDistrRecHitEnergyHF_;
  MonitorElement * hiDistrRecHitEnergyHO_;

  MonitorElement * hiDistrProbeJetEnergy_;
  MonitorElement * hiDistrProbeJetEta_;
  MonitorElement * hiDistrProbeJetPhi_;

  MonitorElement * hiDistrTagJetEnergy_;
  MonitorElement * hiDistrTagJetEta_;
  MonitorElement * hiDistrTagJetPhi_;
  
  MonitorElement * hiDistrEtThirdJet_;


  /// object to monitor
  edm::EDGetTokenT<reco::CaloJetCollection> jets_;
  edm::EDGetTokenT<EcalRecHitCollection>    ec_;
  edm::EDGetTokenT<HBHERecHitCollection>    hbhe_;
  edm::EDGetTokenT<HORecHitCollection>      ho_;
  edm::EDGetTokenT<HFRecHitCollection>      hf_;
  
  /// DQM folder name
  std::string folderName_; 
 
  /// Write to file 
  bool saveToFile_;

  /// Output file name if required
  std::string fileName_;

  bool allowMissingInputs_;

};

#endif

