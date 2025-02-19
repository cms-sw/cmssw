#ifndef HLTAlCaMonEcalPhiSym_H
#define HLTAlCaMonEcalPhiSym_H

/** \class HLTAlCaMonEcalPhiSym
 * *
 *  DQM Source for phi symmetry stream
 *
 *  $Date: 2009/10/15 11:31:28 $
 *  $Revision: 1.2 $
 *  \author Stefano Argiro'
 *          Andrea Gozzelino - Universita%Gï¿½%@ e INFN Torino
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

class DQMStore;
class MonitorElement;

class HLTAlCaMonEcalPhiSym : public edm::EDAnalyzer {

public:

  HLTAlCaMonEcalPhiSym( const edm::ParameterSet& );
  ~HLTAlCaMonEcalPhiSym();

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
                        

  /// Distribution of rechits in iPhi   
  MonitorElement * hiPhiDistrEB_;
  int hiPhiDistrEB_nbin_;
  double  hiPhiDistrEB_min_;
  double  hiPhiDistrEB_max_;

  /// Distribution of rechits in iEta
  MonitorElement * hiEtaDistrEB_;
  int hiEtaDistrEB_nbin_; 
  double hiEtaDistrEB_min_;
  double hiEtaDistrEB_max_;

  /// Energy Distribution of rechits  
  MonitorElement * hRechitEnergyEB_;
  int hRechitEnergyEB_nbin_;
  double hRechitEnergyEB_min_;
  double hRechitEnergyEB_max_;

  /// Distribution of total event energy
  MonitorElement * hEventEnergyEB_;
  int hEventEnergyEB_nbin_;
  double hEventEnergyEB_min_;
  double hEventEnergyEB_max_;

  /// Distribution of number of RecHits
  MonitorElement * hNRecHitsEB_;
  int hNRecHitsEB_nbin_;
  double hNRecHitsEB_min_;
  double hNRecHitsEB_max_;

  /// Distribution of Mean energy per rechit
  MonitorElement * hMeanRecHitEnergyEB_;
  int hMeanRecHitEnergyEB_nbin_;
  double hMeanRecHitEnergyEB_min_;
  double hMeanRecHitEnergyEB_max_;

  /// Energy Distribution of rechits  
  MonitorElement * hRechitEnergyEE_;
  int hRechitEnergyEE_nbin_;
  double hRechitEnergyEE_min_;
  double hRechitEnergyEE_max_;

  /// Distribution of total event energy
  MonitorElement * hEventEnergyEE_;
  int hEventEnergyEE_nbin_;
  double hEventEnergyEE_min_;
  double hEventEnergyEE_max_;

  /// Distribution of number of RecHits
  MonitorElement * hNRecHitsEE_;
  int hNRecHitsEE_nbin_;
  double hNRecHitsEE_min_;
  double hNRecHitsEE_max_;

  /// Distribution of Mean energy per rechit
  MonitorElement * hMeanRecHitEnergyEE_;
  int    hMeanRecHitEnergyEE_nbin_;
  double hMeanRecHitEnergyEE_min_;
  double hMeanRecHitEnergyEE_max_;

  /// object to monitor
  edm::InputTag productMonitoredEB_;

 /// object to monitor
  edm::InputTag productMonitoredEE_;


  /// Monitor every prescaleFactor_ events
  unsigned int prescaleFactor_;
  
  /// DQM folder name
  std::string folderName_; 
 
  /// Write to file 
  bool saveToFile_;

  /// Output file name if required
  std::string fileName_;

};

#endif

