#ifndef DQMSourcePhiSym_H
#define DQMSourcePhiSym_H

/** \class DQMSourcePhiSym
 * *
 *  DQM Source for phi symmetry stream
 *
 *  $Date: 2008/04/28 22:06:10 $
 *  $Revision: 1.2 $
 *  \author Stefano Argiro'
 *          Andrea Gozzelino - Universita  e INFN Torino
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

class DQMStore;
class MonitorElement;

class DQMSourcePhiSym : public edm::EDAnalyzer {

public:

  DQMSourcePhiSym( const edm::ParameterSet& );
  ~DQMSourcePhiSym();

protected:
   
  void beginJob(const edm::EventSetup& c);

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

  /// Distribution of rechits in iEta
  MonitorElement * hiEtaDistrEB_;

  /// Energy Distribution of rechits  
  MonitorElement * hRechitEnergyEB_;

  /// Distribution of total event energy
  MonitorElement * hEventEnergyEB_;
  
  /// Distribution of number of RecHits
  MonitorElement * hNRecHitsEB_;

  /// Distribution of Mean energy per rechit
  MonitorElement * hMeanRecHitEnergyEB_;


  /// Energy Distribution of rechits  
  MonitorElement * hRechitEnergyEE_;

  /// Distribution of total event energy
  MonitorElement * hEventEnergyEE_;
  
  /// Distribution of number of RecHits
  MonitorElement * hNRecHitsEE_;

  /// Distribution of Mean energy per rechit
  MonitorElement * hMeanRecHitEnergyEE_;

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

