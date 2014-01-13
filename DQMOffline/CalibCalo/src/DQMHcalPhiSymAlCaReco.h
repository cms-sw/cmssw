#ifndef DQMHcalPhiSymAlCaReco_H
#define DQMHcalPhiSymAlCaReco_H

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

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"


class DQMStore;
class MonitorElement;

class DQMHcalPhiSymAlCaReco : public edm::EDAnalyzer {

public:

  DQMHcalPhiSymAlCaReco( const edm::ParameterSet& );
  ~DQMHcalPhiSymAlCaReco();

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
  MonitorElement * hiDistrMBPl2D_;
  MonitorElement * hiDistrNoisePl2D_;
  MonitorElement * hiDistrMBMin2D_;
  MonitorElement * hiDistrNoiseMin2D_;

  MonitorElement * hiDistrMB2Pl2D_;
  MonitorElement * hiDistrNoise2Pl2D_;
  MonitorElement * hiDistrMB2Min2D_;
  MonitorElement * hiDistrNoise2Min2D_;

  MonitorElement * hiDistrVarMBPl2D_;
  MonitorElement * hiDistrVarNoisePl2D_;
  MonitorElement * hiDistrVarMBMin2D_;
  MonitorElement * hiDistrVarNoiseMin2D_;

  MonitorElement * hiDistrHBHEsize1D_;
  MonitorElement * hiDistrHFsize1D_;

  MonitorElement * hFEDsize;
  MonitorElement * hHcalIsZS;
  MonitorElement * hL1Id;
  
  int hiDistr_y_nbin_;
  int hiDistr_x_nbin_;
  double  hiDistr_y_min_;
  double  hiDistr_y_max_;
  double  hiDistr_x_min_;
  double  hiDistr_x_max_;

  int hiDistr_r_nbin_;
  double ihbhe_size_;
  double ihf_size_;


  /// object to monitor
  
  edm::EDGetTokenT<HBHERecHitCollection>  hbherecoMB;
  edm::InputTag  horecoMB;
  edm::EDGetTokenT<HFRecHitCollection>  hfrecoMB;
  
  edm::EDGetTokenT<HBHERecHitCollection>  hbherecoNoise;
  edm::InputTag  horecoNoise;
  edm::EDGetTokenT<HFRecHitCollection>  hfrecoNoise;

  edm::EDGetTokenT<FEDRawDataCollection> rawInLabel_;
  
  /// DQM folder name
  std::string folderName_; 
 
  /// Write to file 
  bool saveToFile_;

  //period of ZS
  unsigned int period_;

  /// Output file name if required
  std::string fileName_;

};

#endif

