#ifndef DQMEcalCalibConstants_H
#define DQMEcalCalibConstants_H

/** \class DQMEcalCalibConstants
 * *
 *  DQM Source to monitor ecal calibration constants
 *
 *  $Date: 2008/08/13 09:20:27 $
 *  $Revision: 1.1 $
 *  \author Stefano Argiro
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

class DQMStore;
class MonitorElement;

class DQMEcalCalibConstants : public edm::EDAnalyzer {

public:

  DQMEcalCalibConstants( const edm::ParameterSet& );
  ~DQMEcalCalibConstants();

protected:
   
  void beginJob(const edm::EventSetup& c);

  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  void analyze(const edm::Event& e, const edm::EventSetup& c) {};

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			      const edm::EventSetup& context) {}

  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			    const edm::EventSetup& c){}

  void endRun(const edm::Run& r, const edm::EventSetup& c){}

  void endJob();

private:
 

  DQMStore*   dbe_;  
  
  /// Constant distribution in EB
  MonitorElement * cDistrEB_;
  /// Constant distribution in EE
  MonitorElement * cDistrEE_;
  /// Constant map EB
  MonitorElement * cMapEB_;
  /// Constant map EE
  MonitorElement * cMapEE_;  

  
  // comparisons 

  /// compare constants distribution
  MonitorElement *  compCDistrEB_;
  /// compare constants map
  MonitorElement *  compCMapEB_;
  /// compare constants eta trend
  MonitorElement *  compCEtaTrendEB_;
  /// compare constants eta profile
  MonitorElement *  compCEtaProfileEB_ ;
  /// compare constants distro in Module 
  MonitorElement *  compCDistrM1_ ;
  ///
  MonitorElement *  compCDistrM2_ ;
  ///
  MonitorElement *  compCDistrM3_ ;
  ///
  MonitorElement *  compCDistrM4_ ;

  // Same for EE+

  /// 
  MonitorElement *  compCDistrEEP_;
  ///
  MonitorElement *  compCMapEEP_;
  ///
  MonitorElement *  compCEtaTrendEEP_;
  ///
  MonitorElement *  compCEtaProfileEEP_ ;
  /// 
  MonitorElement *  compCDistrR1P_ ;
  ///
  MonitorElement *  compCDistrR2P_ ;
  ///
  MonitorElement *  compCDistrR3P_ ;
  ///
  MonitorElement *  compCDistrR4P_ ;
  ///
  MonitorElement *  compCDistrR5P_ ;

  // Same for EE-

  /// 
  MonitorElement *  compCDistrEEM_;
  ///
  MonitorElement *  compCMapEEM_;
  ///
  MonitorElement *  compCEtaTrendEEM_;
  ///
  MonitorElement *  compCEtaProfileEEM_ ;
  /// 
  MonitorElement *  compCDistrR1M_ ;
  ///
  MonitorElement *  compCDistrR2M_ ;
  ///
  MonitorElement *  compCDistrR3M_ ;
  ///
  MonitorElement *  compCDistrR4M_ ;
  ///
  MonitorElement *  compCDistrR5M_ ;

  /// DQM folder name
  std::string folderName_; 
 
  /// Write to file 
  bool saveToFile_;

  /// Output file name if required
  std::string fileName_;

  /// DB to be validated
  std::string DBlabel_; 
 
  /// reference DB label
  std::string RefDBlabel_;

  

};

#endif

