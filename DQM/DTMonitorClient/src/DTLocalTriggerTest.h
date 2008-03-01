#ifndef DTLocalTriggerTest_H
#define DTLocalTriggerTest_H


/** \class DTLocalTriggerTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/11/19 14:30:20 $
 *  $Revision: 1.8 $
 *  \author  C. Battilana S. Marcellini - INFN Bologna
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <boost/cstdint.hpp>
#include <string>
#include <map>

class DTChamberId;
class DTGeometry;
class TH1F;
class TH2F;
class TH1D;

class DTLocalTriggerTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTLocalTriggerTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTLocalTriggerTest();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// Book the new MEs (for each sector)
  void bookSectorHistos(int wheel, int sector, std::string folder, std::string htype );

  /// Book the new MEs (for each wheel)
  void bookWheelHistos(int wheel, std::string folder, std::string htype );

  /// Book the new MEs (for each chamber)
  void bookChambHistos(DTChamberId chambId, std::string htype );

  /// Calculate phi range for histograms
  std::pair<float,float> phiRange(const DTChamberId& id);

  /// Compute efficiency plots
  void makeEfficiencyME(TH1D* numerator, TH1D* denominator, MonitorElement* result);

  /// Compute 2D efficiency plots
  void makeEfficiencyME2D(TH2F* numerator, TH2F* denominator, MonitorElement* result);

  /// Convert ME to Histogram fo type T
  template <class T>  T* getHisto(MonitorElement* me);

  /// Set labels to wheel plots (Phi)
  void setLabelPh(MonitorElement* me);

  /// Set labels to theta plots (Theta)
  void setLabelTh(MonitorElement* me); 

  /// Get the ME name
  std::string getMEName(std::string histoTag, std::string subfolder, const DTChamberId & chambid);

  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);



 private:

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;

  DQMStore* dbe;
  std::string sourceFolder;
  edm::ParameterSet parameters;
  std::string hwSource;
  edm::ESHandle<DTGeometry> muonGeom;
  std::map<int,std::map<std::string,MonitorElement*> >      secME;
  std::map<int,std::map<std::string,MonitorElement*> >      whME;
  std::map<uint32_t,std::map<std::string,MonitorElement*> > chambME;


};

#endif
