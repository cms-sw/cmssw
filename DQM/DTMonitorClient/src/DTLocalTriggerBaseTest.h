#ifndef DTLocalTriggerBaseTest_H
#define DTLocalTriggerBaseTest_H


/** \class DTLocalTriggerBaseTest
 * *
 *  DQM Base for TriggerTests
 *
 *  $Date: 2011/06/10 13:50:12 $
 *  $Revision: 1.10 $
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

class DTLocalTriggerBaseTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTLocalTriggerBaseTest() {};
  
  /// Destructor
  virtual ~DTLocalTriggerBaseTest();

protected:

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& context);

  /// Perform begin lumiblock operations
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Perform client diagnostic in online
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context);

  /// Perform client diagnostic in offline
  void endRun(edm::Run const& run, edm::EventSetup const& context);

  /// EndJob
  void endJob();

  /// Perform client analysis
  virtual void runClientDiagnostic() = 0;

  /// Book the new MEs (for each sector)
  void bookSectorHistos( int wheel, int sector, std::string hTag, std::string folder="" );

  /// Book the new MEs (for each wheel)
  void bookWheelHistos( int wheel, std::string hTag, std::string folder="" );

  /// Book the new MEs (CMS summary)
  void bookCmsHistos( std::string hTag, std::string folder="" , bool isGlb = false);

  /// Calculate phi range for histograms
  std::pair<float,float> phiRange(const DTChamberId& id);

  /// Convert ME to Histogram fo type T
  template <class T>  T* getHisto(MonitorElement* me);

  /// Set configuration variables
  void setConfig(const edm::ParameterSet& ps, std::string name);

/*   /// Set labels to wheel plots (Phi) */
/*   void setLabelPh(MonitorElement* me); */

/*   /// Set labels to theta plots (Theta) */
/*   void setLabelTh(MonitorElement* me);  */

  /// Create fullname from histo partial name
  std::string fullName(std::string htype);

  /// Get the ME name (by chamber)
  std::string getMEName(std::string histoTag, std::string subfolder, const DTChamberId& chambid);

 /// Get the ME name (by wheel)
  std::string getMEName(std::string histoTag, std::string subfolder, int wh);
  
  /// Get top folder name
  inline std::string & topFolder(bool isDCC) { return isDCC ? baseFolderDCC : baseFolderDDU; } ;
  
  /// Get message logger name
  inline std::string category() { return "DTDQM|DTMonitorClient|" + testName + "Test"; } ;
  


  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  std::string testName;
  std::vector<std::string> trigSources;
  std::vector<std::string> hwSources;

  DQMStore* dbe;
  std::string sourceFolder;
  edm::ParameterSet parameters;
  bool runOnline;
  std::string baseFolderDCC;
  std::string baseFolderDDU;
  std::string trigSource;
  std::string hwSource;
  edm::ESHandle<DTGeometry> muonGeom;
  std::map<int,std::map<std::string,MonitorElement*> > secME;
  std::map<int,std::map<std::string,MonitorElement*> > whME;
  std::map<std::string,MonitorElement*> cmsME;

};


template <class T>
T* DTLocalTriggerBaseTest::getHisto(MonitorElement* me) {
  return me ? dynamic_cast<T*>(me->getRootObject()) : 0;
}

#endif


