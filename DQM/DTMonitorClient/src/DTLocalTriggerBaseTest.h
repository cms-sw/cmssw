#ifndef DTLocalTriggerBaseTest_H
#define DTLocalTriggerBaseTest_H

/** \class DTLocalTriggerBaseTest
 * *
 *  DQM Base for TriggerTests
 *
 *  \author  C. Battilana S. Marcellini - INFN Bologna
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
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
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DQMServices/Core/interface/DQMEDHarvester.h>

#include <string>
#include <map>

class DTChamberId;
class DTGeometry;
class TH1F;
class TH2F;
class TH1D;

class DTLocalTriggerBaseTest : public DQMEDHarvester {
public:
  /// Constructor
  DTLocalTriggerBaseTest(){};

  /// Destructor
  ~DTLocalTriggerBaseTest() override;

protected:
  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& context) override;

  /// Perform client diagnostic in online
  void dqmEndLuminosityBlock(DQMStore::IBooker&,
                             DQMStore::IGetter&,
                             edm::LuminosityBlock const&,
                             edm::EventSetup const&) override;

  /// Perform client diagnostic in offline
  void endRun(edm::Run const& run, edm::EventSetup const& context) override;

  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

  /// Perform client analysis
  virtual void runClientDiagnostic(DQMStore::IBooker&, DQMStore::IGetter&) = 0;

  /// Book the new MEs (for each sector)
  void bookSectorHistos(DQMStore::IBooker&, int wheel, int sector, std::string hTag, std::string folder = "");

  /// Book the new MEs (for each wheel)
  void bookWheelHistos(DQMStore::IBooker&, int wheel, std::string hTag, std::string folder = "");

  /// Book the new MEs (CMS summary)
  void bookCmsHistos(DQMStore::IBooker&, std::string hTag, std::string folder = "", bool isGlb = false);

  /// Calculate phi range for histograms
  std::pair<float, float> phiRange(const DTChamberId& id);

  /// Convert ME to Histogram fo type T
  template <class T>
  T* getHisto(MonitorElement* me);

  /// Set configuration variables
  void setConfig(const edm::ParameterSet& ps, std::string name);

  /// Create fullname from histo partial name
  std::string fullName(std::string htype);

  /// Get the ME name (by chamber)
  std::string getMEName(std::string histoTag, std::string subfolder, const DTChamberId& chambid);

  /// Get the ME name (by wheel)
  std::string getMEName(std::string histoTag, std::string subfolder, int wh);

  /// Get top folder name
  inline std::string& topFolder() { return baseFolderTM; };

  /// Get message logger name
  inline std::string category() { return "DTDQM|DTMonitorClient|" + testName + "Test"; };

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  std::string testName;
  std::vector<std::string> trigSources;
  std::vector<std::string> hwSources;

  std::string sourceFolder;
  edm::ParameterSet parameters;
  bool runOnline;
  std::string baseFolderTM;
  std::string trigSource;
  std::string hwSource;
  edm::ESHandle<DTGeometry> muonGeom;
  std::map<int, std::map<std::string, MonitorElement*> > secME;
  std::map<int, std::map<std::string, MonitorElement*> > whME;
  std::map<std::string, MonitorElement*> cmsME;
};

template <class T>
T* DTLocalTriggerBaseTest::getHisto(MonitorElement* me) {
  return me ? dynamic_cast<T*>(me->getRootObject()) : nullptr;
}

#endif
