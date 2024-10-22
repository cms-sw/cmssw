#ifndef DTDataIntegrityTask_H
#define DTDataIntegrityTask_H

/** \class DTDataIntegrityTask
 *
 * Class for DT Data Integrity
 * at Online DQM (Single Thread)
 * expected to monitor uROS
 * Class with MEs vs Time/LS
 *
 * \author Javier Fernandez (Uni. Oviedo) 
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "DataFormats/DTDigi/interface/DTuROSControlData.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <list>

namespace dtdi {
  struct LumiCache {
    int nEventsLS = 0;
  };
}  // namespace dtdi

class DTuROSROSData;
class DTuROSFEDData;
class DTTimeEvolutionHisto;

class DTDataIntegrityTask : public DQMOneEDAnalyzer<edm::LuminosityBlockCache<dtdi::LumiCache>> {
public:
  DTDataIntegrityTask(const edm::ParameterSet& ps);

  ~DTDataIntegrityTask() override;

  void TimeHistos(DQMStore::IBooker&, std::string histoType);

  void processuROS(DTuROSROSData& data, int fed, int uRos);
  void processFED(DTuROSFEDData& data, int fed);

  std::shared_ptr<dtdi::LumiCache> globalBeginLuminosityBlock(const edm::LuminosityBlock& ls,
                                                              const edm::EventSetup& es) const override;
  void globalEndLuminosityBlock(const edm::LuminosityBlock& ls, const edm::EventSetup& es) override;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  void bookHistos(DQMStore::IBooker&, const int fedMin, const int fedMax);
  void bookHistos(DQMStore::IBooker&, std::string folder, const int fed);
  void bookHistosuROS(DQMStore::IBooker&, const int fed, const int uRos);
  void bookHistosROS(DQMStore::IBooker&, const int wheel, const int ros);

  std::string topFolder(bool isFEDIntegrity) const;

  //conversions
  int theDDU(int crate, int slot, int link, bool tenDDU);
  int theROS(int slot, int link);

  //If you want info VS time histos
  bool doTimeHisto;

  int nevents;

  // Monitor Elements
  MonitorElement* nEventMonitor;
  // <histoType, <index , histo> >
  std::map<std::string, std::map<int, MonitorElement*>> fedHistos;
  // <histoType, histo> >
  std::map<std::string, std::map<int, MonitorElement*>> summaryHistos;
  // <key , histo> >
  std::map<unsigned int, MonitorElement*> urosHistos;

  //enum histoTypes for reduced map of MEs urosHistos
  // key = stringEnum*1000 + (fed-minFED)#*100 + (uROS-minuROS)#
  enum histoTypes { uROSEventLength = 0, uROSError = 1, TDCError = 4, TTSValues = 7 };

  // standard ME for monitoring of FED integrity
  MonitorElement* hFEDEntry;
  MonitorElement* hFEDFatal;

  //time histos for FEDs/uROS
  std::map<std::string, std::map<int, DTTimeEvolutionHisto*>> fedTimeHistos;
  // <key, histo> >
  std::map<unsigned int, DTTimeEvolutionHisto*> urosTimeHistos;
  //key =  (fed-minFED)#*100 + (uROS-minuROS)#

#ifdef EDM_ML_DEBUG
  int neventsFED;
  int neventsuROS;
#endif

  const int FEDIDmin;
  const int FEDIDmax;

  int errorX[6][12][5] = {{{0}}};  //5th is notOK flag and 6th is TDC Fatal; Second index is ROS. Last index is wheel
  int nLinksForFatal;  //Minumum number of Links/wheel with notOKFlag or TDC fatal errors to consider a FEDfatal event

  // Number of uROS per FED
  const int NuROS = 12;

  // flag to toggle the creation of only the summaries (for HLT running)
  int mode;
  std::string fedIntegrityFolder;

  // The label to retrieve the digis
  edm::EDGetTokenT<DTuROSFEDDataCollection> fedToken;
};

#endif
