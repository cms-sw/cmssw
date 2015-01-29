#ifndef DTMonitorClient_DTRunConditionVarClient_H
#define DTMonitorClient_DTRunConditionVarClient_H


/** \class DTRunConditionVarClient
 *
 * Description:
 *  
 *
 * \author : Paolo Bellan, Antonio Branca
 * $date   : 23/09/2011 15:42:04 CET $
 *
 * Modification:
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "CondFormats/DTObjects/interface/DTMtime.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DQMServices/Core/interface/DQMEDHarvester.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTChamberId;
class DTLayerId;

class DTRunConditionVarClient: public DQMEDHarvester{

  public:

    /// Constructor
    DTRunConditionVarClient(const edm::ParameterSet& ps);

    /// Destructor
    virtual ~DTRunConditionVarClient();

  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  protected:

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

    /// book the report summary

    void bookWheelHistos(DQMStore::IBooker &,std::string histoType, std::string subfolder, 
                         int wh, int nbins, float min, float max, bool isVDCorr=false);

    /// DQM Client Diagnostic
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const &);


  float varQuality(float var, float maxGood, float minBad);
  void percDevVDrift(DTChamberId indexCh, float meanVD, float sigmaVD, float& devVD, float& errdevVD);

  private:

    MonitorElement* getChamberHistos(DQMStore::IGetter & ,const DTChamberId&, std::string);

    int nevents;      

    float minRangeVDrift;
    float maxRangeVDrift; 
    float minRangeT0;
    float maxRangeT0;

    float maxGoodVDriftDev;
    float minBadVDriftDev; 
    float maxGoodT0;       
    float minBadT0;       

    float maxGoodVDriftSigma;
    float minBadVDriftSigma;
    float maxGoodT0Sigma;
    float minBadT0Sigma;

    edm::ESHandle<DTMtime> mTime;
    const DTMtime* mTimeMap_;

  bool bookingdone;

    MonitorElement* glbVDriftSummary;
    MonitorElement* glbT0Summary;

    std::map<int, std::map<std::string, MonitorElement*> > wheelHistos;
    std::map<std::string, MonitorElement *> summaryHistos;
    std::map<std::string, MonitorElement *> allwheelHistos;

};

#endif
