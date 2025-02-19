#ifndef DTMonitorClient_DTRunConditionVarClient_H
#define DTMonitorClient_DTRunConditionVarClient_H


/** \class DTRunConditionVarClient
 *
 * Description:
 *  
 *
 * \author : Paolo Bellan, Antonio Branca
 * $date   : 23/09/2011 15:42:04 CET $
 * $Revision: 1.3 $
 *
 * Modification:
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

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTChamberId;
class DTLayerId;

class DTRunConditionVarClient: public edm::EDAnalyzer{

  public:

    /// Constructor
    DTRunConditionVarClient(const edm::ParameterSet& ps);

    /// Destructor
    virtual ~DTRunConditionVarClient();

  protected:

    void beginJob();
    void analyze(const edm::Event& e, const edm::EventSetup& c);
    void endJob();

    /// book the report summary
    void bookWheelHistos(std::string histoType, std::string subfolder, int wh, int nbins, float min, float max, bool isVDCorr=false);

    /// DQM Client Diagnostic
    void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context);
    void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

    void beginRun(const edm::Run& run, const edm::EventSetup& setup);
    void endRun(edm::Run const& run, edm::EventSetup const& c);

    // 
    float varQuality(float var, float maxGood, float minBad);

    //
    void percDevVDrift(DTChamberId indexCh, float meanVD, float sigmaVD, float& devVD, float& errdevVD);

  private:

    MonitorElement* getChamberHistos(const DTChamberId&, std::string);

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

    DQMStore* theDbe;

    MonitorElement* glbVDriftSummary;
    MonitorElement* glbT0Summary;

    std::map<int, std::map<std::string, MonitorElement*> > wheelHistos;
    std::map<std::string, MonitorElement *> summaryHistos;
    std::map<std::string, MonitorElement *> allwheelHistos;

};

#endif
