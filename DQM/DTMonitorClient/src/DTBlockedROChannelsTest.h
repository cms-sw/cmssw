#ifndef DTMonitorModule_DTBlockedROChannelsTest_H
#define DTMonitorModule_DTBlockedROChannelsTest_H

/** \class DTBlockedROChannelsTest
 * *
 *  DQM Client to Summarize LS by LS the status of the Read-Out channels.
 *
 *  $Date: 2012/03/13 09:00:51 $
 *  $Revision: 1.6 $
 *  \author G. Cerminara - University and INFN Torino
 *   
 */
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/LuminosityBlock.h>
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

class DQMStore;
class MonitorElement;
class DTReadOutMapping;
class DTTimeEvolutionHisto;

class DTBlockedROChannelsTest: public edm::EDAnalyzer{

  public:

    /// Constructor
    DTBlockedROChannelsTest(const edm::ParameterSet& ps);

    /// Destructor
    ~DTBlockedROChannelsTest();

  protected:

    /// BeginJob
    void beginJob();

    /// BeginRun
    void beginRun(const edm::Run& run, const edm::EventSetup& c);

    /// Analyze
    void analyze(const edm::Event& e, const edm::EventSetup& c);

    /// Endjob
    void endJob();

    void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

    /// DQM Client operations
    void performClientDiagnostic();

    /// DQM Client Diagnostic in online mode
    void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

    /// DQM Client Diagnostic in offline mode
    void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  private:
    int readOutToGeometry(int dduId, int rosNumber, int& wheel, int& sector);

  private:

    //Number of onUpdates
    int nupdates;

    // prescale on the # of LS to update the test
    int prescaleFactor;

    int nevents;
    int neventsPrev;
    unsigned int nLumiSegs;
    unsigned int prevNLumiSegs;
    double prevTotalPerc;

    int run;


    DQMStore* dbe;
    edm::ESHandle<DTReadOutMapping> mapping;


    // Monitor Elements
    std::map<int, MonitorElement*> wheelHitos;
    MonitorElement *summaryHisto;

    bool offlineMode;

    std::map<int, double> resultsPerLumi;
    DTTimeEvolutionHisto* hSystFractionVsLS;


    class DTRobBinsMap {
      public:
        DTRobBinsMap(const int fed, const int ros, const DQMStore* dbe);

        DTRobBinsMap();


        ~DTRobBinsMap();

        // add a rob to the set of robs
        void addRobBin(int robBin);
        void init(bool v) {init_ = v;}

        bool robChanged(int robBin);

        double getChamberPercentage();

        void readNewValues();

      private:
        int getValueRobBin(int robBin) const;
        int getValueRos() const;

        int rosBin;
        bool init_;

        std::map<int, int> robsAndValues;
        int rosValue;

        const MonitorElement* meROS;
        const MonitorElement* meDDU;

        std::string rosHName;
        std::string dduHName;

        const DQMStore* theDbe;
    };

    std::map<DTChamberId, DTRobBinsMap> chamberMap;

};

#endif
