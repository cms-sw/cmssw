#ifndef DTMonitorModule_DTBlockedROChannelsTest_H
#define DTMonitorModule_DTBlockedROChannelsTest_H

/** \class DTBlockedROChannelsTest
 * *
 *  DQM Client to Summarize LS by LS the status of the Read-Out channels.
 *
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
#include <DQMServices/Core/interface/DQMEDHarvester.h>

class DQMStore;
class MonitorElement;
class DTReadOutMapping;
class DTTimeEvolutionHisto;

class DTBlockedROChannelsTest: public DQMEDHarvester {

  public:

    /// Constructor
    DTBlockedROChannelsTest(const edm::ParameterSet& ps);

    /// Destructor
    ~DTBlockedROChannelsTest();

  protected:

    /// BeginRun
    void beginRun(const edm::Run& , const edm::EventSetup&);

    void fillChamberMap( DQMStore::IGetter & igetter, const edm::EventSetup& c); 


    /// DQM Client operations
    void performClientDiagnostic(DQMStore::IGetter & igetter);

    /// DQM Client Diagnostic in online mode
    void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&); 
    void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);

private:
    int readOutToGeometry(int dduId, int rosNumber, int& wheel, int& sector);

  private:

    //Number of onUpdates
    int nupdates;

    // prescale on the # of LS to update the test
    int prescaleFactor;
    bool offlineMode;
    int nevents;
    int neventsPrev;
    unsigned int nLumiSegs;
    unsigned int prevNLumiSegs;
    double prevTotalPerc;

    int run;

    edm::ESHandle<DTReadOutMapping> mapping;


    // Monitor Elements
    std::map<int, MonitorElement*> wheelHitos;
    MonitorElement *summaryHisto;

    std::map<int, double> resultsPerLumi;
    DTTimeEvolutionHisto* hSystFractionVsLS;


    class DTRobBinsMap {
      public:
        DTRobBinsMap(DQMStore::IGetter & igetter,const int fed, const int ros);

        DTRobBinsMap();


        ~DTRobBinsMap();

        // add a rob to the set of robs
        void addRobBin(int robBin);
        void init(bool v) {init_ = v;}

        bool robChanged(int robBin);

        double getChamberPercentage(DQMStore::IGetter &);

        void readNewValues(DQMStore::IGetter & igetter);

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
    };

    std::map<DTChamberId, DTRobBinsMap> chamberMap;

};

#endif
