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
    ~DTBlockedROChannelsTest() override;

  protected:

    /// BeginRun
    void beginRun(const edm::Run& , const edm::EventSetup&) override;

    void fillChamberMap( DQMStore::IGetter & igetter, const edm::EventSetup& c); 


    /// DQM Client operations
    void performClientDiagnostic(DQMStore::IGetter & igetter);

    /// DQM Client Diagnostic in online mode
    void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) override; 
    void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
    int readOutToGeometry(int dduId, int rosNumber, int robNumber, int& wheel, int &station, int& sector);

    int theDDU(int crate, int slot, int link, bool tenDDU);
    int theROS(int slot, int link);
    int theROB(int slot, int link);

    //Number of onUpdates
    int nupdates;

    // prescale on the # of LS to update the test
    int prescaleFactor;
    bool offlineMode;
    bool checkUros;
    int nevents;
    int neventsPrev;
    unsigned int nLumiSegs;
    unsigned int prevNLumiSegs;
    double prevTotalPerc;

    int run;

    edm::ESHandle<DTReadOutMapping> mapping;


    // Monitor Elements
    std::map<int, MonitorElement*> wheelHistos;
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

   // For uROS starting in Year 2018
    class DTLinkBinsMap{
      public:
        DTLinkBinsMap(DQMStore::IGetter & igetter,const int fed, const int mapSlot);

        DTLinkBinsMap();

        ~DTLinkBinsMap();

        // add a rob to the set of robs
        void addLinkBin(int linkBin);
        void init(bool v) {init_ = v;}

        bool linkChanged(int linkBin);

        double getChamberPercentage(DQMStore::IGetter &);

        void readNewValues(DQMStore::IGetter & igetter);

      private:
        int getValueLinkBin(int linkBin) const;

        bool init_;

        std::map<int, int> linksAndValues;

        const MonitorElement* meuROS;

        std::string urosHName;
    };

    std::map<DTChamberId, DTLinkBinsMap> chamberMapUros;



};

#endif
