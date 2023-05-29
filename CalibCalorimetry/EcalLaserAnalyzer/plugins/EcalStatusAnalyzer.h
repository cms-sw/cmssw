// $Id: EcalStatusAnalyzer.h

#include <vector>
#include <map>

#include <memory>
#include <FWCore/Framework/interface/one/EDAnalyzer.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h>

class Timestamp;

class EcalStatusAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalStatusAnalyzer(const edm::ParameterSet& iConfig);
  ~EcalStatusAnalyzer() override = default;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void beginJob() override;
  void endJob() override;

  enum EcalLaserColorType {
    iBLUE = 0,
    iGREEN = 1,
    iRED = 3,  // in fact should be 2
    iIR = 2    // in fact should be 3
  };

private:
  int iEvent;

  const std::string _dataType;

  const std::string resdir_;
  const std::string statusfile_;
  const std::string eventHeaderCollection_;
  const std::string eventHeaderProducer_;
  const edm::EDGetTokenT<EcalRawDataCollection> dccToken_;
  const edm::EDGetTokenT<EcalTBEventHeader> headToken_;

  std::map<int, int> isFedLasCreated;
  std::map<int, int> isFedTPCreated;
  std::map<int, int> isFedPedCreated;

  std::vector<int> fedIDsLas;
  std::vector<int> fedIDsTP;
  std::vector<int> fedIDsPed;
  std::vector<int> dccIDsLas;
  std::vector<int> dccIDsTP;
  std::vector<int> dccIDsPed;

  // Identify run type

  int runType;
  int runNum;
  int event;
  int nSM;
  int fedID;
  int dccID;

  unsigned long long timeStampCur;

  std::map<int, unsigned long long> timeStampBegLas;
  std::map<int, unsigned long long> timeStampEndLas;

  std::map<int, unsigned long long> timeStampBegTP;
  std::map<int, unsigned long long> timeStampEndTP;

  std::map<int, unsigned long long> timeStampBegPed;
  std::map<int, unsigned long long> timeStampEndPed;

  std::map<int, short> MGPAGainLas;
  std::map<int, short> MEMGainLas;

  std::map<int, short> MGPAGainTP;
  std::map<int, short> MEMGainTP;

  std::map<int, short> MGPAGainPed;
  std::map<int, short> MEMGainPed;

  std::map<int, int> laserPowerBlue;
  std::map<int, int> laserFilterBlue;
  std::map<int, int> laserDelayBlue;

  std::map<int, int> laserPowerRed;
  std::map<int, int> laserFilterRed;
  std::map<int, int> laserDelayRed;

  std::map<int, int> nEvtsLas;
  std::map<int, int> nBlueLas;
  std::map<int, int> nRedLas;
  std::map<int, int> runTypeLas;

  std::map<int, int> nEvtsTP;
  std::map<int, int> runTypeTP;

  std::map<int, int> nEvtsPed;
  std::map<int, int> runTypePed;
};
