#ifndef RPCDCSINFO_H
#define RPCDCSINFO_H

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Run.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/ServiceRegistry/interface/Service.h>

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

///Data Format
#include "DataFormats/Scalers/interface/DcsStatus.h"

class RPCDcsInfo: public DQMEDAnalyzer{

public:

  /// Constructor
  RPCDcsInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCDcsInfo();

protected:

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c);



private:

  void makeDcsInfo(const edm::Event& e);  

  // DQMStore *dbe_;
  edm::ParameterSet parameters_;
  std::string subsystemname_;
  std::string dcsinfofolder_;
  
  bool dcs;
   // histograms
  MonitorElement * DCSbyLS_ ;
  edm::EDGetTokenT<DcsStatusCollection> scalersRawToDigiLabel_;
  
};

#endif
