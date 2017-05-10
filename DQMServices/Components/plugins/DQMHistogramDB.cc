#include "DQMHistogramDB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/MakerMacros.h"

namespace dqmservices {

DQMHistogramDB::DQMHistogramDB(edm::ParameterSet const & ps) : DQMHistogramStats(ps){
  dbw_.reset(new DQMDatabaseWriter(ps));
  dbw_->initDatabase();
};

void DQMHistogramDB::dqmBeginRun(DQMStore::IBooker &, 
                                 DQMStore::IGetter &iGetter,
                                 edm::Run const &iRun, 
                                 edm::EventSetup const&){

  edm::LogInfo("DQMDatabaseHarvester") << "DQMDatabaseHarvester::dqmBeginRun " << std::endl;
  std::cout << "****Collecting info from DQMStore" << std::endl;
  HistoStats stats = collect(iGetter, histograms_);
  dbw_->dqmPropertiesDbDrop(stats, iRun.run());
}

void DQMHistogramDB::dqmEndLuminosityBlock(DQMStore::IBooker &,
                                           DQMStore::IGetter &iGetter,
                                           edm::LuminosityBlock const &iLumi,
                                           edm::EventSetup const &) {
  if (dumpOnEndLumi_){
    edm::LogInfo("DQMDatabaseHarvester") << "DQMDatabaseHarvester::dqmEndLuminosityBlock " << std::endl;
    HistoStats stats = (histogramNamesEndLumi_.size() > 0) ? collect(iGetter, histogramNamesEndLumi_) : collect(iGetter);
    dbw_->dqmValuesDbDrop(stats, iLumi.run(), iLumi.luminosityBlock());
  }
}

void DQMHistogramDB::dqmEndRun(DQMStore::IBooker &, 
                            DQMStore::IGetter &iGetter,
                            edm::Run const &iRun, 
                            edm::EventSetup const&) {
  if (dumpOnEndRun_){
    edm::LogInfo("DQMDatabaseHarvester") <<  "DQMDatabaseHarvester::endRun" << std::endl;
    HistoStats stats = (histogramNamesEndRun_.size() > 0) ? collect(iGetter, histograms_) : collect(iGetter);
    //for run based histograms, we use lumisection value set to 0.
    dbw_->dqmValuesDbDrop(stats, iRun.run(), 0);
  }
}

DEFINE_FWK_MODULE(DQMHistogramDB);
}  // end of namespace


