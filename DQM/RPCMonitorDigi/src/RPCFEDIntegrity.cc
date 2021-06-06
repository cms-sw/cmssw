/*  \author Anna Cimmino*/
#include "DQM/RPCMonitorDigi/interface/RPCFEDIntegrity.h"

#include "DQM/RPCMonitorDigi/interface/RPCRawDataCountsHistoMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

RPCFEDIntegrity::RPCFEDIntegrity(const edm::ParameterSet& ps) {
  edm::LogVerbatim("rpcfedintegrity") << "[RPCFEDIntegrity]: Constructor";

  rawCountsLabel_ = consumes<RPCRawDataCounts>(ps.getUntrackedParameter<edm::InputTag>("RPCRawCountsInputTag"));
  prefixDir_ = ps.getUntrackedParameter<std::string>("RPCPrefixDir", "RPC/FEDIntegrity");
  merge_ = ps.getUntrackedParameter<bool>("MergeRuns", false);
  minFEDNum_ = ps.getUntrackedParameter<int>("MinimumFEDID", 790);
  maxFEDNum_ = ps.getUntrackedParameter<int>("MaximumFEDID", 792);

  init_ = false;
  numOfFED_ = maxFEDNum_ - minFEDNum_ + 1;
  FATAL_LIMIT = 5;
}

RPCFEDIntegrity::~RPCFEDIntegrity() {
  edm::LogVerbatim("rpcfedintegrity") << "[RPCFEDIntegrity]: Destructor ";
  //  dbe_=0;
}

void RPCFEDIntegrity::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::LogVerbatim("rpcfedintegrity") << "[RPCFEDIntegrity]: Begin booking histograms ";

  this->bookFEDMe(ibooker);
}

void RPCFEDIntegrity::analyze(const edm::Event& iEvent, const edm::EventSetup& c) {
  //get hold of raw data counts
  edm::Handle<RPCRawDataCounts> rawCounts;
  iEvent.getByToken(rawCountsLabel_, rawCounts);
  if (!rawCounts.isValid())
    return;

  const RPCRawDataCounts& counts = *rawCounts.product();

  for (int fed = minFEDNum_; fed <= maxFEDNum_; ++fed) {
    if (counts.fedBxRecords(fed))
      fedMe_[Entries]->Fill(fed);
    if (counts.fedFormatErrors(fed))
      fedMe_[Fatal]->Fill(fed);
    if (counts.fedErrorRecords(fed))
      fedMe_[NonFatal]->Fill(fed);
  }
}

//Fill report summary
void RPCFEDIntegrity::bookFEDMe(DQMStore::IBooker& ibooker) {
  ibooker.cd();
  ibooker.setCurrentFolder(prefixDir_);

  fedMe_[Entries] = ibooker.book1D("FEDEntries", "FED Entries", numOfFED_, minFEDNum_, maxFEDNum_ + 1);
  this->labelBins(fedMe_[Entries]);
  fedMe_[Fatal] = ibooker.book1D("FEDFatal", "FED Fatal Errors", numOfFED_, minFEDNum_, maxFEDNum_ + 1);
  this->labelBins(fedMe_[Fatal]);
  fedMe_[NonFatal] = ibooker.book1D("FEDNonFatal", "FED NON Fatal Errors", numOfFED_, minFEDNum_, maxFEDNum_ + 1);
  this->labelBins(fedMe_[NonFatal]);

  init_ = true;
}

void RPCFEDIntegrity::labelBins(MonitorElement* myMe) {
  int xbins = myMe->getNbinsX();

  if (xbins != numOfFED_)
    return;

  for (int i = 0; i < xbins; i++) {
    const std::string xLabel = fmt::format("{}", minFEDNum_ + i);
    myMe->setBinLabel(i + 1, xLabel, 1);
  }
}
