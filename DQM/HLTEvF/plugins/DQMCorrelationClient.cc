#include "DQM/HLTEvF/plugins/DQMCorrelationClient.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// -------------------------------------- Constructor --------------------------------------------
//
DQMCorrelationClient::DQMCorrelationClient(const edm::ParameterSet& iConfig)
    : me1onX_(iConfig.getParameter<bool>("me1onX")),
      meXpset_(me1onX_ ? getHistoPSet(iConfig.getParameter<edm::ParameterSet>("me1"))
                       : getHistoPSet(iConfig.getParameter<edm::ParameterSet>("me2"))),
      meYpset_(me1onX_ ? getHistoPSet(iConfig.getParameter<edm::ParameterSet>("me2"))
                       : getHistoPSet(iConfig.getParameter<edm::ParameterSet>("me1"))),
      mepset_(getOutputHistoPSet(iConfig.getParameter<edm::ParameterSet>("me"))) {
  edm::LogInfo("DQMCorrelationClient") << "Constructor  DQMCorrelationClient::DQMCorrelationClient " << std::endl;

  correlation_ = nullptr;
}

MEPSet DQMCorrelationClient::getHistoPSet(edm::ParameterSet pset) {
  return MEPSet{
      pset.getParameter<std::string>("name"),
      pset.getParameter<std::string>("folder"),
      pset.getParameter<bool>("profileX"),
  };
}

OutputMEPSet DQMCorrelationClient::getOutputHistoPSet(edm::ParameterSet pset) {
  return OutputMEPSet{
      pset.getParameter<std::string>("name"),
      pset.getParameter<std::string>("folder"),
      pset.getParameter<bool>("doXaxis"),
      pset.getParameter<int>("nbinsX"),
      pset.getParameter<double>("xminX"),
      pset.getParameter<double>("xmaxX"),
      pset.getParameter<bool>("doYaxis"),
      pset.getParameter<int>("nbinsY"),
      pset.getParameter<double>("xminY"),
      pset.getParameter<double>("xmaxY"),
  };
}

//
// -------------------------------------- beginJob --------------------------------------------
//
void DQMCorrelationClient::beginJob() {
  edm::LogInfo("DQMCorrelationClient") << "DQMCorrelationClient::beginJob " << std::endl;
}

TH1* DQMCorrelationClient::getTH1(MonitorElement* me, bool profileX = true) {
  TH1* th1 = nullptr;

  MonitorElement::Kind kind = me->kind();
  switch (kind) {
    case (MonitorElement::Kind::TH2D):
      th1 = (profileX ? me->getTH2D()->ProfileX() : me->getTH2D()->ProfileY());
      break;
    case (MonitorElement::Kind::TH2F):
      th1 = (profileX ? me->getTH2F()->ProfileX() : me->getTH2F()->ProfileY());
      break;
    case (MonitorElement::Kind::TH2S):
      th1 = (profileX ? me->getTH2S()->ProfileX() : me->getTH2S()->ProfileY());
      break;
    case (MonitorElement::Kind::TPROFILE):
      th1 = me->getTH1();
      break;
    default:
      break;
  }

  return th1;
}
void DQMCorrelationClient::setAxisTitle(MonitorElement* meX, MonitorElement* meY) {
  if (correlation_ == nullptr)
    return;
  correlation_->setAxisTitle(meX->getTH1()->GetYaxis()->GetTitle(), 1);
  correlation_->setAxisTitle(meY->getTH1()->GetYaxis()->GetTitle(), 2);

  if (!mepset_.doXaxis) {
    TAxis* axis = (meX->getTH1()->GetYaxis());
    for (int i = 1; i <= axis->GetNbins(); ++i)
      correlation_->getTH1()->GetXaxis()->SetBinLabel(i, axis->GetBinLabel(i));
  }

  if (!mepset_.doYaxis) {
    TAxis* axis = (meY->getTH1()->GetYaxis());
    for (int i = 1; i <= axis->GetNbins(); ++i)
      correlation_->getTH1()->GetYaxis()->SetBinLabel(i, axis->GetBinLabel(i));
  }
}

//
// -------------------------------------- get and book in the endJob --------------------------------------------
//
void DQMCorrelationClient::dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) {
  std::string hname = "";

  //get available histograms
  hname = meXpset_.folder + "/" + meXpset_.name;
  MonitorElement* meX = igetter_.get(hname);
  hname = meYpset_.folder + "/" + meYpset_.name;
  MonitorElement* meY = igetter_.get(hname);

  if (!meX || !meY) {
    edm::LogError("DQMCorrelationClient")
        << "MEs not found! " << (!meX ? meXpset_.folder + "/" + meXpset_.name + " not found " : "")
        << (!meY ? meYpset_.folder + "/" + meYpset_.name + " not found " : "") << std::endl;
    return;
  }

  // get range and binning for new MEs
  int nbinsX = (mepset_.doXaxis ? mepset_.nbinsX : meX->getNbinsY());
  double xminX = (mepset_.doXaxis ? mepset_.xminX : meX->getTH1()->GetYaxis()->GetXmin());
  double xmaxX = (mepset_.doXaxis ? mepset_.xmaxX : meX->getTH1()->GetYaxis()->GetXmax());
  int nbinsY = (mepset_.doYaxis ? mepset_.nbinsY : meY->getNbinsY());
  double xminY = (mepset_.doYaxis ? mepset_.xminY : meY->getTH1()->GetYaxis()->GetXmin());
  double xmaxY = (mepset_.doYaxis ? mepset_.xmaxY : meY->getTH1()->GetYaxis()->GetXmax());

  // create and cd into new folder
  std::string currentFolder = mepset_.folder;
  ibooker_.setCurrentFolder(currentFolder);

  //book new histogram
  hname = mepset_.name;
  correlation_ = ibooker_.book2D(hname, hname, nbinsX, xminX, xmaxX, nbinsY, xminY, xmaxY);
  setAxisTitle(meX, meY);

  // handle mes
  TH1* x = nullptr;
  TH1* y = nullptr;
  x = getTH1(meX, meXpset_.profileX);
  y = getTH1(meY, meYpset_.profileX);

  size_t size = x->GetXaxis()->GetNbins();

  std::vector<double> xvalue;
  std::vector<int> xbinvalue;
  for (size_t ibin = 1; ibin <= size; ++ibin) {
    // avoid to store points w/ no info
    if (x->GetBinContent(ibin) == 0.)
      continue;
    xvalue.push_back(x->GetBinContent(ibin));
    xbinvalue.push_back(x->GetXaxis()->GetBinCenter(ibin));
  }

  for (size_t i = 0; i < xbinvalue.size(); ++i) {
    int ybin = y->GetXaxis()->FindBin(xbinvalue[i]);
    double yvalue = y->GetBinContent(ybin);
    correlation_->Fill(xvalue[i], yvalue);
  }
}

//
// -------------------------------------- get in the endLumi if needed --------------------------------------------
//
void DQMCorrelationClient::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker_,
                                                 DQMStore::IGetter& igetter_,
                                                 edm::LuminosityBlock const& iLumi,
                                                 edm::EventSetup const& iSetup) {
  edm::LogInfo("DQMCorrelationClient") << "DQMCorrelationClient::endLumi " << std::endl;
}

void DQMCorrelationClient::fillMePSetDescription(edm::ParameterSetDescription& pset) {
  pset.add<std::string>("folder", "");
  pset.add<std::string>("name", "");
  pset.add<bool>("profileX", true);
}

void DQMCorrelationClient::fillOutputMePSetDescription(edm::ParameterSetDescription& pset) {
  //  fillMePSetDescription(pset);
  pset.add<std::string>("folder");
  pset.add<std::string>("name");
  pset.add<bool>("doXaxis", true);
  pset.add<int>("nbinsX", 2500);
  pset.add<double>("xminX", 0.);
  pset.add<double>("xmaxX", 2500.);
  pset.add<bool>("doYaxis", true);
  pset.add<int>("nbinsY", 2500);
  pset.add<double>("xminY", 0.);
  pset.add<double>("xmaxY", 2500.);
}

void DQMCorrelationClient::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("me1onX", true);

  edm::ParameterSetDescription mePSet;
  fillOutputMePSetDescription(mePSet);
  desc.add<edm::ParameterSetDescription>("me", mePSet);

  edm::ParameterSetDescription me1PSet;
  fillMePSetDescription(me1PSet);
  desc.add<edm::ParameterSetDescription>("me1", me1PSet);

  edm::ParameterSetDescription me2PSet;
  fillMePSetDescription(me2PSet);
  desc.add<edm::ParameterSetDescription>("me2", me2PSet);

  descriptions.add("dqmCorrelationClient", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMCorrelationClient);
