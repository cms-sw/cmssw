#include "DQM/TrackingMonitorClient/interface/DQMScaleToClient.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// -------------------------------------- Constructor --------------------------------------------
//
DQMScaleToClient::DQMScaleToClient(const edm::ParameterSet& iConfig) :
  inputmepset_    ( getHistoPSet       (iConfig.getParameter<edm::ParameterSet>("inputme"))  )
  , outputmepset_ ( getOutputHistoPSet (iConfig.getParameter<edm::ParameterSet>("outputme")) )
{
  edm::LogInfo("DQMScaleToClient") <<  "Constructor  DQMScaleToClient::DQMScaleToClient " << std::endl;

  scaled_ = nullptr;

}

MEPSet DQMScaleToClient::getHistoPSet(edm::ParameterSet pset)
{
  return MEPSet{
    pset.getParameter<std::string>("name"),
      pset.getParameter<std::string>("folder"),
  };
}

OutputMEPSet DQMScaleToClient::getOutputHistoPSet(edm::ParameterSet pset)
{
  return OutputMEPSet{
    pset.getParameter<std::string>("name"),
      pset.getParameter<std::string>("folder"),
      pset.getParameter<double>("factor"),
      };
}

//
// -------------------------------------- beginJob --------------------------------------------
//
void DQMScaleToClient::beginJob()
{
  edm::LogInfo("DQMScaleToClient") <<  "DQMScaleToClient::beginJob " << std::endl;
}

//
// -------------------------------------- get and book in the endJob --------------------------------------------
//
void DQMScaleToClient::dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_)
{
  std::string hname = "";

  // create and cd into new folder
  std::string currentFolder = outputmepset_.folder;
  ibooker_.setCurrentFolder(currentFolder);

  //get available histograms
  hname = inputmepset_.folder + "/" + inputmepset_.name;
  MonitorElement* inputme = igetter_.get( hname );

  if (!inputme) {
    edm::LogError("DQMScaleToClient") <<  "MEs not found! " 
				      << inputmepset_.folder + "/" + inputmepset_.name + " not found "
				      << std::endl;
    return;
  }

  //book new histogram
  ibooker_.setCurrentFolder(currentFolder);
  hname = outputmepset_.name;
  scaled_ = ibooker_.book1D(hname,(TH1F*)inputme->getTH1()->Clone(hname.c_str()));

  // handle mes
  double integral = (scaled_->getTH1()->Integral() > 0. ? scaled_->getTH1()->Integral() : 1.);
  scaled_->getTH1()->Scale(outputmepset_.factor/integral);
}

//
// -------------------------------------- get in the endLumi if needed --------------------------------------------
//
void DQMScaleToClient::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker_, DQMStore::IGetter & igetter_, edm::LuminosityBlock const & iLumi, edm::EventSetup const& iSetup) 
{
  edm::LogInfo("DQMScaleToClient") <<  "DQMScaleToClient::endLumi " << std::endl;
}

void DQMScaleToClient::fillMePSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<std::string>("folder","");
  pset.add<std::string>("name","");
}

void DQMScaleToClient::fillOutputMePSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<std::string>("folder","");
  pset.add<std::string>("name","");
  pset.add<double>("factor",1.);
}

void DQMScaleToClient::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{

  edm::ParameterSetDescription desc;
  edm::ParameterSetDescription outputmePSet;
  fillOutputMePSetDescription(outputmePSet);
  desc.add<edm::ParameterSetDescription>("outputme", outputmePSet);

  edm::ParameterSetDescription inputmePSet;
  fillMePSetDescription(inputmePSet);
  desc.add<edm::ParameterSetDescription>("inputme", inputmePSet);

  descriptions.add("dqmScaleToClient", desc);

}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMScaleToClient);
