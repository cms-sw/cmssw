#include "../interface/EcalFEDMonitor.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "FWCore/Framework/interface/MakerMacros.h"

template<int SUBDET>
EcalFEDMonitorTemp<SUBDET>::EcalFEDMonitorTemp(edm::ParameterSet const& _ps) :
  folderName_(_ps.getUntrackedParameter<std::string>("folderName")),
  FEDRawDataToken_(consumes<FEDRawDataCollection>(_ps.getParameter<edm::InputTag>("FEDRawDataCollection"))),
  ebGainErrorsToken_(),
  eeGainErrorsToken_(),
  ebChIdErrorsToken_(),
  eeChIdErrorsToken_(),
  ebGainSwitchErrorsToken_(),
  eeGainSwitchErrorsToken_(),
  towerIdErrorsToken_(consumes<EcalElectronicsIdCollection>(_ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection1"))),
  blockSizeErrorsToken_(consumes<EcalElectronicsIdCollection>(_ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection2"))),
  MEs_(nMEs, 0)
{
  if(_ps.existsAs<edm::InputTag>("EBDetIdCollection1"))
    ebGainErrorsToken_ = consumes<EBDetIdCollection>(_ps.getParameter<edm::InputTag>("EBDetIdCollection1"));
  if(_ps.existsAs<edm::InputTag>("EEDetIdCollection1"))
    eeGainErrorsToken_ = consumes<EEDetIdCollection>(_ps.getParameter<edm::InputTag>("EEDetIdCollection1"));
  if(_ps.existsAs<edm::InputTag>("EBDetIdCollection2"))
    ebChIdErrorsToken_ = consumes<EBDetIdCollection>(_ps.getParameter<edm::InputTag>("EBDetIdCollection2"));
  if(_ps.existsAs<edm::InputTag>("EEDetIdCollection2"))
    eeChIdErrorsToken_ = consumes<EEDetIdCollection>(_ps.getParameter<edm::InputTag>("EEDetIdCollection2"));
  if(_ps.existsAs<edm::InputTag>("EBDetIdCollection3"))
    ebGainSwitchErrorsToken_ = consumes<EBDetIdCollection>(_ps.getParameter<edm::InputTag>("EBDetIdCollection3"));
  if(_ps.existsAs<edm::InputTag>("EEDetIdCollection3"))
    eeGainSwitchErrorsToken_ = consumes<EEDetIdCollection>(_ps.getParameter<edm::InputTag>("EEDetIdCollection3"));
}

template<int SUBDET>
void
EcalFEDMonitorTemp<SUBDET>::dqmBeginRun(edm::Run const&, edm::EventSetup const& _es)
{
  if(!ecaldqm::checkElectronicsMap(false)){
    // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
    edm::ESHandle<EcalElectronicsMapping> elecMapHandle;
    _es.get<EcalMappingRcd>().get(elecMapHandle);
    ecaldqm::setElectronicsMap(elecMapHandle.product());
  }
}

template<int SUBDET>
void
EcalFEDMonitorTemp<SUBDET>::bookHistograms(DQMStore::IBooker& _ibooker, edm::Run const&, edm::EventSetup const&)
{
  _ibooker.cd();

  std::string name;

  if(SUBDET == EcalBarrel || SUBDET < 0){
    _ibooker.setCurrentFolder("EcalBarrel/" + folderName_);

    name = "FEDEntries";
    MEs_[kEBOccupancy] = _ibooker.book1D(name, name, 36, 610, 646);

    name = "FEDFatal";
    MEs_[kEBFatal] = _ibooker.book1D(name, name, 36, 610, 646);

    name = "FEDNonFatal";
    MEs_[kEBNonFatal] = _ibooker.book1D(name, name, 36, 610, 646);
  }
  if(SUBDET == EcalEndcap || SUBDET < 0){
    _ibooker.setCurrentFolder("EcalEndcap/" + folderName_);

    name = "FEDEntries";
    MEs_[kEEOccupancy] = _ibooker.book1D(name, name, 54, 601, 655);

    name = "FEDFatal";
    MEs_[kEEFatal] = _ibooker.book1D(name, name, 54, 601, 655);

    name = "FEDNonFatal";
    MEs_[kEENonFatal] = _ibooker.book1D(name, name, 54, 601, 655);
  }
}

template<int SUBDET>
void
EcalFEDMonitorTemp<SUBDET>::analyze(edm::Event const& _evt, edm::EventSetup const&)
{
  edm::Handle<FEDRawDataCollection> fedHndl;
  if(_evt.getByToken(FEDRawDataToken_, fedHndl)){
    for(unsigned fedId(601); fedId <= 654; fedId++){
      if(SUBDET == EcalBarrel && (fedId < 610 || fedId > 645)) continue;
      if(SUBDET == EcalEndcap && (fedId > 609 && fedId < 646)) continue;

      unsigned occupancy(-1);
      //      unsigned fatal(-1);
      if(fedId < 610 || fedId > 645){
        occupancy = kEEOccupancy;
	//        fatal = kEEFatal;
      }
      else{
        occupancy = kEBOccupancy;
	//        fatal = kEBFatal;
      }

      const FEDRawData& fedData(fedHndl->FEDData(fedId));
      unsigned length(fedData.size() / sizeof(uint64_t));

      if(length > 1){ // FED header is one 64 bit word
        MEs_[occupancy]->Fill(fedId + 0.5);

//  	const uint64_t* pData(reinterpret_cast<const uint64_t*>(fedData.data()));
//  	bool crcError(((pData[length - 1] >> 2) & 0x1) == 0x1);

// 	if(crcError) MEs_[fatal]->Fill(fedId + 0.5);
      }
    }
  }

  edm::Handle<EBDetIdCollection> ebHndl;
  edm::Handle<EEDetIdCollection> eeHndl;
  edm::Handle<EcalElectronicsIdCollection> eleHndl;

  if((SUBDET == EcalBarrel || SUBDET < 0) && _evt.getByToken(ebGainErrorsToken_, ebHndl)){
    EBDetIdCollection::const_iterator ebEnd(ebHndl->end());
    for(EBDetIdCollection::const_iterator ebItr(ebHndl->begin()); ebItr != ebEnd; ++ebItr){
      unsigned iDCC(ecaldqm::dccId(*ebItr) - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[kEBNonFatal]->Fill(iDCC + 601.5, 1. / normalization);
    }
  }
  if((SUBDET == EcalEndcap || SUBDET < 0) && _evt.getByToken(eeGainErrorsToken_, eeHndl)){
    EEDetIdCollection::const_iterator eeEnd(eeHndl->end());
    for(EEDetIdCollection::const_iterator eeItr(eeHndl->begin()); eeItr != eeEnd; ++eeItr){
      unsigned iDCC(ecaldqm::dccId(*eeItr) - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[kEENonFatal]->Fill(iDCC + 601.5, 1. / normalization);
    }
  }

  if((SUBDET == EcalBarrel || SUBDET < 0) && _evt.getByToken(ebChIdErrorsToken_, ebHndl)){
    EBDetIdCollection::const_iterator ebEnd(ebHndl->end());
    for(EBDetIdCollection::const_iterator ebItr(ebHndl->begin()); ebItr != ebEnd; ++ebItr){
      unsigned iDCC(ecaldqm::dccId(*ebItr) - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[kEBNonFatal]->Fill(iDCC + 601.5, 1. / normalization);
    }
  }
  if((SUBDET == EcalEndcap || SUBDET < 0) && _evt.getByToken(eeChIdErrorsToken_, eeHndl)){
    EEDetIdCollection::const_iterator eeEnd(eeHndl->end());
    for(EEDetIdCollection::const_iterator eeItr(eeHndl->begin()); eeItr != eeEnd; ++eeItr){
      unsigned iDCC(ecaldqm::dccId(*eeItr) - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[kEENonFatal]->Fill(iDCC + 601.5, 1. / normalization);
    }
  }

  if((SUBDET == EcalBarrel || SUBDET < 0) && _evt.getByToken(ebGainSwitchErrorsToken_, ebHndl)){
    EBDetIdCollection::const_iterator ebEnd(ebHndl->end());
    for(EBDetIdCollection::const_iterator ebItr(ebHndl->begin()); ebItr != ebEnd; ++ebItr){
      unsigned iDCC(ecaldqm::dccId(*ebItr) - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[kEBNonFatal]->Fill(iDCC + 601.5, 1. / normalization);
    }
  }
  if((SUBDET == EcalEndcap || SUBDET < 0) && _evt.getByToken(eeGainSwitchErrorsToken_, eeHndl)){
    EEDetIdCollection::const_iterator eeEnd(eeHndl->end());
    for(EEDetIdCollection::const_iterator eeItr(eeHndl->begin()); eeItr != eeEnd; ++eeItr){
      unsigned iDCC(ecaldqm::dccId(*eeItr) - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[kEENonFatal]->Fill(iDCC + 601.5, 1. / normalization);
    }
  }

  if(_evt.getByToken(towerIdErrorsToken_, eleHndl)){
    EcalElectronicsIdCollection::const_iterator eleEnd(eleHndl->end());
    for(EcalElectronicsIdCollection::const_iterator eleItr(eleHndl->begin()); eleItr != eleEnd; ++eleItr){
      unsigned iDCC(eleItr->dccId() - 1);

      unsigned nonfatal(-1);
      if((SUBDET == EcalBarrel || SUBDET < 0) && iDCC >= ecaldqm::kEBmLow && iDCC <= ecaldqm::kEBpHigh)
        nonfatal = kEBNonFatal;
      else if((SUBDET == EcalEndcap || SUBDET < 0) && (iDCC <= ecaldqm::kEEmHigh || iDCC >= ecaldqm::kEEpLow))
        nonfatal = kEENonFatal;
      else
        continue;

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[nonfatal]->Fill(iDCC + 601.5, 25. / normalization);
    }
  }

  if(_evt.getByToken(blockSizeErrorsToken_, eleHndl)){
    EcalElectronicsIdCollection::const_iterator eleEnd(eleHndl->end());
    for(EcalElectronicsIdCollection::const_iterator eleItr(eleHndl->begin()); eleItr != eleEnd; ++eleItr){
      unsigned iDCC(eleItr->dccId() - 1);

      unsigned nonfatal(-1);
      if((SUBDET == EcalBarrel || SUBDET < 0) && iDCC >= ecaldqm::kEBmLow && iDCC <= ecaldqm::kEBpHigh)
        nonfatal = kEBNonFatal;
      else if((SUBDET == EcalEndcap || SUBDET < 0) && (iDCC <= ecaldqm::kEEmHigh || iDCC >= ecaldqm::kEEpLow))
        nonfatal = kEENonFatal;
      else
        continue;

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[nonfatal]->Fill(iDCC + 601.5, 25. / normalization);
    }
  }
}

typedef EcalFEDMonitorTemp<EcalBarrel> EBHltTask;
typedef EcalFEDMonitorTemp<EcalEndcap> EEHltTask;
typedef EcalFEDMonitorTemp<-1> EcalFEDMonitor;

DEFINE_FWK_MODULE(EBHltTask);
DEFINE_FWK_MODULE(EEHltTask);
DEFINE_FWK_MODULE(EcalFEDMonitor);
