#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_SingleLepton.h"

#include <limits>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Math/interface/deltaR.h"

namespace{
  void SetBinLabels(MonitorElement * const me){
    me->setBinLabel(1, "No CSV Reqs.");
    me->setBinLabel(2, "N_{CSVL} >= 1");
    me->setBinLabel(3, "N_{CSVL} >= 2");
    me->setBinLabel(4, "N_{CSVL} >= 3");
    me->setBinLabel(5, "N_{CSVL} >= 4");
    me->setBinLabel(6, "N_{CSVM} >= 1");
    me->setBinLabel(7, "N_{CSVM} >= 2");
    me->setBinLabel(8, "N_{CSVM} >= 3");
    me->setBinLabel(9, "N_{CSVM} >= 4");
    me->setBinLabel(10, "N_{CSVT} >= 1");
    me->setBinLabel(11, "N_{CSVT} >= 2");
    me->setBinLabel(12, "N_{CSVT} >= 3");
    me->setBinLabel(13, "N_{CSVT} >= 4");
  }

  double GetMass(const double E, const double px, const double py, const double pz){
    const double rx=px/E;
    const double ry=py/E;
    const double rz=pz/E;
    return E*sqrt(1.0-rx*rx-ry*ry-rz*rz);
  }
}

SUSY_HLT_SingleLepton::SUSY_HLT_SingleLepton(const edm::ParameterSet &ps):
  theElectronTag_(ps.getParameter<edm::InputTag>("ElectronCollection")),
  theElectronCollection_(consumes<reco::GsfElectronCollection>(theElectronTag_)),
  theMuonTag_(ps.getParameter<edm::InputTag>("MuonCollection")),
  theMuonCollection_(consumes<reco::MuonCollection>(theMuonTag_)),
  thePfMETTag_(ps.getParameter<edm::InputTag>("pfMETCollection")),
  thePfMETCollection_(consumes<reco::PFMETCollection>(thePfMETTag_)),
  thePfJetTag_(ps.getParameter<edm::InputTag>("pfJetCollection")),
  thePfJetCollection_(consumes<reco::PFJetCollection>(thePfJetTag_)),
  theJetTagTag_(ps.getParameter<edm::InputTag>("jetTagCollection")),
  theJetTagCollection_(consumes<reco::JetTagCollection>(theJetTagTag_)),
  theTriggerResultsTag_(ps.getParameter<edm::InputTag>("TriggerResults")),
  theTriggerResults_(consumes<edm::TriggerResults>(theTriggerResultsTag_)),
  theTrigSummaryTag_(ps.getParameter<edm::InputTag>("trigSummary")),
  theTrigSummary_(consumes<trigger::TriggerEvent>(theTrigSummaryTag_)),
  fHltConfig_(),
  HLTProcess_(ps.getParameter<std::string>("HLTProcess")),
  triggerPath_(ps.getParameter<std::string>("TriggerPath")),
  triggerPathAuxiliary_(ps.getParameter<std::string>("TriggerPathAuxiliary")),
  triggerFilter_(ps.getParameter<edm::InputTag>("TriggerFilter")),
  jetPtCut_(ps.getUntrackedParameter<double>("JetPtCut")),
  jetEtaCut_(ps.getUntrackedParameter<double>("JetEtaCut")),
  leptonPtCut_(ps.getUntrackedParameter<double>("LeptonPtCut")),
  leptonPtPlateau_(ps.getUntrackedParameter<double>("LeptonPtPlateau")),
  htPlateau_(ps.getUntrackedParameter<double>("HtPlateau")),
  metPlateau_(ps.getUntrackedParameter<double>("MetPlateau")),
  csvPlateau_(ps.getUntrackedParameter<double>("CsvPlateau")),
  h_triggerLepPt_(nullptr),
  h_triggerLepEta_(nullptr),
  h_triggerLepPhi_(nullptr),
  h_CSVTurnOn_num_(nullptr),
  h_CSVTurnOn_den_(nullptr),
  h_pfMetTurnOn_num_(nullptr),
  h_pfMetTurnOn_den_(nullptr),
  h_pfHTTurnOn_num_(nullptr),
  h_pfHTTurnOn_den_(nullptr),
  h_leptonTurnOn_num_(nullptr),
  h_leptonTurnOn_den_(nullptr){
  edm::LogInfo("SUSY_HLT_SingleLepton") << "Constructor SUSY_HLT_SingleLepton::SUSY_HLT_SingleLepton " << std::endl;
  }

SUSY_HLT_SingleLepton::~SUSY_HLT_SingleLepton(){
  edm::LogInfo("SUSY_HLT_SingleLepton") << "Destructor SUSY_HLT_SingleLepton::~SUSY_HLT_SingleLepton " << std::endl;
}

void SUSY_HLT_SingleLepton::dqmBeginRun(const edm::Run &run, const edm::EventSetup &e){ 
  bool changed;
  
  if(!fHltConfig_.init(run, e, HLTProcess_, changed)){
    edm::LogError("SUSY_HLT_SingleLepton") << "Initialization of HLTConfigProvider failed!!";
    return;
  }

  bool pathFound = false;
  const std::vector<std::string> allTrigNames = fHltConfig_.triggerNames();
  for(size_t j = 0; j <allTrigNames.size(); ++j){
    if(allTrigNames[j].find(triggerPath_) != std::string::npos){
      pathFound = true;
    }
  }

  if(!pathFound){
    edm::LogError ("SUSY_HLT_SingleLepton") << "Path not found: " << triggerPath_ << std::endl;
    return;
  }

  edm::LogInfo("SUSY_HLT_SingleLepton") << "SUSY_HLT_SingleLepton::beginRun" << std::endl;
}

void SUSY_HLT_SingleLepton::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run &, const edm::EventSetup &){
  edm::LogInfo("SUSY_HLT_SingleLepton") << "SUSY_HLT_SingleLepton::bookHistograms" << std::endl;
  //book at beginRun
  ibooker.cd();
  ibooker.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);

  //offline quantities

  //online quantities 
  h_triggerLepPt_ = ibooker.book1D("triggerLepPt", "Trigger Lepton p_{T};Lepton p_{T} [GeV]", 20, 0.0, 500.0);
  h_triggerLepEta_ = ibooker.book1D("triggerLepEta", "Trigger Lepton Eta;Lepton #eta", 20, -3.0, 3.0);
  h_triggerLepPhi_ = ibooker.book1D("triggerLepPhi", "Trigger Lepton Phi;Lepton #phi", 20, -3.5, 3.5);

  //num and den hists to be divided in harvesting step to make turn on curves
  h_leptonTurnOn_num_ = ibooker.book1D("leptonTurnOn_num", "Lepton p_{T} Turn On Numerator;Offline lepton p_{T} [GeV]", 30, 0.0, 150 );
  h_leptonTurnOn_den_ = ibooker.book1D("leptonTurnOn_den", "Lepton p_{T} Turn On Denominator;Offline lepton p_{T} [GeV]", 30, 0.0, 150.0 );
  h_pfHTTurnOn_num_ = ibooker.book1D("pfHTTurnOn_num", "PF H_{T} Turn On Numerator;Offline H_{T} [GeV]", 30, 0.0, 1500.0 );
  h_pfHTTurnOn_den_ = ibooker.book1D("pfHTTurnOn_den", "PF H_{T} Turn On Denominator;Offline H_{T} [GeV]", 30, 0.0, 1500.0 );
  h_pfMetTurnOn_num_ = ibooker.book1D("pfMetTurnOn_num", "PF MET Turn On Numerator;Offline MET [GeV]", 20, 0.0, 500.0 );
  h_pfMetTurnOn_den_ = ibooker.book1D("pfMetTurnOn_den", "PF MET Turn On Denominator;Offline MET [GeV]", 20, 0.0, 500.0 );
  h_CSVTurnOn_num_ = ibooker.book1D("CSVTurnOn_num", "CSV Turn On Numerator;Offline CSV Requirements", 13, -0.5, 12.5);
  h_CSVTurnOn_den_ = ibooker.book1D("CSVTurnOn_den", "CSV Turn On Denominator;Offline CSV Requirements", 13, -0.5, 12.5);

  SetBinLabels(h_CSVTurnOn_num_);
  SetBinLabels(h_CSVTurnOn_den_);

  ibooker.cd();
}

void SUSY_HLT_SingleLepton::beginLuminosityBlock(const edm::LuminosityBlock &lumiSeg,
						 const edm::EventSetup &context){
  edm::LogInfo("SUSY_HLT_SingleLepton") << "SUSY_HLT_SingleLepton::beginLuminosityBlock" << std::endl;
}

void SUSY_HLT_SingleLepton::analyze(const edm::Event &e, const edm::EventSetup &eSetup){
  edm::LogInfo("SUSY_HLT_SingleLepton") << "SUSY_HLT_SingleLepton::analyze" << std::endl;

  //MET
  edm::Handle<reco::PFMETCollection> pfMETCollection;
  if(thePfMETTag_.label() != ""){
    e.getByToken(thePfMETCollection_, pfMETCollection);
    if( !pfMETCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton") << "Invalid PFMETCollection: " << thePfMETTag_.label() << '\n';
    }
  }

  //Jets
  edm::Handle<reco::PFJetCollection> pfJetCollection;
  if(thePfJetTag_.label() != ""){
    e.getByToken (thePfJetCollection_,pfJetCollection);
    if( !pfJetCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton") << "Invalid PFJetCollection: " << thePfJetTag_.label() << '\n';
    }
  }

  //b-tags
  edm::Handle<reco::JetTagCollection> jetTagCollection;
  if(theJetTagTag_.label() != ""){
    e.getByToken(theJetTagCollection_, jetTagCollection);
    if( !jetTagCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton") << "Invalid JetTagCollection: " << theJetTagTag_.label() << '\n';
    }
  }
  
  //Electron
  edm::Handle<reco::GsfElectronCollection> ElectronCollection;
  if(theElectronTag_.label() != ""){
    e.getByToken (theElectronCollection_, ElectronCollection);
    if( !ElectronCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton") << "Invalid GsfElectronCollection: " << theElectronTag_.label() << '\n';
    }
  }
  
  //Muon
  edm::Handle<reco::MuonCollection> MuonCollection;
  if(theMuonTag_.label() != ""){ 
    e.getByToken (theMuonCollection_, MuonCollection);
    if( !MuonCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton") << "Invalid MuonCollection: " << theMuonTag_.label() << '\n';
    }
  }
  
  //Trigger
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(theTriggerResults_, hltresults);
  if(!hltresults.isValid()){
    edm::LogWarning("SUSY_HLT_SingleLepton") << "Invalid TriggerResults: " << theTriggerResultsTag_.label() << '\n';
  }
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  e.getByToken(theTrigSummary_, triggerSummary);
  if(!triggerSummary.isValid()){
    edm::LogWarning("SUSY_HLT_SingleLepton") << "Invalid TriggerEvent: " << theTrigSummaryTag_.label() << '\n';
  }

  //Get online objects
  std::vector<float> ptLepton, etaLepton, phiLepton;
  if(triggerSummary.isValid()){
    size_t filterIndex = triggerSummary->filterIndex( triggerFilter_ );
    trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();
    if( !(filterIndex >= triggerSummary->sizeFilters()) ){
      const trigger::Keys &keys = triggerSummary->filterKeys( filterIndex );
      for( size_t j = 0; j < keys.size(); ++j ){
	trigger::TriggerObject foundObject = triggerObjects[keys[j]];

	h_triggerLepPt_->Fill(foundObject.pt());
	h_triggerLepEta_->Fill(foundObject.eta());
	h_triggerLepPhi_->Fill(foundObject.phi());

	ptLepton.push_back(foundObject.pt());
	etaLepton.push_back(foundObject.eta());
	phiLepton.push_back(foundObject.phi());
      }
    }
  }

  //Test whether main and auxilliary triggers fired
  bool hasFired = false;
  bool hasFiredAuxiliary = false;
  if(hltresults.isValid()){
    const edm::TriggerNames &trigNames = e.triggerNames(*hltresults);
    for( unsigned int hltIndex = 0; hltIndex < trigNames.size(); ++hltIndex ){
      if(trigNames.triggerName(hltIndex)==triggerPath_
	  && hltresults->wasrun(hltIndex)
	  && hltresults->accept(hltIndex)){
	hasFired = true;
      }

      if(trigNames.triggerName(hltIndex)==triggerPathAuxiliary_
	  && hltresults->wasrun(hltIndex)
	  && hltresults->accept(hltIndex)){
	hasFiredAuxiliary = true;
      }
    }
  }

  //Fill DQM plots if event is of interest
  if(hasFiredAuxiliary || triggerPathAuxiliary_=="" || !e.isRealData()){
    std::vector<bool> have_matched_ele(ptLepton.size(), false);
    std::vector<size_t> matched_ele(ptLepton.size(), 0);
    std::vector<bool> have_matched_mu(ptLepton.size(), false);
    std::vector<size_t> matched_mu(ptLepton.size(), 0);

    float max_lep_pt = -1.0;

    //Loop over trigger leptons
    for(size_t ihlt = 0; ihlt < ptLepton.size(); ++ihlt){
      float min_dr = std::numeric_limits<float>::max();

      //Try to match a reco electron
      if(ElectronCollection.isValid()){
	for(size_t ireco = 0; ireco < ElectronCollection->size(); ++ireco){
	  const reco::GsfElectron &reco_ele = ElectronCollection->at(ireco);
	  const float dr = deltaR(reco_ele.eta(), reco_ele.phi(), etaLepton.at(ihlt), phiLepton.at(ihlt));
	  if(dr < min_dr && dr < 0.5){
	    have_matched_mu.at(ihlt) = false;
	    have_matched_ele.at(ihlt) = true;
	    matched_ele.at(ihlt) = ireco;
	    if(reco_ele.pt() > max_lep_pt) max_lep_pt = reco_ele.pt();
	    min_dr = dr;
	  }
	}
      }
	
      //Try to match a reco muon
      if(MuonCollection.isValid()){
	for(size_t ireco = 0; MuonCollection.isValid() && ireco < MuonCollection->size(); ++ireco){
	  const reco::Muon &reco_mu = MuonCollection->at(ireco);
	  const float dr = deltaR(reco_mu.eta(), reco_mu.phi(), etaLepton.at(ihlt), phiLepton.at(ihlt));
	  if(dr < min_dr && dr < 0.5){
	    have_matched_mu.at(ihlt) = true;
	    have_matched_ele.at(ihlt) = false;
	    matched_mu.at(ihlt) = ireco;
	    if(reco_mu.pt() > max_lep_pt) max_lep_pt = reco_mu.pt();
	    min_dr = dr;
	  }
	}
      }
    }
     
    double pfHT = -1.0;
    if(pfJetCollection.isValid()){
      pfHT=0.0;
      for(reco::PFJetCollection::const_iterator i_pfjet = pfJetCollection->begin(); i_pfjet != pfJetCollection->end(); ++i_pfjet){
	if(i_pfjet->pt() < jetPtCut_) continue;
	if(fabs(i_pfjet->eta()) > jetEtaCut_) continue;
	pfHT += i_pfjet->pt();
      }
    }

    double pfMET = -1.0;
    if(pfMETCollection.isValid() && pfMETCollection->size()) pfMET = pfMETCollection->front().et();

    float maxCSV = -1.0;
    unsigned num_csvl = 0;
    unsigned num_csvm = 0;
    unsigned num_csvt = 0;
    if(jetTagCollection.isValid()){
      for(const auto &jet: *jetTagCollection){
	const float CSV = jet.second;
	if(jet.first->pt()>jetPtCut_){
	  if(CSV>maxCSV){
	    maxCSV=CSV;
	  }
	  if(CSV>0.244){
	    ++num_csvl;
	    if(CSV>0.679){
	      ++num_csvm;
	      if(CSV>0.898){
		++num_csvt;
	      }
	    }
	  }
	}
      }
    }

    const bool lep_plateau = (max_lep_pt > leptonPtPlateau_ || leptonPtPlateau_<=0.0);
    const bool ht_plateau = (pfHT > htPlateau_ || htPlateau_<=0.0);
    const bool met_plateau = (pfMET > metPlateau_ || metPlateau_<=0.0);
    const bool csv_plateau = (maxCSV > csvPlateau_ || csvPlateau_<=0.0);

    //Fill lepton pt efficiency plot
    if(ht_plateau && met_plateau && csv_plateau){
      const double z_mass = 91.1876;
      if(ElectronCollection.isValid()){
	for(unsigned i = 0; i < matched_ele.size(); ++i){
	  if(!have_matched_ele.at(i)) continue;
	  const unsigned iprobe = matched_ele.at(i);
	  const auto &probe = ElectronCollection->at(iprobe);
	  for(unsigned itag = 0; itag < ElectronCollection->size(); ++itag){
	    if(itag==iprobe) continue;
	    const auto &tag = ElectronCollection->at(itag);
	    if(tag.pt()<leptonPtCut_) continue;

	    const double mass = GetMass(probe.energy()+tag.energy(),
					probe.px()+tag.px(),
					probe.py()+tag.py(),
					probe.pz()+tag.pz());
	    if(fabs(mass-z_mass) > 10.0 || probe.charge()*tag.charge()>0) continue;

	    h_leptonTurnOn_den_->Fill(probe.pt());
	    if(hasFired) h_leptonTurnOn_num_->Fill(probe.pt());
	  }
	}
      }
      if(MuonCollection.isValid()){
	for(unsigned i = 0; i < matched_mu.size(); ++i){
	  if(!have_matched_mu.at(i)) continue;
	  const unsigned iprobe = matched_mu.at(i);
	  const auto &probe = MuonCollection->at(iprobe);
	  for(unsigned itag = 0; itag < MuonCollection->size(); ++itag){
	    if(itag==iprobe) continue;
	    const auto &tag = MuonCollection->at(itag);
	    if(tag.pt()<leptonPtCut_) continue;

	    const double mass = GetMass(probe.energy()+tag.energy(),
					probe.px()+tag.px(),
					probe.py()+tag.py(),
					probe.pz()+tag.pz());
	    if(fabs(mass-z_mass) > 10.0 || probe.charge()*tag.charge()>0) continue;

	    h_leptonTurnOn_den_->Fill(probe.pt());
	    if(hasFired) h_leptonTurnOn_num_->Fill(probe.pt());
	  }
	}
      }
    }

    //Fill HT efficiency plot
    if(lep_plateau && met_plateau && csv_plateau){
      h_pfHTTurnOn_den_->Fill(pfHT);
      if(hasFired) h_pfHTTurnOn_num_->Fill(pfHT);
    }

    //Fill MET efficiency plot
    if(lep_plateau && ht_plateau && csv_plateau){
      h_pfMetTurnOn_den_->Fill(pfMET);
      if(hasFired) h_pfMetTurnOn_num_->Fill(pfMET);
    }

    //Fill CSV efficiency plot
    if(lep_plateau && ht_plateau && met_plateau){
      switch(num_csvl){
      default: h_CSVTurnOn_den_->Fill(4);
      case 3 : h_CSVTurnOn_den_->Fill(3);
      case 2 : h_CSVTurnOn_den_->Fill(2);
      case 1 : h_CSVTurnOn_den_->Fill(1);
      case 0 : h_CSVTurnOn_den_->Fill(0);
      }
      switch(num_csvm){
      default: h_CSVTurnOn_den_->Fill(8);
      case 3 : h_CSVTurnOn_den_->Fill(7);
      case 2 : h_CSVTurnOn_den_->Fill(6);
      case 1 : h_CSVTurnOn_den_->Fill(5);
      case 0 : break;//Don't double count in the no tag bin
      }
      switch(num_csvt){
      default: h_CSVTurnOn_den_->Fill(12);
      case 3 : h_CSVTurnOn_den_->Fill(11);
      case 2 : h_CSVTurnOn_den_->Fill(10);
      case 1 : h_CSVTurnOn_den_->Fill(9);
      case 0 : break;//Don't double count in the no tag bin
      }
      if(hasFired){
	switch(num_csvl){
	default: h_CSVTurnOn_num_->Fill(4);
	case 3 : h_CSVTurnOn_num_->Fill(3);
	case 2 : h_CSVTurnOn_num_->Fill(2);
	case 1 : h_CSVTurnOn_num_->Fill(1);
	case 0 : h_CSVTurnOn_num_->Fill(0);
	}
	switch(num_csvm){
	default: h_CSVTurnOn_num_->Fill(8);
	case 3 : h_CSVTurnOn_num_->Fill(7);
	case 2 : h_CSVTurnOn_num_->Fill(6);
	case 1 : h_CSVTurnOn_num_->Fill(5);
	case 0 : break;//Don't double count in the no tag bin
	}
	switch(num_csvt){
	default: h_CSVTurnOn_num_->Fill(12);
	case 3 : h_CSVTurnOn_num_->Fill(11);
	case 2 : h_CSVTurnOn_num_->Fill(10);
	case 1 : h_CSVTurnOn_num_->Fill(9);
	case 0 : break;//Don't double count in the no tag bin
	}
      }
    }
  }
}

void SUSY_HLT_SingleLepton::endLuminosityBlock(const edm::LuminosityBlock &lumiSeg, const edm::EventSetup &eSetup){
  edm::LogInfo("SUSY_HLT_SingleLepton") << "SUSY_HLT_SingleLepton::endLuminosityBlock" << std::endl;
}

void SUSY_HLT_SingleLepton::endRun(const edm::Run &run, const edm::EventSetup &eSetup){
  edm::LogInfo("SUSY_HLT_SingleLepton") << "SUSY_HLT_SingleLepton::endRun" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_SingleLepton);
