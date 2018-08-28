#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowersCreator.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
// severity level for ECAL
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"


CaloTowersCreator::CaloTowersCreator(const edm::ParameterSet& conf) : 
  algo_(conf.getParameter<double>("EBThreshold"),
	conf.getParameter<double>("EEThreshold"),

	conf.getParameter<bool>("UseEtEBTreshold"),
	conf.getParameter<bool>("UseEtEETreshold"),
	conf.getParameter<bool>("UseSymEBTreshold"),
	conf.getParameter<bool>("UseSymEETreshold"),


	conf.getParameter<double>("HcalThreshold"),
	conf.getParameter<double>("HBThreshold"),
	conf.getParameter<double>("HBThreshold1"),
	conf.getParameter<double>("HBThreshold2"),
	conf.getParameter<double>("HESThreshold"),
	conf.getParameter<double>("HESThreshold1"),
	conf.getParameter<double>("HEDThreshold"),
	conf.getParameter<double>("HEDThreshold1"),
	conf.getParameter<double>("HOThreshold0"),
	conf.getParameter<double>("HOThresholdPlus1"),
	conf.getParameter<double>("HOThresholdMinus1"),
	conf.getParameter<double>("HOThresholdPlus2"),
	conf.getParameter<double>("HOThresholdMinus2"),
	conf.getParameter<double>("HF1Threshold"),
	conf.getParameter<double>("HF2Threshold"),
        conf.getParameter<std::vector<double> >("EBGrid"),
        conf.getParameter<std::vector<double> >("EBWeights"),
        conf.getParameter<std::vector<double> >("EEGrid"),
        conf.getParameter<std::vector<double> >("EEWeights"),
        conf.getParameter<std::vector<double> >("HBGrid"),
        conf.getParameter<std::vector<double> >("HBWeights"),
        conf.getParameter<std::vector<double> >("HESGrid"),
        conf.getParameter<std::vector<double> >("HESWeights"),
        conf.getParameter<std::vector<double> >("HEDGrid"),
        conf.getParameter<std::vector<double> >("HEDWeights"),
        conf.getParameter<std::vector<double> >("HOGrid"),
        conf.getParameter<std::vector<double> >("HOWeights"),
        conf.getParameter<std::vector<double> >("HF1Grid"),
        conf.getParameter<std::vector<double> >("HF1Weights"),
        conf.getParameter<std::vector<double> >("HF2Grid"),
        conf.getParameter<std::vector<double> >("HF2Weights"),
	conf.getParameter<double>("EBWeight"),
	conf.getParameter<double>("EEWeight"),
	conf.getParameter<double>("HBWeight"),
	conf.getParameter<double>("HESWeight"),
	conf.getParameter<double>("HEDWeight"),
	conf.getParameter<double>("HOWeight"),
	conf.getParameter<double>("HF1Weight"),
	conf.getParameter<double>("HF2Weight"),
	conf.getParameter<double>("EcutTower"),
	conf.getParameter<double>("EBSumThreshold"),
	conf.getParameter<double>("EESumThreshold"),
	conf.getParameter<bool>("UseHO"),
	// (for momentum reconstruction algorithm)
        conf.getParameter<int>("MomConstrMethod"),
        conf.getParameter<double>("MomHBDepth"),
        conf.getParameter<double>("MomHEDepth"),
        conf.getParameter<double>("MomEBDepth"),
        conf.getParameter<double>("MomEEDepth"),
        conf.getParameter<int>("HcalPhase")
	),

  ecalLabels_(conf.getParameter<std::vector<edm::InputTag> >("ecalInputs")),
  allowMissingInputs_(conf.getParameter<bool>("AllowMissingInputs")),

  theHcalAcceptSeverityLevel_(conf.getParameter<unsigned int>("HcalAcceptSeverityLevel")),

  theRecoveredHcalHitsAreUsed_(conf.getParameter<bool>("UseHcalRecoveredHits")),
  theRecoveredEcalHitsAreUsed_(conf.getParameter<bool>("UseEcalRecoveredHits")),

  // paramaters controlling the use of rejected hits

  useRejectedHitsOnly_(conf.getParameter<bool>("UseRejectedHitsOnly")),

  theHcalAcceptSeverityLevelForRejectedHit_(conf.getParameter<unsigned int>("HcalAcceptSeverityLevelForRejectedHit")),


  useRejectedRecoveredHcalHits_(conf.getParameter<bool>("UseRejectedRecoveredHcalHits")),
  useRejectedRecoveredEcalHits_(conf.getParameter<bool>("UseRejectedRecoveredEcalHits"))



{
  algo_.setMissingHcalRescaleFactorForEcal(conf.getParameter<double>("missingHcalRescaleFactorForEcal"));

  // register for data access
  tok_hbhe_ = consumes<HBHERecHitCollection>(conf.getParameter<edm::InputTag>("hbheInput"));
  tok_ho_ = consumes<HORecHitCollection>(conf.getParameter<edm::InputTag>("hoInput"));
  tok_hf_ = consumes<HFRecHitCollection>(conf.getParameter<edm::InputTag>("hfInput"));

  const unsigned nLabels = ecalLabels_.size();
  for ( unsigned i=0; i != nLabels; i++ ) 
    toks_ecal_.push_back(consumes<EcalRecHitCollection>(ecalLabels_[i]));


  EBEScale=eScales_.EBScale; 
  EEEScale=eScales_.EEScale; 
  HBEScale=eScales_.HBScale; 
  HESEScale=eScales_.HESScale; 
  HEDEScale=eScales_.HEDScale; 
  HOEScale=eScales_.HOScale; 
  HF1EScale=eScales_.HF1Scale; 
  HF2EScale=eScales_.HF2Scale; 

  // get the Ecal severities to be excluded
  const std::vector<std::string> severitynames = 
    conf.getParameter<std::vector<std::string> >("EcalRecHitSeveritiesToBeExcluded");

   theEcalSeveritiesToBeExcluded_ =  StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynames);

  // get the Ecal severities to be used for bad towers
   theEcalSeveritiesToBeUsedInBadTowers_ =  
     StringToEnumValue<EcalSeverityLevel::SeverityLevel>(conf.getParameter<std::vector<std::string> >("EcalSeveritiesToBeUsedInBadTowers") );

  if (eScales_.instanceLabel.empty()) produces<CaloTowerCollection>();
  else produces<CaloTowerCollection>(eScales_.instanceLabel);

  /*
  std::cout << "VI Producer " 
	    << (useRejectedHitsOnly_ ? "use rejectOnly " : " ")
	    << (allowMissingInputs_ ? "allowMissing " : " " )
	    <<  nLabels << ' ' << severitynames.size() 
	    << std::endl;
  */
}

void CaloTowersCreator::produce(edm::Event& e, const edm::EventSetup& c) {
  // get the necessary event setup objects...
  edm::ESHandle<CaloGeometry> pG;
  edm::ESHandle<HcalTopology> htopo;
  edm::ESHandle<CaloTowerTopology> cttopo;
  edm::ESHandle<CaloTowerConstituentsMap> ctmap;
  c.get<CaloGeometryRecord>().get(pG);
  c.get<HcalRecNumberingRecord>().get(htopo);
  c.get<HcalRecNumberingRecord>().get(cttopo);
  c.get<CaloGeometryRecord>().get(ctmap);
 
  // ECAL channel status map ****************************************
  edm::ESHandle<EcalChannelStatus> ecalChStatus;
  c.get<EcalChannelStatusRcd>().get( ecalChStatus );
  const EcalChannelStatus* dbEcalChStatus = ecalChStatus.product();
 
  // HCAL channel status map ****************************************
  edm::ESHandle<HcalChannelQuality> hcalChStatus;    
  c.get<HcalChannelQualityRcd>().get( "withTopo", hcalChStatus );
    
  const HcalChannelQuality* dbHcalChStatus = hcalChStatus.product();
    
  // Assignment of severity levels **********************************
  edm::ESHandle<HcalSeverityLevelComputer> hcalSevLvlComputerHndl;
  c.get<HcalSeverityLevelComputerRcd>().get(hcalSevLvlComputerHndl);
  const HcalSeverityLevelComputer* hcalSevLvlComputer = hcalSevLvlComputerHndl.product();

  edm::ESHandle<EcalSeverityLevelAlgo> ecalSevLvlAlgoHndl;
  c.get<EcalSeverityLevelAlgoRcd>().get(ecalSevLvlAlgoHndl);
  const EcalSeverityLevelAlgo* ecalSevLvlAlgo = ecalSevLvlAlgoHndl.product();

  
  algo_.setEBEScale(EBEScale);
  algo_.setEEEScale(EEEScale);
  algo_.setHBEScale(HBEScale);
  algo_.setHESEScale(HESEScale);
  algo_.setHEDEScale(HEDEScale);
  algo_.setHOEScale(HOEScale);
  algo_.setHF1EScale(HF1EScale);
  algo_.setHF2EScale(HF2EScale);
  algo_.setGeometry(cttopo.product(),ctmap.product(),htopo.product(),pG.product());

  // for treatment of problematic and anomalous cells

  algo_.setHcalChStatusFromDB(dbHcalChStatus);
  algo_.setEcalChStatusFromDB(dbEcalChStatus);
   
  algo_.setHcalAcceptSeverityLevel(theHcalAcceptSeverityLevel_);
  algo_.setEcalSeveritiesToBeExcluded(theEcalSeveritiesToBeExcluded_);

  algo_.setRecoveredHcalHitsAreUsed(theRecoveredHcalHitsAreUsed_);
  algo_.setRecoveredEcalHitsAreUsed(theRecoveredEcalHitsAreUsed_);

  algo_.setHcalSevLvlComputer(hcalSevLvlComputer);
  algo_.setEcalSevLvlAlgo(ecalSevLvlAlgo);


  algo_.setUseRejectedHitsOnly(useRejectedHitsOnly_);

  algo_.setHcalAcceptSeverityLevelForRejectedHit(theHcalAcceptSeverityLevelForRejectedHit_);
  algo_.SetEcalSeveritiesToBeUsedInBadTowers (theEcalSeveritiesToBeUsedInBadTowers_);

  algo_.setUseRejectedRecoveredHcalHits(useRejectedRecoveredHcalHits_);
  algo_.setUseRejectedRecoveredEcalHits(useRejectedRecoveredEcalHits_);

  /*
  std::cout << "VI Produce: " 
	    << (useRejectedHitsOnly_ ? "use rejectOnly " : " ")
	    << (allowMissingInputs_ ? "allowMissing " : " " )
	    << (theRecoveredEcalHitsAreUsed_ ? "use RecoveredEcal ": " " )
	    <<  toks_ecal_.size()
	    << ' ' << theEcalSeveritiesToBeExcluded_.size()
	    << ' ' << theEcalSeveritiesToBeUsedInBadTowers_.size() 
	    << std::endl;
  */

  algo_.begin(); // clear the internal buffer

  // can't chain these in a big OR statement, or else it'll
  // get triggered for each of the first three events
  bool check1 = hcalSevLevelWatcher_.check(c);
  bool check2 = hcalChStatusWatcher_.check(c);
  bool check3 = caloTowerConstituentsWatcher_.check(c);
  if(check1 || check2 || check3)
  {
    algo_.makeHcalDropChMap();
  }

  // check ecal SevLev
  if (ecalSevLevelWatcher_.check(c)) algo_.makeEcalBadChs();

  // ----------------------------------------------------------
  // For ecal error handling need to 
  // have access to the EB and EE collections at the end of 
  // tower reconstruction.

  edm::Handle<EcalRecHitCollection> ebHandle;
  edm::Handle<EcalRecHitCollection> eeHandle;

  for (std::vector<edm::EDGetTokenT<EcalRecHitCollection> >::const_iterator i=toks_ecal_.begin(); 
       i!=toks_ecal_.end(); i++) {
    
    edm::Handle<EcalRecHitCollection> ec_tmp;
    
    if (! e.getByToken(*i,ec_tmp) ) continue;
    if (ec_tmp->empty()) continue;

    // check if this is EB or EE
    if ( (ec_tmp->begin()->detid()).subdetId() == EcalBarrel ) {
      ebHandle = ec_tmp;
    }
    else if ((ec_tmp->begin()->detid()).subdetId() == EcalEndcap) {
      eeHandle = ec_tmp;
    }
  }

  algo_.setEbHandle(ebHandle);
  algo_.setEeHandle(eeHandle);

  //-----------------------------------------------------------

  bool present;

  // Step A/C: Get Inputs and process (repeatedly)
  edm::Handle<HBHERecHitCollection> hbhe;
  present=e.getByToken(tok_hbhe_,hbhe);
  if (present || !allowMissingInputs_)  algo_.process(*hbhe);

  edm::Handle<HORecHitCollection> ho;
  present=e.getByToken(tok_ho_,ho);
  if (present || !allowMissingInputs_) algo_.process(*ho);

  edm::Handle<HFRecHitCollection> hf;
  present=e.getByToken(tok_hf_,hf);
  if (present || !allowMissingInputs_) algo_.process(*hf);

  std::vector<edm::EDGetTokenT<EcalRecHitCollection> >::const_iterator i;
  for (i=toks_ecal_.begin(); i!=toks_ecal_.end(); i++) {
    edm::Handle<EcalRecHitCollection> ec;
    present=e.getByToken(*i,ec);
    if (present || !allowMissingInputs_) algo_.process(*ec);
  }

  // Step B: Create empty output
  auto prod = std::make_unique<CaloTowerCollection>();

  // Step C: Process
  algo_.finish(*prod);

  /*
  int totc=0; float totE=0;
  reco::LeafCandidate::LorentzVector totP4;
  for (auto const & tw : (*prod) ) { totc += tw.constituents().size(); totE+=tw.energy(); totP4+=tw.p4();}
  std::cout << "VI " << (*prod).size() << " " << totc << " " << totE << " " << totP4 << std::endl;
  */

  // Step D: Put into the event
  if (eScales_.instanceLabel.empty()) e.put(std::move(prod));
  else e.put(std::move(prod),eScales_.instanceLabel);


}

void CaloTowersCreator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	edm::ParameterSetDescription desc;
	desc.add<double>("EBSumThreshold", 0.2);
	desc.add<double>("HF2Weight", 1.0);
	desc.add<double>("EBWeight", 1.0);
	desc.add<double>("EESumThreshold", 0.45);
	desc.add<double>("HOThreshold0", 1.1);
	desc.add<double>("HOThresholdPlus1", 3.5);
	desc.add<double>("HOThresholdMinus1", 3.5);
	desc.add<double>("HOThresholdPlus2", 3.5);
	desc.add<double>("HOThresholdMinus2", 3.5);
	desc.add<double>("HBThreshold", 0.7);
	desc.add<double>("HBThreshold1", 0.7);
	desc.add<double>("HBThreshold2", 0.7);
	desc.add<double>("HF1Threshold", 0.5);
	desc.add<double>("HEDWeight", 1.0);
	desc.add<double>("EEWeight", 1.0);
	desc.add<double>("HESWeight", 1.0);
	desc.add<double>("HF1Weight", 1.0);
	desc.add<double>("HOWeight", 1.0);
	desc.add<double>("EBThreshold", 0.07);
	desc.add<double>("EEThreshold", 0.3);
	desc.add<double>("HcalThreshold", -1000.0);
	desc.add<double>("HF2Threshold", 0.85);
	desc.add<double>("HESThreshold", 0.8);
	desc.add<double>("HESThreshold1", 0.8);
	desc.add<double>("HEDThreshold", 0.8);
	desc.add<double>("HEDThreshold1", 0.8);
	desc.add<double>("EcutTower", -1000.0);
	desc.add<double>("HBWeight", 1.0);
	desc.add<double>("MomHBDepth", 0.2);
	desc.add<double>("MomHEDepth", 0.4);   
	desc.add<double>("MomEBDepth", 0.3);
	desc.add<double>("MomEEDepth", 0.0);
	desc.add<bool>("UseHO", true);
	desc.add<bool>("UseEtEBTreshold", false);
	desc.add<bool>("UseSymEBTreshold", true);
	desc.add<bool>("UseEtEETreshold", false);
	desc.add<bool>("UseSymEETreshold", true);
	desc.add<bool>("UseHcalRecoveredHits", true);
	desc.add<bool>("UseEcalRecoveredHits", false);
	desc.add<bool>("UseRejectedHitsOnly", false);
	desc.add<bool>("UseRejectedRecoveredHcalHits", true);
	desc.add<bool>("UseRejectedRecoveredEcalHits", false);
	desc.add<double>("missingHcalRescaleFactorForEcal", 0.0);
	desc.add<bool>("AllowMissingInputs", false);
	desc.add<std::vector<double> >("HBGrid", {-1.0, 1.0, 10.0, 100.0, 1000.0});
	desc.add<std::vector<double> >("EEWeights", {1.0, 1.0, 1.0, 1.0, 1.0});
	desc.add<std::vector<double> >("HF2Weights", {1.0, 1.0, 1.0, 1.0, 1.0});
	desc.add<std::vector<double> >("HOWeights", {1.0, 1.0, 1.0, 1.0, 1.0});
	desc.add<std::vector<double> >("EEGrid", {-1.0, 1.0, 10.0, 100.0, 1000.0});
	desc.add<std::vector<double> >("HBWeights", {1.0, 1.0, 1.0, 1.0, 1.0});
	desc.add<std::vector<double> >("HF2Grid", {-1.0, 1.0, 10.0, 100.0, 1000.0});
	desc.add<std::vector<double> >("HEDWeights", {1.0, 1.0, 1.0, 1.0, 1.0});
	desc.add<std::vector<double> >("HF1Grid", {-1.0, 1.0, 10.0, 100.0, 1000.0});
	desc.add<std::vector<double> >("EBWeights", {1.0, 1.0, 1.0, 1.0, 1.0});
	desc.add<std::vector<double> >("HF1Weights", {1.0, 1.0, 1.0, 1.0, 1.0});
	desc.add<std::vector<double> >("HESGrid", {-1.0, 1.0, 10.0, 100.0, 1000.0});
	desc.add<std::vector<double> >("HESWeights", {1.0, 1.0, 1.0, 1.0, 1.0});
	desc.add<std::vector<double> >("HEDGrid", {-1.0, 1.0, 10.0, 100.0, 1000.0});
	desc.add<std::vector<double> >("HOGrid", {-1.0, 1.0, 10.0, 100.0, 1000.0});
	desc.add<std::vector<double> >("EBGrid", {-1.0, 1.0, 10.0, 100.0, 1000.0});
	desc.add<edm::InputTag>("hfInput", edm::InputTag("hfreco"));
	desc.add<edm::InputTag>("hbheInput", edm::InputTag("hbhereco"));
	desc.add<edm::InputTag>("hoInput", edm::InputTag("horeco"));
	desc.add<std::vector<edm::InputTag> >("ecalInputs", {edm::InputTag("ecalRecHit","EcalRecHitsEB"), edm::InputTag("ecalRecHit","EcalRecHitsEE")});
	desc.add<int>("MomConstrMethod", 1);
	desc.add<unsigned int>("HcalAcceptSeverityLevel", 9);
	desc.add<std::vector<std::string> >("EcalRecHitSeveritiesToBeExcluded", {"kTime","kWeird","kBad"});
	desc.add<unsigned int>("HcalAcceptSeverityLevelForRejectedHit", 9999);
	desc.add<std::vector<std::string> >("EcalSeveritiesToBeUsedInBadTowers", {});
	desc.add<int>("HcalPhase", 0);

	descriptions.addDefault(desc);
}
