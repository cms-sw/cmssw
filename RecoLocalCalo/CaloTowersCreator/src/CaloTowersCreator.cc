#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowersCreator.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoLocalCalo/CaloTowersCreator/interface/EScales.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
// severity level for ECAL
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"

const std::vector<double>& 
CaloTowersCreator::getGridValues()
{
  static std::vector<double> retval;
  
  if (retval.size() == 0)
    {
      retval.push_back(0.);
      retval.push_back(10.);
      retval.push_back(20.);
      retval.push_back(30.);
      retval.push_back(40.);
      retval.push_back(50.);
      retval.push_back(100.);
      retval.push_back(1000.); 
    }

  return retval;
}


CaloTowersCreator::CaloTowersCreator(const edm::ParameterSet& conf) : 
  algo_(conf.getParameter<double>("EBThreshold"),
	      conf.getParameter<double>("EEThreshold"),

	conf.getParameter<bool>("UseEtEBTreshold"),
	conf.getParameter<bool>("UseEtEETreshold"),
	conf.getParameter<bool>("UseSymEBTreshold"),
	conf.getParameter<bool>("UseSymEETreshold"),


	      conf.getParameter<double>("HcalThreshold"),
	      conf.getParameter<double>("HBThreshold"),
	      conf.getParameter<double>("HESThreshold"),
	      conf.getParameter<double>("HEDThreshold"),
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
        conf.getParameter<double>("MomEEDepth")
	),

  hbheLabel_(conf.getParameter<edm::InputTag>("hbheInput")),
  hoLabel_(conf.getParameter<edm::InputTag>("hoInput")),
  hfLabel_(conf.getParameter<edm::InputTag>("hfInput")),
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
  EBEScale=EScales.EBScale; 
  EEEScale=EScales.EEScale; 
  HBEScale=EScales.HBScale; 
  HESEScale=EScales.HESScale; 
  HEDEScale=EScales.HEDScale; 
  HOEScale=EScales.HOScale; 
  HF1EScale=EScales.HF1Scale; 
  HF2EScale=EScales.HF2Scale; 

  // get the Ecal severities to be excluded
  const std::vector<std::string> severitynames = 
    conf.getParameter<std::vector<std::string> >("EcalRecHitSeveritiesToBeExcluded");

   theEcalSeveritiesToBeExcluded_ =  StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynames);

  // get the Ecal severities to be used for bad towers
   theEcalSeveritiesToBeUsedInBadTowers_ =  
     StringToEnumValue<EcalSeverityLevel::SeverityLevel>(conf.getParameter<std::vector<std::string> >("EcalSeveritiesToBeUsedInBadTowers") );

  if (EScales.instanceLabel=="") produces<CaloTowerCollection>();
  else produces<CaloTowerCollection>(EScales.instanceLabel);
}

void CaloTowersCreator::produce(edm::Event& e, const edm::EventSetup& c) {
  // get the necessary event setup objects...
  edm::ESHandle<CaloGeometry> pG;
  edm::ESHandle<HcalTopology> htopo;
  edm::ESHandle<CaloTowerConstituentsMap> cttopo;
  c.get<CaloGeometryRecord>().get(pG);
  c.get<IdealGeometryRecord>().get(htopo);
  c.get<IdealGeometryRecord>().get(cttopo);
 
  // ECAL channel status map ****************************************
  edm::ESHandle<EcalChannelStatus> ecalChStatus;
  c.get<EcalChannelStatusRcd>().get( ecalChStatus );
  const EcalChannelStatus* dbEcalChStatus = ecalChStatus.product();
 
  // HCAL channel status map ****************************************
  edm::ESHandle<HcalChannelQuality> hcalChStatus;    
  c.get<HcalChannelQualityRcd>().get( hcalChStatus );
  const HcalChannelQuality* dbHcalChStatus = hcalChStatus.product();

  if (!dbHcalChStatus->topo()) dbHcalChStatus->setTopo(htopo.product());

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
  algo_.setGeometry(cttopo.product(),htopo.product(),pG.product());

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

  // ----------------------------------------------------------
  // For ecal error handling need to 
  // have access to the EB and EE collections at the end of 
  // tower reconstruction.

  edm::Handle<EcalRecHitCollection> ebHandle;
  edm::Handle<EcalRecHitCollection> eeHandle;

  for (std::vector<edm::InputTag>::const_iterator i=ecalLabels_.begin(); 
       i!=ecalLabels_.end(); i++) {
    
    edm::Handle<EcalRecHitCollection> ec_tmp;
    
    if (! e.getByLabel(*i,ec_tmp) ) continue;
    if (ec_tmp->size()==0) continue;

    // check if this is EB or EE
    if ( (ec_tmp->begin()->detid()).subdetId() == EcalBarrel ) {
      ebHandle = ec_tmp;
    }
    else if ((ec_tmp->begin()->detid()).subdetId() == EcalEndcap ) {
      eeHandle = ec_tmp;
    }

  }

  algo_.setEbHandle(ebHandle);
  algo_.setEeHandle(eeHandle);

  //-----------------------------------------------------------



  bool present;

  // Step A/C: Get Inputs and process (repeatedly)
  edm::Handle<HBHERecHitCollection> hbhe;
  present=e.getByLabel(hbheLabel_,hbhe);
  if (present || !allowMissingInputs_)  algo_.process(*hbhe);

  edm::Handle<HORecHitCollection> ho;
  present=e.getByLabel(hoLabel_,ho);
  if (present || !allowMissingInputs_) algo_.process(*ho);

  edm::Handle<HFRecHitCollection> hf;
  present=e.getByLabel(hfLabel_,hf);
  if (present || !allowMissingInputs_) algo_.process(*hf);

  std::vector<edm::InputTag>::const_iterator i;
  for (i=ecalLabels_.begin(); i!=ecalLabels_.end(); i++) {
    edm::Handle<EcalRecHitCollection> ec;
    present=e.getByLabel(*i,ec);
    if (present || !allowMissingInputs_) algo_.process(*ec);
  }

  // Step B: Create empty output
  std::auto_ptr<CaloTowerCollection> prod(new CaloTowerCollection());

  // Step C: Process
  algo_.finish(*prod);

  // Step D: Put into the event
  if (EScales.instanceLabel=="") e.put(prod);
  else e.put(prod,EScales.instanceLabel);


}

