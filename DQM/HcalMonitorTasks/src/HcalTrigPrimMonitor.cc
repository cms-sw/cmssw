#include "DQM/HcalMonitorTasks/interface/HcalTrigPrimMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"

HcalTrigPrimMonitor::HcalTrigPrimMonitor() {}

HcalTrigPrimMonitor::~HcalTrigPrimMonitor() 
{
}

void HcalTrigPrimMonitor::reset(){}

void HcalTrigPrimMonitor::clearME(){

  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();
  }
} // void HcalTrigPrimMonitor::clearME()


void HcalTrigPrimMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"TrigPrimMonitor";

  ZSAlarmThreshold_ = ps.getUntrackedParameter<int>("TrigPrimMonitor_ZSAlarmThreshold", 0);

  if ( m_dbe !=NULL ) {    

    std::string type;
    m_dbe->setCurrentFolder(baseFolder_);

    //------------- Summary -------------------------
    type = "TrigPrim Event Number";
    meEVT_ = m_dbe->bookInt(type);
    meEVT_->Fill(ievt_);
    meTOTALEVT_ = m_dbe->bookInt("TrigPrim Total Events Processed");
    meTOTALEVT_->Fill(tevt_);
    
    type = "Summary";
    Summary_ = m_dbe->book2D(type, type, 2, 0, 2, 2, 0, 2);
    Summary_->setBinLabel(1, "Good");
    Summary_->setBinLabel(2, "Bad");
    Summary_->setBinLabel(1, "HBHE", 2);
    Summary_->setBinLabel(2, "HF", 2);

    type = "Summary for ZS run";
    SummaryZS_ = m_dbe->book2D(type, type, 2, 0, 2, 2, 0, 2);
    SummaryZS_->setBinLabel(1, "Good");
    SummaryZS_->setBinLabel(2, "Bad");
    SummaryZS_->setBinLabel(1, "HBHE", 2);
    SummaryZS_->setBinLabel(2, "HF", 2);

    type = "Error Flag";
    ErrorFlagSummary_ = m_dbe->book2D(type, type, kNErrorFlag, 0, kNErrorFlag, 2, 0, 2);
    ErrorFlagSummary_->setBinLabel(1, "Matched");
    ErrorFlagSummary_->setBinLabel(2, "Mismatched E");
    ErrorFlagSummary_->setBinLabel(3, "Mismatched FG");
    ErrorFlagSummary_->setBinLabel(4, "Data Only");
    ErrorFlagSummary_->setBinLabel(5, "Emul Only");
    ErrorFlagSummary_->setBinLabel(6, "Missing Data");
    ErrorFlagSummary_->setBinLabel(7, "Missing Emul");
    ErrorFlagSummary_->setBinLabel(1, "HBHE", 2);
    ErrorFlagSummary_->setBinLabel(2, "HF", 2);

    type = "Error Flag for ZS run";
    ErrorFlagSummaryZS_ = m_dbe->book2D(type, type, kNErrorFlag, 0, kNErrorFlag, 2, 0, 2);
    ErrorFlagSummaryZS_->setBinLabel(1, "Matched");
    ErrorFlagSummaryZS_->setBinLabel(2, "Mismatched E");
    ErrorFlagSummaryZS_->setBinLabel(3, "Mismatched FG");
    ErrorFlagSummaryZS_->setBinLabel(4, "Data Only");
    ErrorFlagSummaryZS_->setBinLabel(5, "Emul Only");
    ErrorFlagSummaryZS_->setBinLabel(6, "Missing Data");
    ErrorFlagSummaryZS_->setBinLabel(7, "Missing Emul");
    ErrorFlagSummaryZS_->setBinLabel(1, "HBHE", 2);
    ErrorFlagSummaryZS_->setBinLabel(2, "HF", 2);

    type = "EtCorr HBHE";
    EtCorr_[0] = m_dbe->book2D(type,type,50,0,256,50,0,256);

    type = "EtCorr HF";
    EtCorr_[1] = m_dbe->book2D(type,type,50,0,100,50,0,100);
    //---------------------------------------------

    //-------------- TP Occupancy ----------------
    m_dbe->setCurrentFolder(baseFolder_ + "/TP Map");

    type = "TP Occupancy";
    TPOccupancy_= m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    type = "Non Zero TP";
    TPOccupancyEta_ = m_dbe->book1D("TPOccupancyVsEta","TP Occupancy Vs. Eta", etaBins_,etaMin_,etaMax_);
    TPOccupancyPhi_ = m_dbe->book1D("TPOccupancyVsPhi","TP Occupancy Vs. Phi", phiBins_,phiMin_,phiMax_);
    NonZeroTP_ = m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    type = "Matched TP";
    MatchedTP_ = m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    type = "Mismatched Et";
    MismatchedEt_ = m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    type = "Mismatched FG";
    MismatchedFG_ = m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    type = "Data Only";
    DataOnly_ = m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    type = "Emul Only";
    EmulOnly_ = m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    type = "Missing Data";
    MissingData_ = m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    type = "Missing Emul";
    MissingEmul_ = m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    //---------------------------------------------

    //------------ Energy Plots ------------------
    // HBHE
    m_dbe->setCurrentFolder(baseFolder_ + "/Energy Plots/HBHE");

    type = "Energy HBHE - All Data";
    EnergyPlotsAllData_[0] = m_dbe->book1D(type, type, 256, 0, 256);
    type = "Energy HBHE - All Emul";
    EnergyPlotsAllEmul_[0] = m_dbe->book1D(type, type, 256, 0, 256);
    type = "Energy HBHE - Mismatched FG";
    EnergyPlotsMismatchedFG_[0] = m_dbe->book1D(type, type, 256, 0, 256);
    type = "Energy HBHE - Data Only";
    EnergyPlotsDataOnly_[0] = m_dbe->book1D(type, type, 256, 0, 256);
    type = "Energy HBHE - Emul Only";
    EnergyPlotsEmulOnly_[0] = m_dbe->book1D(type, type, 256, 0, 256);
    type = "Energy HBHE - Missing Data";
    EnergyPlotsMissingData_[0] = m_dbe->book1D(type, type, 256, 0, 256);
    type = "Energy HBHE - Missing Emul";
    EnergyPlotsMissingEmul_[0] = m_dbe->book1D(type, type, 256, 0, 256);

    // HF
    m_dbe->setCurrentFolder(baseFolder_ + "/Energy Plots/HF");

    type = "Energy HF - All Data";
    EnergyPlotsAllData_[1] = m_dbe->book1D(type, type, 256, 0, 256);
    type = "Energy HF - All Emul";
    EnergyPlotsAllEmul_[1] = m_dbe->book1D(type, type, 256, 0, 256);
    type = "Energy HF - Mismatched FG";
    EnergyPlotsMismatchedFG_[1] = m_dbe->book1D(type, type, 256, 0, 256);
    type = "Energy HF - Data Only";
    EnergyPlotsDataOnly_[1] = m_dbe->book1D(type, type, 256, 0, 256);
    type = "Energy HF - Emul Only";
    EnergyPlotsEmulOnly_[1] = m_dbe->book1D(type, type, 256, 0, 256);
    type = "Energy HF - Missing Data";
    EnergyPlotsMissingData_[1] = m_dbe->book1D(type, type, 256, 0, 256);
    type = "Energy HF - Missing Emul";
    EnergyPlotsMissingEmul_[1] = m_dbe->book1D(type, type, 256, 0, 256);
  }

  return;
} // void HcalTrigPrimMonitor::setup(...)

void HcalTrigPrimMonitor::processEvent(const HBHERecHitCollection& hbHits, 
				       const HORecHitCollection& hoHits, 
				       const HFRecHitCollection& hfHits,
				       const HBHEDigiCollection& hbhedigi,
				       const HODigiCollection& hodigi,
				       const HFDigiCollection& hfdigi,
                               const HcalTrigPrimDigiCollection& tpDigis,
                               const HcalTrigPrimDigiCollection& emultpDigis,
                               const FEDRawDataCollection& rawraw,
				       const HcalElectronicsMap& emap
				       )
{

  if(!m_dbe) { 
    if (fVerbosity>0) cout <<"HcalTrigPrimMonitor::processEvent   DQMStore not instantiated!!!"<<endl;  
    return; 
  }

  HcalBaseMonitor::processEvent();

  buildFrontEndErrMap(rawraw, emap);
  ErrorFlagPerEvent_[0] = 0;
  ErrorFlagPerEvent_[1] = 0;
  ErrorFlagPerEventZS_[0] = 0;
  ErrorFlagPerEventZS_[1] = 0;
  
  for (HcalTrigPrimDigiCollection::const_iterator data = tpDigis.begin();
                                                  data != tpDigis.end();
                                                  ++data){


    int IsHF = (data->id().ietaAbs() >= 29) ? 1 : 0;
    int ieta = data->id().ieta();
    int iphi = data->id().iphi();

    TPOccupancy_->Fill(ieta, iphi);
    TPOccupancyEta_->Fill(ieta);
    TPOccupancyPhi_->Fill(iphi);
    // Check missing from emulator
    HcalTrigPrimDigiCollection::const_iterator emul = emultpDigis.find(data->id());
    if (emul == emultpDigis.end()) {
      ErrorFlagPerEvent_[IsHF] |= (0x1 << kMissingEmul);
      MissingEmul_->Fill(ieta, iphi);

      int dataTPmax = 0;
      for (int i=0; i<data->size(); ++i) {
        int dataEt = data->sample(i).compressedEt();
        if (dataEt > dataTPmax) dataTPmax = dataEt;
        EnergyPlotsMissingEmul_[IsHF]->Fill(dataEt);
      }
      // Check ZS Alarm Threshold
      if (dataTPmax > ZSAlarmThreshold_) ErrorFlagPerEventZS_[IsHF] |= (0x1 << kMissingEmul);
      continue;
    }
    
    if (FrontEndErrors.find( data->id().rawId() ) != FrontEndErrors.end()) {
      //Front End Format Error
      continue;
    }


    for (int i=0; i < data->size(); ++i){
      int dataEt = data->sample(i).compressedEt();
      int dataFG = data->sample(i).fineGrain();
      int emulEt = emul->sample(i).compressedEt();
      int emulFG = emul->sample(i).fineGrain();

      if (dataEt > 0) NonZeroTP_->Fill(ieta, iphi);

      //Determine Error Flag
      //Default: kUnkown
      ErrorFlag errflag = kUnknown;
      if (dataEt == emulEt){
        if (dataFG == emulFG){
          if (dataEt == 0) errflag = kZeroTP; //Matched:  zero TP
            else errflag = kMatched; //Matched:  non-zeor TP
          }
        else errflag = kMismatchedFG; // Mismatched FG bit
      } //end if (dataEt == emulEt)
      else{
        if (dataEt == 0) errflag = kEmulOnly;
        else if (emulEt == 0 ) errflag = kDataOnly;
        else errflag = kMismatchedEt;
      }

      //TODO: Message log Unknown errflag for debug
      // Fill histogram for non-zero TPs
      switch (errflag){
        case kMatched:
          MatchedTP_->Fill(ieta, iphi);
          EtCorr_[IsHF]->Fill(dataEt, emulEt);
          break;
        case kMismatchedEt:
          MismatchedEt_->Fill(ieta, iphi);
          EtCorr_[IsHF]->Fill(dataEt, emulEt);
          break;
        case kMismatchedFG:
          MismatchedFG_->Fill(ieta, iphi);
          EtCorr_[IsHF]->Fill(dataEt, emulEt);
          EnergyPlotsMismatchedFG_[IsHF]->Fill(dataEt);
          break;
        case kDataOnly:
          DataOnly_->Fill(ieta, iphi);
          EnergyPlotsDataOnly_[IsHF]->Fill(dataEt);
        case kEmulOnly:
          EmulOnly_->Fill(ieta, iphi);
          EnergyPlotsEmulOnly_[IsHF]->Fill(emulEt);
        default:
          break;
      }

      // Other plots
      if (errflag != kZeroTP){
        // Don't count in ErrorFlagPerEvent_ for matched TP.
        if (errflag != kMatched) ErrorFlagPerEvent_[IsHF] |= (0x1 << errflag);
        EnergyPlotsAllData_[IsHF]->Fill(dataEt);
        EnergyPlotsAllEmul_[IsHF]->Fill(emulEt);
      }
    }
    
    // Check ZS Alarm Threshold
    // Check peak shift +/- 1 SOI for HBHE
    // max(|emul.et - data.et|) for HF
    if (IsHF){
      int maxdiff = 0;
      for (int i=0; i<data->size(); ++i){
        int diff = abs(data->sample(i).compressedEt() - emul->sample(i).compressedEt());
        if (diff > maxdiff) maxdiff = diff;
      }
      if (maxdiff > ZSAlarmThreshold_) ErrorFlagPerEventZS_[IsHF] |= (0x1 << kMismatchedEt);
    }
    else {
      int dataEtSOI = data->SOI_compressedEt();
      int mindiff = 0;
      for (int i=data->presamples()-1; i<=data->presamples()+1 && i<emul->size(); ++i){
        if (i<0) continue;
        int diff = abs(emul->sample(i).compressedEt() - dataEtSOI);
        if (diff < mindiff) mindiff = diff;
      }
      if (mindiff > ZSAlarmThreshold_) ErrorFlagPerEventZS_[IsHF] |= (0x1 << kMismatchedEt);
    }
  }

  // Checking missing from data
  for (HcalTrigPrimDigiCollection::const_iterator emul = emultpDigis.begin();
                                                  emul != emultpDigis.end();
                                                  ++emul){

    int IsHF = (emul->id().ietaAbs() >= 29) ? 1 : 0;
    int ieta = emul->id().ieta();
    int iphi = emul->id().iphi();

    HcalTrigPrimDigiCollection::const_iterator data = tpDigis.find(emul->id());
    if ( data ==  emultpDigis.end() ) {
      ErrorFlagPerEvent_[IsHF] |= (0x1 << kMissingData);
      MissingData_->Fill(ieta, iphi);
      int emulTPmax = 0;
      for (int i=0; i<emul->size(); ++i){
        int emulEt = emul->sample(i).compressedEt();
        if (emulEt > emulTPmax) emulTPmax = emulEt;
        EnergyPlotsMissingData_[IsHF]->Fill(emulEt);
      }
      if (emulTPmax > ZSAlarmThreshold_) ErrorFlagPerEventZS_[IsHF] |= (0x1 << kMissingData);
    }
  } //loop emul TP digis

  // Fill Summary (Per Event)
  for (unsigned int IsHF=0; IsHF<=1; ++IsHF){
    if (ErrorFlagPerEvent_[IsHF] == 0) {
      // 0 - Good = No Error Flag per event
      // 1 - Bad
      Summary_->Fill(0, IsHF);
      ErrorFlagSummary_->Fill(kMatched, IsHF);
    }
    else Summary_->Fill(1, IsHF);
    if (ErrorFlagPerEventZS_[IsHF] == 0) {
      SummaryZS_->Fill(0, IsHF);
      ErrorFlagSummaryZS_->Fill(kMatched, IsHF);
    }
    else SummaryZS_->Fill(1, IsHF);
    for (unsigned int i=1; i<kNErrorFlag; ++i){
      if ((ErrorFlagPerEvent_[IsHF] & (0x1 << i)) > 0) ErrorFlagSummary_->Fill(i,IsHF);
      if ((ErrorFlagPerEventZS_[IsHF] & (0x1 << i)) > 0) ErrorFlagSummaryZS_->Fill(i,IsHF);
    }
  } // Fill Summary
  return;
}

void HcalTrigPrimMonitor::buildFrontEndErrMap(const FEDRawDataCollection& rawraw, const HcalElectronicsMap& emap){
  //Front End Format Errors
  FrontEndErrors.clear();
  for(int i=FEDNumbering::getHcalFEDIds().first; i<=FEDNumbering::getHcalFEDIds().second; ++i) {
    const FEDRawData& raw = rawraw.FEDData(i);
    if (raw.size()<12) continue;
    const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
    if(!dccHeader) continue;
    HcalHTRData htr;
    for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
      if (!dccHeader->getSpigotPresent(spigot)) continue;
      dccHeader->getSpigotData(spigot,htr,raw.size());
      int dccid = dccHeader->getSourceId();
      int errWord = htr.getErrorsWord() & 0x1FFFF;
      //bool HTRError = (!htr.check() || htr.isHistogramEvent() || (errWord ^ 0x8000)!=0);
      bool HTRError = (!htr.check() || htr.isHistogramEvent() || (errWord & 0x800)!=0);

      if(HTRError) {
        bool valid =false;
        for(int fchan=0; fchan<3 && !valid; fchan++) {
          for(int fib=0; fib<9 && !valid; fib++) {
            HcalElectronicsId eid(fchan,fib,spigot,dccid-FEDNumbering::getHcalFEDIds().first);
            eid.setHTR(htr.readoutVMECrateId(),htr.htrSlot(),htr.htrTopBottom());
            DetId detId = emap.lookup(eid);
            if(detId.null()) continue;
            HcalSubdetector subdet=(HcalSubdetector(detId.subdetId()));
            if (detId.det()!=4||
              (subdet!=HcalBarrel && subdet!=HcalEndcap &&
              subdet!=HcalForward )) continue;
            std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry.towerIds(detId);
            for (std::vector<HcalTrigTowerDetId>::const_iterator triggerId=ids.begin(); triggerId != ids.end(); ++triggerId) {
              FrontEndErrors.insert(triggerId->rawId());
            }
            //valid = true;
          }
        }
      }
    }
  }
}
