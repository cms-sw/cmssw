#include "DQM/HcalMonitorTasks/interface/HcalZDCMonitor.h"

static const float HFQIEConst = 2.6;

HcalZDCMonitor::HcalZDCMonitor() {
}
HcalZDCMonitor::~HcalZDCMonitor() {
}
void HcalZDCMonitor::reset() {
}

void HcalZDCMonitor::setup(const edm::ParameterSet & ps, DQMStore * dbe) {
    HcalBaseMonitor::setup(ps, dbe);

    baseFolder_ = rootFolder_ + "ZDCMonitor_Hcal";

    if (showTiming) {
        cpu_timer.reset();
        cpu_timer.start();
    }

    if (fVerbosity > 0)
        std::cout << "<HcalZDCMonitor::setup>  Setting up histograms" << std::endl;

    if (fVerbosity > 1)
        std::cout << "<HcalZDCMonitor::setup> Getting variable values from cfg files" << std::endl;

    // Set initial event # to 0
    ievt_ = 0;

//Histograms
    if (m_dbe) {
        if (fVerbosity > 1)
            std::cout << "<HcalZDCMonitor::setup>  Setting up Histograms" << std::endl;

        m_dbe->setCurrentFolder(baseFolder_);
        meEVT_ = m_dbe->bookInt("ZDC Event Number");
        meEVT_->Fill(ievt_);
        char name[128];
        char title[128];
	
        h_channel_side_charge_Ave = m_dbe->book2D("2D_DigiChargeAve", "Ave. Charge", 2, 0, 2, 9, 0, 9);
        h_channel_side_charge_Ave->setBinLabel(1,"ZDC+",1);
        h_channel_side_charge_Ave->setBinLabel(2,"ZDC-",1);
        h_channel_side_charge_Ave->setBinLabel(1,"EM1",2);
        h_channel_side_charge_Ave->setBinLabel(2,"EM2",2);
        h_channel_side_charge_Ave->setBinLabel(3,"EM3",2);
        h_channel_side_charge_Ave->setBinLabel(4,"EM4",2);
        h_channel_side_charge_Ave->setBinLabel(5,"EM5",2);
        h_channel_side_charge_Ave->setBinLabel(6,"HAD1",2);
        h_channel_side_charge_Ave->setBinLabel(7,"HAD2",2);
        h_channel_side_charge_Ave->setBinLabel(8,"HAD3",2);
        h_channel_side_charge_Ave->setBinLabel(9,"HAD4",2);

        h_channel_side_TSMean_Ave = m_dbe->book2D("2D_TSMeanAve", "Ave. TSMean", 2, 0, 2, 9, 0, 9);
        h_channel_side_TSMean_Ave->setBinLabel(1,"ZDC+",1);
        h_channel_side_TSMean_Ave->setBinLabel(2,"ZDC-",1);
        h_channel_side_TSMean_Ave->setBinLabel(1,"EM1",2);
        h_channel_side_TSMean_Ave->setBinLabel(2,"EM2",2);
        h_channel_side_TSMean_Ave->setBinLabel(3,"EM3",2);
        h_channel_side_TSMean_Ave->setBinLabel(4,"EM4",2);
        h_channel_side_TSMean_Ave->setBinLabel(5,"EM5",2);
        h_channel_side_TSMean_Ave->setBinLabel(6,"HAD1",2);
        h_channel_side_TSMean_Ave->setBinLabel(7,"HAD2",2);
        h_channel_side_TSMean_Ave->setBinLabel(8,"HAD3",2);
        h_channel_side_TSMean_Ave->setBinLabel(9,"HAD4",2);

        h_channel_side_energy_Ave = m_dbe->book2D("2D_RecHitEnergyAve", "Ave. RechitEnergy", 2, 0, 2, 9, 0, 9);
        h_channel_side_energy_Ave->setBinLabel(1,"ZDC+",1);
        h_channel_side_energy_Ave->setBinLabel(2,"ZDC-",1);
        h_channel_side_energy_Ave->setBinLabel(1,"EM1",2);
        h_channel_side_energy_Ave->setBinLabel(2,"EM2",2);
        h_channel_side_energy_Ave->setBinLabel(3,"EM3",2);
        h_channel_side_energy_Ave->setBinLabel(4,"EM4",2);
        h_channel_side_energy_Ave->setBinLabel(5,"EM5",2);
        h_channel_side_energy_Ave->setBinLabel(6,"HAD1",2);
        h_channel_side_energy_Ave->setBinLabel(7,"HAD2",2);
        h_channel_side_energy_Ave->setBinLabel(8,"HAD3",2);
        h_channel_side_energy_Ave->setBinLabel(9,"HAD4",2);

        h_channel_side_RecHitTime_Ave = m_dbe->book2D("2D_RecHitTimeAve", "Average RechitTiming", 2, 0, 2, 9, 0, 9);
        h_channel_side_RecHitTime_Ave->setBinLabel(1,"ZDC+",1);
        h_channel_side_RecHitTime_Ave->setBinLabel(2,"ZDC-",1);
        h_channel_side_RecHitTime_Ave->setBinLabel(1,"EM1",2);
        h_channel_side_RecHitTime_Ave->setBinLabel(2,"EM2",2);
        h_channel_side_RecHitTime_Ave->setBinLabel(3,"EM3",2);
        h_channel_side_RecHitTime_Ave->setBinLabel(4,"EM4",2);
        h_channel_side_RecHitTime_Ave->setBinLabel(5,"EM5",2);
        h_channel_side_RecHitTime_Ave->setBinLabel(6,"HAD1",2);
        h_channel_side_RecHitTime_Ave->setBinLabel(7,"HAD2",2);
        h_channel_side_RecHitTime_Ave->setBinLabel(8,"HAD3",2);
        h_channel_side_RecHitTime_Ave->setBinLabel(9,"HAD4",2);
	
        m_dbe->setCurrentFolder(baseFolder_ + "/Digis");

        for (int i = 0; i < 5; ++i) {
            // pulse Plus Side 
            sprintf(title, "h_ZDCP_EMChan_%i_Pulse", i + 1);
            sprintf(name, "ZDC Plus EM Section Pulse for channel %i", i + 1);
            h_ZDCP_EM_Pulse[i] = m_dbe->book1D(title, name, 10, -0.5, 9.5);
	    h_ZDCP_EM_Pulse[i]->setAxisTitle("Time Slice id",1);
	    h_ZDCP_EM_Pulse[i]->setAxisTitle("Ave. Pulse Height",2);
            // pulse Minus Side
            sprintf(title, "h_ZDCM_EMChan_%i_Pulse", i + 1);
            sprintf(name, "ZDC Minus EM Section Pulse for channel %i", i + 1);
            h_ZDCM_EM_Pulse[i] = m_dbe->book1D(title, name, 10, -0.5, 9.5);
	    h_ZDCM_EM_Pulse[i]->setAxisTitle("Time Slice id",1);
	    h_ZDCM_EM_Pulse[i]->setAxisTitle("Ave. Pulse Height",2);
            // integrated charge over 10 time samples
            sprintf(title, "h_ZDCP_EMChan_%i_Charge", i + 1);
            sprintf(name, "ZDC Plus EM Section Charge for channel %i", i + 1);
            h_ZDCP_EM_Charge[i] = m_dbe->book1D(title, name, 110, -100., 1000.);
	    h_ZDCP_EM_Charge[i]->setAxisTitle("Charge {fC}",1);
	    h_ZDCP_EM_Charge[i]->setAxisTitle("Events",2);
            // integrated charge over 10 time samples
            sprintf(title, "h_ZDCM_EMChan_%i_Charge", i + 1);
            sprintf(name, "ZDC Minus EM Section Charge for channel %i", i + 1);
            h_ZDCM_EM_Charge[i] = m_dbe->book1D(title, name, 110, -100., 1000.);
	    h_ZDCM_EM_Charge[i]->setAxisTitle("Charge {fC}",1);
	    h_ZDCM_EM_Charge[i]->setAxisTitle("Events",2);
            // charge weighted time slice
            sprintf(title, "h_ZDCP_EMChan_%i_TSMean", i + 1);
            sprintf(name, "ZDC Plus EM Section TSMean for channel %i", i + 1);
            h_ZDCP_EM_TSMean[i] = m_dbe->book1D(title, name, 100, -0.5, 9.5);
	    h_ZDCP_EM_TSMean[i]->setAxisTitle("Timing",1);
	    h_ZDCP_EM_TSMean[i]->setAxisTitle("Events",2);
            // charge weighted time slice
            sprintf(title, "h_ZDCM_EMChan_%i_TSMean", i + 1);
            sprintf(name, "ZDC Minus EM Section TSMean for channel %i", i + 1);
            h_ZDCM_EM_TSMean[i] = m_dbe->book1D(title, name, 100, -0.5, 9.5);
	    h_ZDCM_EM_TSMean[i]->setAxisTitle("Timing",1);
	    h_ZDCM_EM_TSMean[i]->setAxisTitle("Events",2);
        }

        for (int i = 0; i < 4; ++i) {
            // pulse Plus Side 
            sprintf(title, "h_ZDCP_HADChan_%i_Pulse", i + 1);
            sprintf(name, "ZDC Plus HAD Section Pulse for channel %i", i + 1);
            h_ZDCP_HAD_Pulse[i] = m_dbe->book1D(title, name, 10, -0.5, 9.5);
	    h_ZDCP_HAD_Pulse[i]->setAxisTitle("Time Slice id",1);
	    h_ZDCP_HAD_Pulse[i]->setAxisTitle("Ave. Pulse Height",2);
            // pulse Minus Side 
            sprintf(title, "h_ZDCM_HADChan_%i_Pulse", i + 1);
            sprintf(name, "ZDC Minus HAD Section Pulse for channel %i", i + 1);
            h_ZDCM_HAD_Pulse[i] = m_dbe->book1D(title, name, 10, -0.5, 9.5);
	    h_ZDCP_HAD_Pulse[i]->setAxisTitle("Time Slice id",1);
	    h_ZDCP_HAD_Pulse[i]->setAxisTitle("Ave. Pulse Height",2);
            // integrated charge over 10 time samples 
            sprintf(title, "h_ZDCP_HADChan_%i_Charge", i + 1);
            sprintf(name, "ZDC Plus HAD Section Charge for channel %i", i + 1);
            h_ZDCP_HAD_Charge[i] = m_dbe->book1D(title, name, 110, -100., 1000.);
	    h_ZDCP_HAD_Charge[i]->setAxisTitle("Charge {fC}",1);
	    h_ZDCP_HAD_Charge[i]->setAxisTitle("Events",2);
            // integrated charge over 10 time samples 
            sprintf(title, "h_ZDCM_HADChan_%i_Charge", i + 1);
            sprintf(name, "ZDC Minus HAD Section Charge for channel %i", i + 1);
            h_ZDCM_HAD_Charge[i] = m_dbe->book1D(title, name, 110, -100., 1000.);
	    h_ZDCM_HAD_Charge[i]->setAxisTitle("Charge {fC}",1);
	    h_ZDCM_HAD_Charge[i]->setAxisTitle("Events",2);
            // charge weighted time slice 
            sprintf(title, "h_ZDCP_HADChan_%i_TSMean", i + 1);
            sprintf(name, "ZDC Plus HAD Section TSMean for channel %i", i + 1);
            h_ZDCP_HAD_TSMean[i] = m_dbe->book1D(title, name, 100, -0.5, 9.5);
	    h_ZDCP_HAD_TSMean[i]->setAxisTitle("Timing",1);
	    h_ZDCP_HAD_TSMean[i]->setAxisTitle("Events",2);
            // charge weighted time slice 
            sprintf(title, "h_ZDCM_HADChan_%i_TSMean", i + 1);
            sprintf(name, "ZDC Minus HAD Section TSMean for channel %i", i + 1);
            h_ZDCM_HAD_TSMean[i] = m_dbe->book1D(title, name, 100, -0.5, 9.5);
	    h_ZDCM_HAD_TSMean[i]->setAxisTitle("Timing",1);
	    h_ZDCM_HAD_TSMean[i]->setAxisTitle("Events",2);
        }

        m_dbe->setCurrentFolder(baseFolder_ + "/RecHits");

        for (int i = 0; i < 5; ++i) {
	    //RecHitEnergy Plus Side
            sprintf(title,"h_ZDCP_EMChan_%i_RecHit_Energy",i+1);
	    sprintf(name,"ZDC EM Section Rechit Energy for channel %i",i+1);
	    h_ZDCP_EM_RecHitEnergy[i] = m_dbe->book1D(title, name, 110, -100., 1000.);
	    h_ZDCP_EM_RecHitEnergy[i]->setAxisTitle("Energy (GeV)",1);
	    h_ZDCP_EM_RecHitEnergy[i]->setAxisTitle("Events",2);
	    //RecHitEnergy Minus Side
	    sprintf(title,"h_ZDCM_EMChan_%i_RecHit_Energy",i+1);
	    sprintf(name,"ZDC EM Section Rechit Energy for channel %i",i+1);
	    h_ZDCM_EM_RecHitEnergy[i] = m_dbe->book1D(title, name, 110, -100., 1000.);
	    h_ZDCM_EM_RecHitEnergy[i]->setAxisTitle("Energy (GeV)",1);
	    h_ZDCM_EM_RecHitEnergy[i]->setAxisTitle("Events",2);
	    //RecHit Timing Plus Side 
	    sprintf(title,"h_ZDCP_EMChan_%i_RecHit_Timing",i+1);
	    sprintf(name,"ZDC EM Section Rechit Timing for channel %i",i+1);
	    h_ZDCP_EM_RecHitTiming[i] = m_dbe->book1D(title, name, 100, -100., 100.);
	    h_ZDCP_EM_RecHitTiming[i]->setAxisTitle("RecHit Time",1);
	    h_ZDCP_EM_RecHitTiming[i]->setAxisTitle("Events",2);
	    //RecHit Timing Minus Side 
	    sprintf(title,"h_ZDCM_EMChan_%i_RecHit_Timing",i+1);
	    sprintf(name,"ZDC EM Section Rechit Timing for channel %i",i+1);
	    h_ZDCM_EM_RecHitTiming[i] = m_dbe->book1D(title, name, 100, -100., 100.);	
	    h_ZDCM_EM_RecHitTiming[i]->setAxisTitle("RecHit Time",1);
	    h_ZDCM_EM_RecHitTiming[i]->setAxisTitle("Events",2);
	}

        for (int i = 0; i < 4; ++i) {
	    //RecHitEnergy Plus Side
	    sprintf(title,"h_ZDCP_HADChan_%i_RecHit_Energy",i+1);
	    sprintf(name,"ZDC HAD Section Rechit Energy for channel %i",i+1);
	    h_ZDCP_HAD_RecHitEnergy[i] = m_dbe->book1D(title, name, 110, -100., 1000.);
	    h_ZDCP_HAD_RecHitEnergy[i]->setAxisTitle("Energy (GeV)",1);
	    h_ZDCP_HAD_RecHitEnergy[i]->setAxisTitle("Events",2);
	    //RecHitEnergy Minus Side
	    sprintf(title,"h_ZDCM_HADChan_%i_RecHit_Energy",i+1);
	    sprintf(name,"ZDC HAD Section Rechit Energy for channel %i",i+1);
	    h_ZDCM_HAD_RecHitEnergy[i] = m_dbe->book1D(title, name, 110, -100., 1000.);
	    h_ZDCM_HAD_RecHitEnergy[i]->setAxisTitle("Energy (GeV)",1);
	    h_ZDCM_HAD_RecHitEnergy[i]->setAxisTitle("Events",2);
	    //RecHit Timing Plus Side 
	    sprintf(title,"h_ZDCP_HADChan_%i_RecHit_Timing",i+1);
	    sprintf(name,"ZDC HAD Section Rechit Timing for channel %i",i+1);
	    h_ZDCP_HAD_RecHitTiming[i] = m_dbe->book1D(title, name, 100, -100., 100.);	
	    h_ZDCP_HAD_RecHitTiming[i]->setAxisTitle("RecHit Time",1);
	    h_ZDCP_HAD_RecHitTiming[i]->setAxisTitle("Events",2);
	    //RecHit Timing Minus Side 
	    sprintf(title,"h_ZDCM_HADChan_%i_RecHit_Timing",i+1);
	    sprintf(name,"ZDC HAD Section Rechit Timing for channel %i",i+1);
	    h_ZDCM_HAD_RecHitTiming[i] = m_dbe->book1D(title, name, 100, -100., 100.);	
	    h_ZDCM_HAD_RecHitTiming[i]->setAxisTitle("RecHit Time",1);
	    h_ZDCM_HAD_RecHitTiming[i]->setAxisTitle("Events",2);
	}

    }
    return;
}

void HcalZDCMonitor::processEvent(const ZDCDigiCollection& digi, const ZDCRecHitCollection& rechit) {
    if (fVerbosity > 0)
        std::cout << "<HcalZDCMonitor::processEvent> Processing Event..." << std::endl;
    if (showTiming) 
      {
        cpu_timer.reset();
        cpu_timer.start();
    }
    ++ievt_;
    meEVT_->Fill(ievt_);

    //--------------------------------------
    // ZDC Digi part 
    //--------------------------------------
    double fSum = 0.;
    std::vector<double> fData;

    for (ZDCDigiCollection::const_iterator digi_iter = digi.begin(); 
	 digi_iter != digi.end(); ++digi_iter) 
      {
	const ZDCDataFrame digi = (const ZDCDataFrame) (*digi_iter);
	//HcalZDCDetId id(digi_iter->id());
	int iSide = digi_iter->id().zside();
	int iSection = digi_iter->id().section();
	int iChannel = digi_iter->id().channel();
	unsigned int fTS = digi_iter->size();
	while (fData.size()<fTS)
	  fData.push_back(-999);
	while (fData.size()>fTS)
	  fData.pop_back(); // delete last elements 

	fSum = 0.;
	for (unsigned int i = 0; i < fTS; ++i) 
	  {
	    //fData[i] = HFQIEConst * digi[i].nominal_fC();  //  !! important. adc or nominal fC ?  multiple with QIE const ? 
	    //fData[i] = digi[i].adc();
	    // For now, use nominal_fC, without HFQIEConstant
	    fData[i]=digi[i].nominal_fC();
	  }
      
	if (!isGood(fData, 10, 0.001))
	  continue; // Decide the signal quality !!!!! How should we set the cut ? Ask ZDCers. 
      
	double fTSMean = getTime(fData, fSum);
	//std::cout << "Side= " << iSide << " Section= " << iSection << " Channel= " << iChannel << "\tCharge\t" << fSum <<std::endl; 
      
	if (iSection == 1) 
	  {    // EM
	    if (iSide == 1) {   // Plus
	      for (unsigned int i = 0; i < fTS; ++i) {
		h_ZDCP_EM_Pulse[iChannel - 1]->Fill(i, fData[i]/10.);
	      }
	      h_ZDCP_EM_Charge[iChannel - 1]->Fill(fSum);
	      h_ZDCP_EM_TSMean[iChannel - 1]->Fill(fTSMean);
	    } // Plus
	    if (iSide == -1) {  // Minus
	      for (unsigned int i = 0; i < fTS; ++i) {
		h_ZDCM_EM_Pulse[iChannel - 1]->Fill(i, fData[i]/10.);
	      }
	      h_ZDCM_EM_Charge[iChannel - 1]->Fill(fSum);
	      h_ZDCM_EM_TSMean[iChannel - 1]->Fill(fTSMean);
	    } // Minus
	  }// EM
      
	else if (iSection == 2) 
	  {    // HAD
	    if (iSide == 1) {   // Plus 
	      for (unsigned int i = 0; i < fTS; ++i) {
		h_ZDCP_HAD_Pulse[iChannel - 1]->Fill(i, fData[i]/10.);
	      }
	      h_ZDCP_HAD_Charge[iChannel - 1]->Fill(fSum);
	      h_ZDCP_HAD_TSMean[iChannel - 1]->Fill(fTSMean);
	    } // Plus
	    if (iSide == -1) {  // Minus
	      for (unsigned int i = 0; i < fTS; ++i) {
		h_ZDCM_HAD_Pulse[iChannel - 1]->Fill(i, fData[i]/10.);
	      }
	      h_ZDCM_HAD_Charge[iChannel - 1]->Fill(fSum);
	      h_ZDCM_HAD_TSMean[iChannel - 1]->Fill(fTSMean);
	    }// Minus
	  } // HAD 
      } // loop on zdc digi collection

    //--------------------------------------
    // ZDC RecHit part 
    //--------------------------------------
    for (ZDCRecHitCollection::const_iterator rechit_iter = rechit.begin(); 
	 rechit_iter != rechit.end(); ++rechit_iter)
      {		
	HcalZDCDetId id(rechit_iter->id());
	int Side      = (rechit_iter->id()).zside();
	int Section   = (rechit_iter->id()).section();
	int Channel   = (rechit_iter->id()).channel();
	//std::cout << "RecHitEnergy  " << zhit->energy() << "  RecHitTime  " << zhit->time() << std::endl;
	if(Section==1)
	  { //EM
	    if (Side ==1 ){ // Plus
	      h_ZDCP_EM_RecHitEnergy[Channel-1]->Fill(rechit_iter->energy());
	      h_ZDCP_EM_RecHitTiming[Channel-1]->Fill(rechit_iter->time());
	    }
	    if (Side == -1 ){ //Minus
	      h_ZDCM_EM_RecHitEnergy[Channel-1]->Fill(rechit_iter->energy());
	      h_ZDCM_EM_RecHitTiming[Channel-1]->Fill(rechit_iter->time());
	    }
	  } //EM
	else if(Section==2)
	  { //HAD
	    if (Side ==1 ){ //Plus
	      h_ZDCP_HAD_RecHitEnergy[Channel-1]->Fill(rechit_iter->energy());
	      h_ZDCP_HAD_RecHitTiming[Channel-1]->Fill(rechit_iter->time());
	    }
	    if (Side == -1 ){ //Minus
	      h_ZDCM_HAD_RecHitEnergy[Channel-1]->Fill(rechit_iter->energy());
	      h_ZDCM_HAD_RecHitTiming[Channel-1]->Fill(rechit_iter->time());
	    }
	  } // HAD
      } // loop on rechits
    
} // end of event processing 

bool HcalZDCMonitor::isGood(std::vector<double>fData, double fCut, double fPercentage) {
  bool dec = false;
  int ts_max = -1;
  
  ts_max = getTSMax(fData);
  if (ts_max == 0 || ts_max == (int)(fData.size() - 1))
    return false;
  float sum = fData[ts_max - 1] + fData[ts_max + 1];
  
  // cout << "tsMax " << ts_max << " data[tsmax] " << mData[ts_max] << " sum " << sum << endl;
  if (fData[ts_max] > fCut && sum > (fData[ts_max] * fPercentage))
    dec = true;
  return dec;
} // bool HcalZDCMonitor::isGood

int HcalZDCMonitor::getTSMax(std::vector<double>fData) 
{
  int ts_max = -100;
  double max = -999.;
  
  for (unsigned int j = 0; j < fData.size(); ++j) {
    if (max < fData[j]) {
      max = fData[j];
      ts_max = j;
    }
  }
  return ts_max;
} // int HcalZDCMonitor::getTSMax()

double HcalZDCMonitor::getTime(std::vector<double>fData, double &fSum) {
  int ts_max = getTSMax(fData);
  double Time = 0,
    SumT = 0;             // ,MaxT=-10;
  
  if (ts_max >= 0) {
    Time = ts_max * fData[ts_max];
    SumT = fData[ts_max];
    if (ts_max > 0) {
      Time += (ts_max - 1) * fData[ts_max - 1];
      SumT += fData[ts_max - 1];
    }
    if (ts_max < (int)(fData.size() - 1)) {
      Time += (ts_max + 1) * fData[ts_max + 1];
      SumT += fData[ts_max + 1];
    }
            Time = Time / SumT;
  }
  if (SumT > 0.)
    fSum = SumT;
  
  return Time;
}


void HcalZDCMonitor::endLuminosityBlock() 
{

    for (int i = 0; i < 5; ++i) {   // EM Channels 
	// ZDC Plus 
	//h_ZDCP_EM_Pulse[i]->Scale(10. / h_ZDCP_EM_Pulse[i]->getEntries());
	h_channel_side_charge_Ave->setBinContent(1, i + 1, h_ZDCP_EM_Charge[i]->getMean());
	h_channel_side_TSMean_Ave->setBinContent(1, i + 1, h_ZDCP_EM_TSMean[i]->getMean());
	h_channel_side_energy_Ave->setBinContent(1, i + 1, h_ZDCP_EM_RecHitEnergy[i]->getMean());
	h_channel_side_RecHitTime_Ave->setBinContent(1, i + 1, h_ZDCP_EM_RecHitTiming[i]->getMean());
	// ZDC Minus
	//h_ZDCM_EM_Pulse[i]->Scale(10. / h_ZDCM_EM_Pulse[i]->getEntries());
	h_channel_side_charge_Ave->setBinContent(2, i + 1, h_ZDCM_EM_Charge[i]->getMean());
	h_channel_side_TSMean_Ave->setBinContent(2, i + 1, h_ZDCM_EM_TSMean[i]->getMean());
	h_channel_side_energy_Ave->setBinContent(2, i + 1, h_ZDCM_EM_RecHitEnergy[i]->getMean());
	h_channel_side_RecHitTime_Ave->setBinContent(2, i + 1, h_ZDCM_EM_RecHitTiming[i]->getMean());
    }

    for (int i = 0; i < 4; ++i) {   // HAD channels 
	// ZDC Plus 
	//h_ZDCP_HAD_Pulse[i]->Scale(10. / h_ZDCP_HAD_Pulse[i]->getEntries());
	h_channel_side_charge_Ave->setBinContent(1, i + 6, h_ZDCP_HAD_Charge[i]->getMean());
	h_channel_side_TSMean_Ave->setBinContent(1, i + 6, h_ZDCP_HAD_TSMean[i]->getMean());
	h_channel_side_energy_Ave->setBinContent(1, i + 6, h_ZDCP_HAD_RecHitEnergy[i]->getMean());
	h_channel_side_RecHitTime_Ave->setBinContent(1, i + 6, h_ZDCP_HAD_RecHitTiming[i]->getMean());
	// ZDC Minus
	//h_ZDCM_HAD_Pulse[i]->Scale(10. / h_ZDCM_HAD_Pulse[i]->getEntries());
	h_channel_side_charge_Ave->setBinContent(2, i + 6, h_ZDCM_HAD_Charge[i]->getMean());
	h_channel_side_TSMean_Ave->setBinContent(2, i + 6, h_ZDCM_HAD_TSMean[i]->getMean());
	h_channel_side_energy_Ave->setBinContent(2, i + 6, h_ZDCM_HAD_RecHitEnergy[i]->getMean());
	h_channel_side_RecHitTime_Ave->setBinContent(2, i + 6, h_ZDCM_HAD_RecHitTiming[i]->getMean());
    }
} // void HcalZDCMonitor::endLuminosityBlock()

