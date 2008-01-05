#include "DQM/HcalMonitorTasks/interface/HcalDeadCellMonitor.h"
#include <math.h>

using namespace std;

HcalDeadCellMonitor::HcalDeadCellMonitor(){
  ievt_=0;
}

HcalDeadCellMonitor::~HcalDeadCellMonitor() {
  /*
  reset_Nevents(hbHists);
  reset_Nevents(heHists);
  reset_Nevents(hoHists);
  reset_Nevents(hfHists);
  */
}

namespace HcalDeadCellDigiCheck
{

  /*
    CheckForDeadDigis looks for digis with ADCs = 0 
    or digis with ADCs < pedestal+Nsigma_
  */ 
  template<class Digi>
  void CheckForDeadDigis(const Digi& digi, DeadCellHists& hist, 
			 DeadCellHists& all,
			 float Nsigma, float mincount,
			 HcalCalibrations calibs, 
			 HcalCalibrationWidths widths, 
			 DaqMonitorBEInterface* dbe, string baseFolder)
  {
    string type;
    if(hist.type==1) type = "HB";
    else if(hist.type==2) type = "HE"; 
    else if(hist.type==3) type = "HO"; 
    else if(hist.type==4) type = "HF"; 
    else return;
    if(dbe) dbe->setCurrentFolder(baseFolder+"/"+type);


    int ADCsum=0;
    int capADC[4];
    capADC[0]=0;
    capADC[1]=0;
    capADC[2]=0;
    capADC[3]=0;

    // Use offset to show plots of cells at different depths?
    // Maybe add this feature later?
    int offset;
    //offset= (digi.id().depth()-1)*abs(digi.id().ieta())/digi.id().ieta()*2;
    offset=0;

    // Fill (eta,phi) map if digi is found for that cell
    hist.digiCheck->Fill(digi.id().ieta()+offset,digi.id().iphi());
    all.digiCheck->Fill(digi.id().ieta()+offset,digi.id().iphi());

    // Loop over the 10 time slices of the digi
    for (int i=0;i<digi.size();i++)
      {
	ADCsum+=digi.sample(i).adc();
	//if (ADCsum!=0) break;
	int thisCapid = digi.sample(i).capid();

	/* If ADC value above (pedestal+Nsgima_), fill hist.above_pedestal_temp
	   (Cool cells will later be found by looking for empty spots in the
	   above_pedsetal_temp histogram)
	*/
 
	if (digi.sample(i).adc()>calibs.pedestal(thisCapid)+Nsigma*widths.pedestal(thisCapid))
	{
	    hist.above_pedestal_temp->Fill(digi.id().ieta()+offset,digi.id().iphi());
	    all.above_pedestal_temp->Fill(digi.id().ieta()+offset,digi.id().iphi());
	  }
	capADC[thisCapid]+=digi.sample(i).adc();

	// FIXME:  Still need to work on Capid check -- 29 Oct 2007
	if (thisCapid!=(i%4)) // do all digis start with capid of 0 on first slice?
	  {
	    hist.noADC_ID_map->Fill(digi.id().ieta(),digi.id().iphi());
	    hist.noADC_ID_eta->Fill(digi.id().ieta());
	    all.noADC_ID_map->Fill(digi.id().ieta(),digi.id().iphi());
	    all.noADC_ID_eta->Fill(digi.id().ieta());
	    
	  }
	// Not yet sure if this histogram is useful, but it gives an idea of the ADC distributions
	hist.ADCdist->Fill(digi.sample(i).adc());
	all.ADCdist->Fill(digi.sample(i).adc());
	
      }

    // If ADCsum <= mincount, cell is considered dead
    if (ADCsum<=mincount)
      {
	hist.deadADC_map->Fill(digi.id().ieta(),digi.id().iphi());
	hist.deadADC_eta->Fill(digi.id().ieta());
	all.deadADC_map->Fill(digi.id().ieta(),digi.id().iphi());
	all.deadADC_eta->Fill(digi.id().ieta());
      }

    // look for 
    for (int zz=0;zz<4;zz++)
      {
	if (capADC[zz]<=mincount)
	  {
	    hist.deadcapADC_map[zz]->Fill(digi.id().ieta(),digi.id().iphi());
	    all.deadcapADC_map[zz]->Fill(digi.id().ieta(),digi.id().iphi());
	  }
      }
    return;
  }

  /*CheckHits searches HCal hits for cells with energies much less than their
    neighbors'
  */
  template<class Hits>
  void CheckHits(double coolcellfrac, const Hits& hits, 
		 DeadCellHists& hist, DeadCellHists& all, 
		 DaqMonitorBEInterface* dbe, string baseFolder)
  { 
    
    string type;
    if(hist.type==1) type = "HB";
    else if(hist.type==2) type = "HE"; 
    else if(hist.type==3) type = "HO"; 
    else if(hist.type==4) type = "HF"; 
    else {
      //cout <<"<HcalDeadCellMonitor:  CheckHits Error> Hit collection type not specified!"<<endl;
      return;
    }
	
    if(dbe) dbe->setCurrentFolder(baseFolder+"/"+type);

    
    typename Hits::const_iterator _cell;
    for (_cell=hits.begin();
	 _cell!=hits.end(); 
	 _cell++)
      {
	// Allow for offsets in eta for cells with depth >1?
	int offset;
	//offset= (_cell->id().depth()-1)*abs(_cell->id().ieta())/_cell->id().ieta()*2;
	offset=0;

	// Fill histogram if cell found in hist region
	if ((_cell->id().subdet())!=hist.type) continue;

	hist.cellCheck->Fill(_cell->id().ieta()+offset,_cell->id().iphi());
	all.cellCheck->Fill(_cell->id().ieta()+offset,_cell->id().iphi());

	if (_cell->id().depth()==2) continue; // skip depth=2 for now
	//if (vetoCell(_cell->id())) continue;

	// Sum energies of neighbors around cell
	double neighborE=0.;
	int neighbors=0;
	for (typename Hits::const_iterator neighbor=hits.begin();neighbor!=hits.end();neighbor++)
	  {
	    //if (vetoCell(neighbor->id())) continue;
	    if  ((HcalSubdetector)(neighbor->id().subdet())!=(HcalSubdetector)(_cell->id().subdet())) continue;
	    if (neighbor->id().depth()!=_cell->id().depth()) continue;
	    if ( (abs(neighbor->id().iphi()-_cell->id().iphi()))<2 && 
		 (abs(neighbor->id().ieta()-_cell->id().ieta()))<2)
	      {
		neighborE+=neighbor->energy();
		neighbors+=1;
	      }
	  }// for (Hits::const_iterator neighbor=hits.begin()...
	
	// Remove cell energy from neighbor calculation
	neighborE-=_cell->energy();
	neighbors-=1;

	if (_cell->energy()<coolcellfrac*(1.0*neighborE/neighbors))
	  {
	    hist.NADA_cool_cell_map->Fill(_cell->id().ieta(),_cell->id().iphi());
	    all.NADA_cool_cell_map->Fill(_cell->id().ieta(),_cell->id().iphi());
	  }
      } // for (_cell=hits.begin()...)

  } // void CheckHits



} // namespace HcalDeadCellDigiCheck

void HcalDeadCellMonitor::reset(){}


void HcalDeadCellMonitor::setup(const edm::ParameterSet& ps,
				DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  
  baseFolder_ = rootFolder_+"DeadCellMonitor";

  // Get ps parameters here

  // Set input parameters from .cfi file
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  if (fVerbosity) cout << "DeadCell eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  
  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  if (fVerbosity) cout << "DeadCell phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;

  coolcellfrac_ = ps.getUntrackedParameter<double>("coolcellfrac",0.25);
  checkNevents_ = ps.getUntrackedParameter<int>("checkNevents",1000);
  Nsigma_ = ps.getUntrackedParameter<double>("ped_Nsigma",2.);
  minADCcount_ = ps.getUntrackedParameter<double>("minADCcount",0.);
  if (fVerbosity)
    {
      cout <<"DeadCell NADA coolcells must have energy fraction of <"<<coolcellfrac_<<"* (neighbors' average energy)"<<endl;
      cout <<"DeadCell cool digis are checked every "<<checkNevents_<<" events"<<endl;
      cout <<"\tCool digis must have energy <(pedestal + "<<Nsigma_<<"sigma)"<<endl;
      cout <<"DeadCell digis are considered dead if ADC count is <= "<<minADCcount_<<endl;
    }

  ievt_=0;
  if (m_dbe !=NULL) {
    m_dbe->setCurrentFolder(baseFolder_);
    
    meEVT_ = m_dbe->bookInt("DeadCell Task Event Number");    
    meEVT_->Fill(ievt_);


    // HB
    m_dbe->setCurrentFolder(baseFolder_+"/HB");
    hbHists.type=1;
    hbHists.deadADC_map = m_dbe->book2D("HB_deadADCOccupancyMap","HB No ADC Count Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.noADC_ID_map = m_dbe->book2D("HB_noADCIDOccupancyMap","HB No ADC ID Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.deadADC_eta = m_dbe->book1D("HB_deadADCEta","HB No ADC Count Eta ",etaBins_,etaMin_,etaMax_);
    hbHists.noADC_ID_eta = m_dbe->book1D("HB_noADCIDEta","HB No ADC ID Eta ",etaBins_,etaMin_,etaMax_);
    hbHists.ADCdist = m_dbe->book1D("HB_ADCdist","HB ADC count distribution",128,0,128);
    hbHists.NADA_cool_cell_map = m_dbe->book2D("HB_NADA_CoolCellMap","HB Cool Cells",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.digiCheck = m_dbe->book2D("HB_digiCheck","HB Check that digi was found",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.cellCheck = m_dbe->book2D("HB_cellCheck","HB Check that cell hit was found",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.deadcapADC_map.push_back(m_dbe->book2D("HB_DeadCap0","Map of HB Events with no ADC hits for capid=0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hbHists.deadcapADC_map.push_back(m_dbe->book2D("HB_DeadCap1","Map of HB Events with no ADC hits for capid=1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hbHists.deadcapADC_map.push_back(m_dbe->book2D("HB_DeadCap2","Map of HB Events with no ADC hits for capid=2",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hbHists.deadcapADC_map.push_back(m_dbe->book2D("HB_DeadCap3","Map of HB Events with no ADC hits for capid=3",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hbHists.above_pedestal = m_dbe->book2D("HB_abovePed","HB cells above pedestal+Nsigma",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.coolcell_below_pedestal = m_dbe->book2D("HB_CoolCell_belowPed","HB cells below (pedestal+Nsigma)",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.above_pedestal_temp = new TH2F("HB_abovePedTemp","Don't look at this!",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);


    // HE
    m_dbe->setCurrentFolder(baseFolder_+"/HE");
    heHists.type=2;
    heHists.deadADC_map = m_dbe->book2D("HE_deadADCOccupancyMap","HE No ADC Count Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.noADC_ID_map = m_dbe->book2D("HE_noADCIDOccupancyMap","HE No ADC ID Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.deadADC_eta = m_dbe->book1D("HE_deadADCEta","HE No ADC Count Eta ",etaBins_,etaMin_,etaMax_);
    heHists.noADC_ID_eta = m_dbe->book1D("HE_noADCIDEta","HE No ADC ID Eta ",etaBins_,etaMin_,etaMax_);
    heHists.ADCdist = m_dbe->book1D("HE_ADCdist","HE ADC count distribution",128,0,128);
    heHists.NADA_cool_cell_map = m_dbe->book2D("HE_NADA_CoolCellMap","HE Cool Cells",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.digiCheck = m_dbe->book2D("HE_digiCheck","HE Check that digi was found",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.cellCheck = m_dbe->book2D("HE_cellCheck","HE Check that cell hit was found",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.deadcapADC_map.push_back(m_dbe->book2D("HE_DeadCap0","Map of HE Events with no ADC hits for capid=0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    heHists.deadcapADC_map.push_back(m_dbe->book2D("HE_DeadCap1","Map of HE Events with no ADC hits for capid=1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    heHists.deadcapADC_map.push_back(m_dbe->book2D("HE_DeadCap2","Map of HE Events with no ADC hits for capid=2",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    heHists.deadcapADC_map.push_back(m_dbe->book2D("HE_DeadCap3","Map of HE Events with no ADC hits for capid=3",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    heHists.above_pedestal = m_dbe->book2D("HE_abovePed","HE cells above pedestal+Nsigma",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.coolcell_below_pedestal = m_dbe->book2D("HE_CoolCell_belowPed","HE cells below (pedestal+Nsigma)",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.above_pedestal_temp = new TH2F("HE_abovePedTemp","Don't look at this!",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    
    // HO
    m_dbe->setCurrentFolder(baseFolder_+"/HO");
    hoHists.type=3;
    hoHists.deadADC_map = m_dbe->book2D("HO_deadADCOccupancyMap","HO No ADC Count Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.noADC_ID_map = m_dbe->book2D("HO_noADCIDOccupancyMap","HO No ADC ID Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.deadADC_eta = m_dbe->book1D("HO_deadADCEta","HO No ADC Count Eta ",etaBins_,etaMin_,etaMax_);
    hoHists.noADC_ID_eta = m_dbe->book1D("HO_noADCIDEta","HO No ADC ID Eta ",etaBins_,etaMin_,etaMax_);
    hoHists.ADCdist = m_dbe->book1D("HO_ADCdist","HO ADC count distribution",128,0,128);
    hoHists.NADA_cool_cell_map = m_dbe->book2D("HO_NADA_CoolCellMap","HO Cool Cells",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.digiCheck = m_dbe->book2D("HO_digiCheck","HO Check that digi was found",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.cellCheck = m_dbe->book2D("HO_cellCheck","HO Check that cell hit was found",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.deadcapADC_map.push_back(m_dbe->book2D("HO_DeadCap0","Map of HO Events with no ADC hits for capid=0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hoHists.deadcapADC_map.push_back(m_dbe->book2D("HO_DeadCap1","Map of HO Events with no ADC hits for capid=1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hoHists.deadcapADC_map.push_back(m_dbe->book2D("HO_DeadCap2","Map of HO Events with no ADC hits for capid=2",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hoHists.deadcapADC_map.push_back(m_dbe->book2D("HO_DeadCap3","Map of HO Events with no ADC hits for capid=3",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hoHists.above_pedestal = m_dbe->book2D("HO_abovePed","HO cells above pedestal+Nsigma",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.coolcell_below_pedestal = m_dbe->book2D("HO_CoolCell_belowPed","HO cells below (pedestal+Nsigma)",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.above_pedestal_temp = new TH2F("HO_abovePedTemp","Don't look at this!",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);


    // HF
    m_dbe->setCurrentFolder(baseFolder_+"/HF");
    hfHists.type=10; //1+2+3+4
    hfHists.deadADC_map = m_dbe->book2D("HF_deadADCOccupancyMap","HF No ADC Count Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.noADC_ID_map = m_dbe->book2D("HF_noADCIDOccupancyMap","HF No ADC ID Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.deadADC_eta = m_dbe->book1D("HF_deadADCEta","HF No ADC Count Eta ",etaBins_,etaMin_,etaMax_);
    hfHists.noADC_ID_eta = m_dbe->book1D("HF_noADCIDEta","HF No ADC ID Eta ",etaBins_,etaMin_,etaMax_);
    hfHists.ADCdist = m_dbe->book1D("HF_ADCdist","HF ADC count distribution",128,0,128);
    hfHists.NADA_cool_cell_map = m_dbe->book2D("HF_NADA_CoolCellMap","HF Cool Cells",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.digiCheck = m_dbe->book2D("HF_digiCheck","HF Check that digi was found",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.cellCheck = m_dbe->book2D("HF_cellCheck","HF Check that cell hit was found",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.deadcapADC_map.push_back(m_dbe->book2D("HF_DeadCap0","Map of HF Events with no ADC hits for capid=0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hfHists.deadcapADC_map.push_back(m_dbe->book2D("HF_DeadCap1","Map of HF Events with no ADC hits for capid=1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hfHists.deadcapADC_map.push_back(m_dbe->book2D("HF_DeadCap2","Map of HF Events with no ADC hits for capid=2",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hfHists.deadcapADC_map.push_back(m_dbe->book2D("HF_DeadCap3","Map of HF Events with no ADC hits for capid=3",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_)); 
    hfHists.above_pedestal = m_dbe->book2D("HF_abovePed","HF cells above pedestal+Nsigma",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.coolcell_below_pedestal = m_dbe->book2D("HF_CoolCell_belowPed","HF cells below (pedestal+Nsigma)",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.above_pedestal_temp = new TH2F("HF_abovePedTemp","Don't look at this!",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

 // HF
    m_dbe->setCurrentFolder(baseFolder_+"/HCAL");
    hcalHists.type=4;
    hcalHists.deadADC_map = m_dbe->book2D("HCAL_deadADCOccupancyMap","HCAL No ADC Count Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hcalHists.noADC_ID_map = m_dbe->book2D("HCAL_noADCIDOccupancyMap","HCAL No ADC ID Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hcalHists.deadADC_eta = m_dbe->book1D("HCAL_deadADCEta","HCAL No ADC Count Eta ",etaBins_,etaMin_,etaMax_);
    hcalHists.noADC_ID_eta = m_dbe->book1D("HCAL_noADCIDEta","HCAL No ADC ID Eta ",etaBins_,etaMin_,etaMax_);
    hcalHists.ADCdist = m_dbe->book1D("HCAL_ADCdist","HCAL ADC count distribution",128,0,128);
    hcalHists.NADA_cool_cell_map = m_dbe->book2D("HCAL_NADA_CoolCellMap","HCAL Cool Cells",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hcalHists.digiCheck = m_dbe->book2D("HCAL_digiCheck","HCAL Check that digi was found",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hcalHists.cellCheck = m_dbe->book2D("HCAL_cellCheck","HCAL Check that cell hit was found",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hcalHists.deadcapADC_map.push_back(m_dbe->book2D("HCAL_DeadCap0","Map of HCAL Events with no ADC hits for capid=0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hcalHists.deadcapADC_map.push_back(m_dbe->book2D("HCAL_DeadCap1","Map of HCAL Events with no ADC hits for capid=1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hcalHists.deadcapADC_map.push_back(m_dbe->book2D("HCAL_DeadCap2","Map of HCAL Events with no ADC hits for capid=2",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    hcalHists.deadcapADC_map.push_back(m_dbe->book2D("HCAL_DeadCap3","Map of HCAL Events with no ADC hits for capid=3",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_)); 
    hcalHists.above_pedestal = m_dbe->book2D("HCAL_abovePed","HCAL cells above pedestal+Nsigma",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hcalHists.coolcell_below_pedestal = m_dbe->book2D("HCAL_CoolCell_belowPed","HCAL cells below (pedestal+Nsigma)",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hcalHists.above_pedestal_temp = new TH2F("HCAL_abovePedTemp","Don't look at this!",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  } // if (m_dbe!=NULL)

  return;
}// void HcalDeadCellMonitor::setup


void HcalDeadCellMonitor::processEvent(const HBHERecHitCollection& hbHits, 
				       const HORecHitCollection& hoHits, 
				       const HFRecHitCollection& hfHits,
				       const HBHEDigiCollection& hbhedigi,
				       const HODigiCollection& hodigi,
				       const HFDigiCollection& hfdigi,
				       const HcalDbService& cond)
{
  if(!m_dbe) 
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent    DaqMonitorBEInterface not instantiated!!!\n";
      return;
    }
  ievt_++;
  meEVT_->Fill(ievt_);
  if (fVerbosity) cout <<"HcalDeadCellMonitor::processEvent     Starting process"<<endl;
  
  processEvent_digi(hbhedigi,hodigi,hfdigi,cond); // check for dead digis
  processEvent_hits(hbHits,hoHits,hfHits); // check for dead cell hits



  // Look for cells that have been "cool" for (checkNevents_) consecutive events
  if ((ievt_%checkNevents_)==0)
    {
      reset_Nevents(hbHists);
      reset_Nevents(heHists);
      reset_Nevents(hoHists);
      reset_Nevents(hfHists);
    }
  
} // void HcalDeadCellMonitor::processEvent


void HcalDeadCellMonitor::processEvent_digi(const HBHEDigiCollection& hbhedigi,
					    const HODigiCollection& hodigi,
					    const HFDigiCollection& hfdigi,
					    const HcalDbService& cond)
{

  if (fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_digi     Starting process"<<endl;

  HcalCalibrationWidths widths;
 
  try
    {
      for (HBHEDigiCollection::const_iterator j=hbhedigi.begin(); j!=hbhedigi.end(); j++)
	{
	  const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
	  cond.makeHcalCalibration(digi.id(), &calibs_);
	  cond.makeHcalCalibrationWidth(digi.id(),&widths);

	  if ((HcalSubdetector)(digi.id().subdet())==HcalBarrel)
	    HcalDeadCellDigiCheck::CheckForDeadDigis(digi,hbHists,hcalHists,
						     Nsigma_,minADCcount_,calibs_,widths,m_dbe,baseFolder_);
	  else if ((HcalSubdetector)(digi.id().subdet())==HcalEndcap)
	    HcalDeadCellDigiCheck::CheckForDeadDigis(digi,heHists,hcalHists,
						     Nsigma_,minADCcount_,calibs_,widths,m_dbe,baseFolder_);
	}
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_digi   No HBHE Digis."<<endl;
    }

  try
    {
      for (HODigiCollection::const_iterator j=hodigi.begin(); j!=hodigi.end(); j++)
	{
	  const HODataFrame digi = (const HODataFrame)(*j);
	  cond.makeHcalCalibration(digi.id(), &calibs_);
	  cond.makeHcalCalibrationWidth(digi.id(),&widths);
	  HcalDeadCellDigiCheck::CheckForDeadDigis(digi,hoHists,hcalHists,
						   Nsigma_,minADCcount_,calibs_,widths,m_dbe,baseFolder_);
	}
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_digi   No HO Digis."<<endl;
    }

  try
    {
      for (HODigiCollection::const_iterator j=hodigi.begin(); j!=hodigi.end(); j++)
	{
	  const HODataFrame digi = (const HODataFrame)(*j);
	  cond.makeHcalCalibration(digi.id(), &calibs_);
	  cond.makeHcalCalibrationWidth(digi.id(),&widths);
	  HcalDeadCellDigiCheck::CheckForDeadDigis(digi,hfHists,hcalHists,
						   Nsigma_,minADCcount_,calibs_,widths,m_dbe,baseFolder_);
	}
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_digi   No HF Digis."<<endl;
    }



  return;

} // void HcalDeadCellMonitor::processEvent_digi

void HcalDeadCellMonitor::processEvent_hits(const HBHERecHitCollection& hbHits, 
					    const HORecHitCollection& hoHits, 
					    const HFRecHitCollection& hfHits)
{
  if(!m_dbe) 
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits    DaqMonitorBEInterface not instantiated!!!\n";
      return;
    }
  if (fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits     Starting process"<<endl;
  try
    {
      HcalDeadCellDigiCheck::CheckHits(coolcellfrac_,hbHits,hbHists,hcalHists,
				       m_dbe, baseFolder_);
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HB Hits"<<endl;
    }
  try
    {
      HcalDeadCellDigiCheck::CheckHits(coolcellfrac_,hbHits,heHists,hcalHists,
				       m_dbe, baseFolder_);
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HE Hits"<<endl;
    }
  try
    {
      HcalDeadCellDigiCheck::CheckHits(coolcellfrac_,hoHits,hoHists,hcalHists,
				       m_dbe, baseFolder_);
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HO Hits"<<endl;
    }
  try
    {
      HcalDeadCellDigiCheck::CheckHits(coolcellfrac_,hfHits,hfHists,hcalHists,
				       m_dbe, baseFolder_);
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HF Hits"<<endl;
    }

  return;

} // void HcalDeadCellMonitor::processEvent_hits

void HcalDeadCellMonitor::reset_Nevents(DeadCellHists &h)

{


  for (int eta=0;eta<etaBins_;eta++)
    {
      // FIXME -- Work on better implementation of boundary conditions
      // (Something like GetBinContent for MonitorElements?)
      if ((h.type==0 ||h.type==2) && fabs(eta+etaMin_-1)>16) 
	continue;
      else if (h.type==1 && (fabs(eta+etaMin_-1)<15||fabs(eta+etaMin_-1)>30)) 
	continue;
      else if (h.type==3 && fabs(eta+etaMin_-1)<28) 
	continue;
      for (int phi=0;phi<phiBins_;phi++)
	{
	  double temp=h.above_pedestal_temp->GetBinContent(eta,phi);
	  if (temp==0)
	    {
	      h.coolcell_below_pedestal->Fill(eta+etaMin_-1,phi+phiMin_-1);
	      hcalHists.coolcell_below_pedestal->Fill(eta+etaMin_-1,phi+phiMin_-1);
	    }
	  else
	    {
	      h.above_pedestal->Fill(eta+etaMin_-1,phi+phiMin_-1,temp);
	      hcalHists.above_pedestal->Fill(eta+etaMin_-1,phi+phiMin_-1,temp);
	    }
	}
    }
  h.above_pedestal_temp->Reset();
  hcalHists.above_pedestal_temp->Reset();

  return;
}
