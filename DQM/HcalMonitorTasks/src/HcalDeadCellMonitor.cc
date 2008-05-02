#include "DQM/HcalMonitorTasks/interface/HcalDeadCellMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <math.h>
#include <sstream>
using namespace std;

HcalDeadCellMonitor::HcalDeadCellMonitor(){
  ievt_=0;
}

HcalDeadCellMonitor::~HcalDeadCellMonitor() {
}

namespace HcalDeadCellCheck
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
			 DQMStore* dbe, string baseFolder)
  {
    if (hist.check==0) return;
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
    for (int i=0;i<digi.size();++i)
      {
	ADCsum+=digi.sample(i).adc();
	//if (ADCsum!=0) break;
	int thisCapid = digi.sample(i).capid();

	/* If ADC value above (pedestal+Nsgima_), fill hist.above_pedestal_temp
	   (Cool cells will later be found by looking for empty spots in the
	   above_pedestal_temp histogram)
	*/
	if (digi.sample(i).adc()>calibs.pedestal(thisCapid)+Nsigma*widths.pedestal(thisCapid))
	  {
	    hist.above_pedestal_temp->Fill(digi.id().ieta()+offset,digi.id().iphi());
	    all.above_pedestal_temp->Fill(digi.id().ieta()+offset,digi.id().iphi());
	  }
	capADC[thisCapid]+=digi.sample(i).adc();


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

    // look for individual dead caps
    for (int zz=0;zz<4;++zz)
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
		 DQMStore* dbe, string baseFolder)
  { 
    if (hist.check==false) return;
    string type;
    type=hist.subdet;

    if(dbe) dbe->setCurrentFolder(baseFolder+"/"+type);

    
    typename Hits::const_iterator _cell;
    for (_cell=hits.begin();
	 _cell!=hits.end(); 
	 ++_cell)
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

	int allneighbors=0;
	int etaFactor;  // correct for eta regions where phi segmentation is > 5 degrees/cell

	for (typename Hits::const_iterator neighbor=hits.begin();neighbor!=hits.end();++neighbor)
	  {
	    //if (vetoCell(neighbor->id())) continue;
	    if  ((HcalSubdetector)(neighbor->id().subdet())!=(HcalSubdetector)(_cell->id().subdet())) continue;
	    if (neighbor->id().depth()!=_cell->id().depth()) continue;
	    
	    int NeighborEta=neighbor->id().ieta();
	    etaFactor = 1+(abs(NeighborEta)>20)+2*(abs(NeighborEta)>39);
	    
	    if ( (abs(neighbor->id().iphi()-_cell->id().iphi()))<2*etaFactor && 
		 (abs(neighbor->id().ieta()-_cell->id().ieta()))<2)
	      {

		// Skip neighbors with negative energy?
		//if (neighbor->energy()<0) continue;
		allneighbors++;
		if (neighbor->energy()-_cell->energy()>hist.mindiff)
		  {
		    neighborE+=neighbor->energy();
		    neighbors++;
		  }
	      }
	  }// for (Hits::const_iterator neighbor=hits.begin()...
	
	// Remove cell energy from neighbor calculation
	neighborE-=_cell->energy();
	neighbors-=1;

	// Skip?
	if (_cell->energy()<hist.floor)
	  {
	    hist.NADA_cool_cell_map->Fill(_cell->id().ieta(),_cell->id().iphi());
	    all.NADA_cool_cell_map->Fill(_cell->id().ieta(),_cell->id().iphi());
	  }

	// Require at least half of adjoining cells exceed minimum difference 
	//if ((neighbors>2) &&
	//   (_cell->energy()>0. && _cell->energy()<coolcellfrac*(1.0*neighborE/neighbors)))

	if (allneighbors==0 || neighbors < 2) continue;
	if (1.*neighbors/allneighbors<coolcellfrac) continue;
	{
	  //if ((1.0*neighborE/neighbors*coolcellfrac-_cell->energy())<hist.mindiff) continue;
	  //cout <<"COOL CELL, NEIGHBOR ENERGY = "<<_cell->energy()<<"  "<<1.0*neighborE/neighbors<<endl;
	  hist.NADA_cool_cell_map->Fill(_cell->id().ieta(),_cell->id().iphi());
	  all.NADA_cool_cell_map->Fill(_cell->id().ieta(),_cell->id().iphi());
	}
      } // for (_cell=hits.begin()...)
    return;

  } // void CheckHits

} // namespace HcalDeadCellCheck



void HcalDeadCellMonitor::reset(){}


void HcalDeadCellMonitor::setup(const edm::ParameterSet& ps,
				DQMStore* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  
  baseFolder_ = rootFolder_+"DeadCellMonitor";

  // Get ps parameters here

  // Set input parameters from .cfi file
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  if (fVerbosity) 
    cout << "DeadCell eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  
  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  if (fVerbosity) 
    cout << "DeadCell phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;

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
  floor_ = ps.getUntrackedParameter<double>("deadcellfloor",-0.5);
  mindiff_ = ps.getUntrackedParameter<double>("deadcellmindiff",0.5);


  ievt_=0;
  if (m_dbe !=NULL) {
    m_dbe->setCurrentFolder(baseFolder_);
    
    meEVT_ = m_dbe->bookInt("DeadCell Task Event Number");    
    meEVT_->Fill(ievt_);


    // Set up subdetector histograms
    hbHists.check=ps.getUntrackedParameter<bool>("checkHB", 1);
    heHists.check=ps.getUntrackedParameter<bool>("checkHE", 1);
    hoHists.check=ps.getUntrackedParameter<bool>("checkHO", 1);
    hfHists.check=ps.getUntrackedParameter<bool>("checkHF", 1);
    hcalHists.check=(hbHists.check || heHists.check || hoHists.check || hfHists.check);

    hbHists.type=1;
    setupHists(hbHists,m_dbe);
    heHists.type=2;
    setupHists(heHists,m_dbe);
    hoHists.type=3;
    setupHists(hoHists,m_dbe);
    hfHists.type=4;
    setupHists(hfHists,m_dbe);
    hcalHists.type=10;
    setupHists(hcalHists,m_dbe);
    
  } // if (m_dbe!=NULL)

  return;
}// void HcalDeadCellMonitor::setup


void HcalDeadCellMonitor::setupHists(DeadCellHists& hist,  DQMStore* dbe)
{
  if (hist.check==0) return;
  if (hist.type==1)
    hist.subdet="HB";
  else if (hist.type==2)
    hist.subdet="HE";
  else if (hist.type==3)
    hist.subdet="HO";
  else if (hist.type==4)
    hist.subdet="HF";
  else if (hist.type==10)
    hist.subdet="HCAL";
  else
    {
      if (fVerbosity) cout <<"<HcalDeadCellMonitor::setupHists> Unrecognized subdetector type "<<hist.type<<endl;
      return;
    }

  hist.floor=floor_;
  hist.mindiff=mindiff_;

  m_dbe->setCurrentFolder(baseFolder_+"/"+hist.subdet.c_str());
  hist.deadADC_map = m_dbe->book2D(hist.subdet+"_deadADCOccupancyMap",hist.subdet+" No ADC Count Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  //hist.noADC_ID_map = m_dbe->book2D(hist.subdet+"_noADCIDOccupancyMap",hist.subdet+" No ADC ID Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  hist.deadADC_eta = m_dbe->book1D(hist.subdet+"_deadADCEta",hist.subdet+" No ADC Count Eta ",etaBins_,etaMin_,etaMax_);
  //hist.noADC_ID_eta = m_dbe->book1D(hist.subdet+"_noADCIDEta",hist.subdet+" No ADC ID Eta ",etaBins_,etaMin_,etaMax_);
  hist.ADCdist = m_dbe->book1D(hist.subdet+"_ADCdist",hist.subdet+" ADC count distribution",128,0,128);
  hist.NADA_cool_cell_map = m_dbe->book2D(hist.subdet+"_NADA_CoolCellMap",hist.subdet+" Cool Cells",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  hist.digiCheck = m_dbe->book2D(hist.subdet+"_digiCheck",hist.subdet+" Check that digi was found",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  hist.cellCheck = m_dbe->book2D(hist.subdet+"_cellCheck",hist.subdet+" Check that cell hit was found",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  hist.deadcapADC_map.push_back(m_dbe->book2D(hist.subdet+"_DeadCap0","Map of "+hist.subdet+" Events with no ADC hits for capid=0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
  hist.deadcapADC_map.push_back(m_dbe->book2D(hist.subdet+"_DeadCap1","Map of "+hist.subdet+" Events with no ADC hits for capid=1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
  hist.deadcapADC_map.push_back(m_dbe->book2D(hist.subdet+"_DeadCap2","Map of "+hist.subdet+" Events with no ADC hits for capid=2",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
  hist.deadcapADC_map.push_back(m_dbe->book2D(hist.subdet+"_DeadCap3","Map of "+hist.subdet+" Events with no ADC hits for capid=3",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
  string Nsig;
  stringstream out;
  out <<Nsigma_;
  Nsig=out.str();
  string consec;
  stringstream out2;
  out2<<checkNevents_;
  consec=out2.str();
  hist.above_pedestal = m_dbe->book2D(hist.subdet+"_abovePed",hist.subdet+" cells above pedestal+"+Nsig+"sigma",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  hist.coolcell_below_pedestal = m_dbe->book2D(hist.subdet+"_CoolCell_belowPed",hist.subdet+" cells below (pedestal+"+Nsig+"sigma) for "+consec+" consecutive events",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  char PedTemp[256];
  sprintf(PedTemp,"%sAbovePedTemp",hist.subdet.c_str());
  hist.above_pedestal_temp = new TH2F(PedTemp,"Don't look at this!",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

  // Set Axis Labels
  hist.deadADC_map->setAxisTitle("i#eta", 1);
  hist.deadADC_map->setAxisTitle("i#phi",2);
  hist.deadADC_eta->setAxisTitle("i#eta", 1);
  hist.deadADC_eta->setAxisTitle("ADC count< minimum",2);
  hist.ADCdist->setAxisTitle("ADC",1);
  hist.ADCdist->setAxisTitle("# of counts",2);
  hist.NADA_cool_cell_map->setAxisTitle("i#eta", 1);
  hist.NADA_cool_cell_map->setAxisTitle("i#phi",2);
  hist.digiCheck->setAxisTitle("i#eta", 1);
  hist.digiCheck->setAxisTitle("i#phi",2);
  hist.cellCheck->setAxisTitle("i#eta", 1);
  hist.cellCheck->setAxisTitle("i#phi",2);
  for (unsigned int icap=0;icap<hist.deadcapADC_map.size();++icap)
    {
      hist.deadcapADC_map[icap]->setAxisTitle("i#eta", 1);
      hist.deadcapADC_map[icap]->setAxisTitle("i#phi",2);
    }
  hist.above_pedestal->setAxisTitle("i#eta", 1);
  hist.above_pedestal->setAxisTitle("i#phi",2);
  hist.coolcell_below_pedestal->setAxisTitle("i#eta", 1);
  hist.coolcell_below_pedestal->setAxisTitle("i#phi",2);

  return;
}

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
      if(fVerbosity) cout <<"<HcalDeadCellMonitor::processEvent>    DQMStore not instantiated!!!\n";
      return;
    }
  ievt_++;
  meEVT_->Fill(ievt_);
  if (fVerbosity) cout <<"<HcalDeadCellMonitor::processEvent>     Starting process"<<endl;
  
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
      for (HBHEDigiCollection::const_iterator j=hbhedigi.begin(); j!=hbhedigi.end(); ++j)
	{
	  const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
	  calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private. 
	  cond.makeHcalCalibrationWidth(digi.id(),&widths);

	  if ((HcalSubdetector)(digi.id().subdet())==HcalBarrel)
	    HcalDeadCellCheck::CheckForDeadDigis(digi,hbHists,hcalHists,
						 Nsigma_,minADCcount_,calibs_,widths,m_dbe,baseFolder_);
	  else if ((HcalSubdetector)(digi.id().subdet())==HcalEndcap)
	    HcalDeadCellCheck::CheckForDeadDigis(digi,heHists,hcalHists,
						 Nsigma_,minADCcount_,calibs_,widths,m_dbe,baseFolder_);
	}
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_digi   No HBHE Digis."<<endl;
    }

  try
    {
      for (HODigiCollection::const_iterator j=hodigi.begin(); j!=hodigi.end(); ++j)
	{
	  const HODataFrame digi = (const HODataFrame)(*j);
	  calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private. 
	  cond.makeHcalCalibrationWidth(digi.id(),&widths);
	  HcalDeadCellCheck::CheckForDeadDigis(digi,hoHists,hcalHists,
					       Nsigma_,minADCcount_,calibs_,widths,m_dbe,baseFolder_);
	}
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_digi   No HO Digis."<<endl;
    }

  try
    {
      for (HFDigiCollection::const_iterator j=hfdigi.begin(); j!=hfdigi.end(); ++j)
	{
	  const HFDataFrame digi = (const HFDataFrame)(*j);
	  calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private. 
	  cond.makeHcalCalibrationWidth(digi.id(),&widths);
	  HcalDeadCellCheck::CheckForDeadDigis(digi,hfHists,hcalHists,
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
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits    DQMStore not instantiated!!!\n";
      return;
    }
  if (fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits     Starting process"<<endl;
  try
    {
      HcalDeadCellCheck::CheckHits(coolcellfrac_,hbHits,hbHists,hcalHists,
				   m_dbe, baseFolder_);
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HB Hits"<<endl;
    }
  try
    {
      HcalDeadCellCheck::CheckHits(coolcellfrac_,hbHits,heHists,hcalHists,
				   m_dbe, baseFolder_);
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HE Hits"<<endl;
    }
  try
    {
      HcalDeadCellCheck::CheckHits(coolcellfrac_,hoHits,hoHists,hcalHists,
				   m_dbe, baseFolder_);
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HO Hits"<<endl;
    }
  try
    {
      HcalDeadCellCheck::CheckHits(coolcellfrac_,hfHits,hfHists,hcalHists,
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
  if (h.check==0) return;

  for (float eta=etaMin_;eta<etaMax_;eta+=1.)
    {
      if (eta==0.) continue; // skip eta=0;
      if (h.type==1 && fabs(eta)>16) continue;
      else if (h.type==2 && (fabs(eta)<16 || fabs(eta)>29))
	continue;
      else if (h.type==3 && fabs(eta)>4) continue;
      else if (h.type==4 && fabs(eta)<30) continue; // FIXME:  is this the correct condition for HF?

      // phi indices start at 1, end at 72
      for (float phi=phiMin_;phi<phiMax_;phi+=1.)
	{
	  if (h.type==2 && (fabs(eta)>20) && (int(phi)%2)==0) continue; // skip HE even-phi counters where they don't exist
	  if (phi==0.) continue;  // skip phi=0
	  double temp=h.above_pedestal_temp->GetBinContent(int(eta-etaMin_+1),int(phi-phiMin_+1));

	  if (temp==0)
	    {
	      h.coolcell_below_pedestal->Fill(int(eta),int(phi));
	      hcalHists.coolcell_below_pedestal->Fill(int(eta),int(phi));
	    }
	  else
	    {
	      h.above_pedestal->Fill(int(eta),int(phi),temp);
	      hcalHists.above_pedestal->Fill(int(eta),int(phi),temp);
	    }
	}
    }
  h.above_pedestal_temp->Reset();
  hcalHists.above_pedestal_temp->Reset();

  return;
}
