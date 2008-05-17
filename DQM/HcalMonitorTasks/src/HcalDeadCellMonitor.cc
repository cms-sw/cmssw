#include "DQM/HcalMonitorTasks/interface/HcalDeadCellMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <math.h>
#include <sstream>
using namespace std;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* Constructor and Destructor */

HcalDeadCellMonitor::HcalDeadCellMonitor()
{
  ievt_=0;
}

HcalDeadCellMonitor::~HcalDeadCellMonitor() 
{
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/*
  Create separate namespace for creating template functions
  (Template 'digi' functions will run on any type of digi collection -- HBHE, HO, or HF.
   Template 'rechit' functions will run on any type of rechit collection.)
*/
   

namespace HcalDeadCellCheck
{

  /*
    CheckForDeadDigis looks for digis with ADCs = 0 
    or digis with ADCs < pedestal+Nsigma_
  */ 

  template<class Digi>
  void CheckForDeadDigis(const Digi& digi, DeadCellHists& hist, 
			 DeadCellHists& all,
			 float Nsigma, float mincount, // specify Nsigma for pedestal check, mincount for ADC check
			 HcalCalibrations calibs, 
			 HcalCalibrationWidths widths, 
			 DQMStore* dbe, string baseFolder)
  {

    if (hist.check==0) return;

    string type;
    // Get subdet name associated with type value
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

    int digi_eta=digi.id().ieta();
    int digi_phi=digi.id().iphi();
    int digi_depth=digi.id().depth();

    // Fill (eta,phi) map if digi is found for that cell
    hist.digiCheck->Fill(digi_eta,digi_phi);
    all.digiCheck->Fill(digi_eta,digi_phi);
    hist.digiCheck_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
    all.digiCheck_depth[digi_depth-1]->Fill(digi_eta,digi_phi);



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
	    hist.above_pedestal_temp->Fill(digi_eta,digi_phi);
	    all.above_pedestal_temp->Fill(digi_eta,digi_phi);
	  }
	capADC[thisCapid]+=digi.sample(i).adc();


	// Not yet sure if this histogram is useful, but it gives an idea of the ADC distributions
	hist.ADCdist->Fill(digi.sample(i).adc());
	all.ADCdist->Fill(digi.sample(i).adc());
      }

    // If ADCsum <= mincount, cell is considered dead
    if (ADCsum<=mincount)
      {
	hist.deadADC_map->Fill(digi_eta,digi_phi);
	hist.deadADC_map_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
	hist.deadADC_eta->Fill(digi_eta);
	all.deadADC_map->Fill(digi_eta,digi_phi);
	all.deadADC_map_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
	all.deadADC_eta->Fill(digi_eta);
      }

    // look for individual dead caps
    for (int zz=0;zz<4;++zz)
      {
	if (capADC[zz]<=mincount)
	  {
	    hist.deadcapADC_map[zz]->Fill(digi_eta,digi_phi);
	    all.deadcapADC_map[zz]->Fill(digi_eta,digi_phi);
	  }
      }
    return;
  } // void CheckForDeadDigis(...)




  /*CheckHits searches HCal hits for cells with energies much less than their
    neighbors'
  */

  template<class Hits>
  void CheckHits(double coolcellfrac, const Hits& hits, 
		 DeadCellHists& hist, DeadCellHists& all, 
		 DQMStore* dbe, string baseFolder)
  { 
    if (hist.check==false) 
      return;
    
    string type;
    type=hist.subdet;

    if(dbe) dbe->setCurrentFolder(baseFolder+"/"+type);

    typename Hits::const_iterator _cell;
    for (_cell=hits.begin();
	 _cell!=hits.end(); 
	 ++_cell)
      {
	// Fill histogram if cell found in hist region
	if ((_cell->id().subdet())!=hist.type) continue; // does this cause a slowdown?  Or has Jason's fix cured this?

	int cell_eta=_cell->id().ieta();
	int cell_phi=_cell->id().iphi();
	int cell_depth=_cell->id().depth();
	int temp_cell_phi = cell_phi;  // temporary variable for dealing with neighbors at boundaries between different phi segmentations
	
	hist.cellCheck->Fill(cell_eta,cell_phi);
	all.cellCheck->Fill(cell_eta,cell_phi);
	hist.cellCheck_depth[cell_depth-1]->Fill(cell_eta,cell_phi);
	all.cellCheck_depth[ cell_depth-1]->Fill(cell_eta,cell_phi);

	// if (_cell->id().depth()==2) continue; // skip depth=2 for now
	// if (vetoCell(_cell->id())) continue;

	// Sum energies of neighbors around cell
	
	int allneighbors=0; // all neighbors with energy>0 around cell
	int neighbors=0; // subset of allneighbors, with energy > (cell energy + mindiff)
	double neighborE=0; // total energy found from neighbors

	int etaFactor;  // correct for eta regions where phi segmentation is > 5 degrees/cell

	for (typename Hits::const_iterator neighbor=hits.begin();neighbor!=hits.end();++neighbor)
	  {
	    //if (vetoCell(neighbor->id())) continue;
	    if  ((HcalSubdetector)(neighbor->id().subdet())!=(HcalSubdetector)(_cell->id().subdet())) continue;
	    if (neighbor->id().depth()!=_cell->id().depth()) continue;
	    
	    int NeighborEta=neighbor->id().ieta();

	    etaFactor = 1+(abs(NeighborEta)>20)+2*(abs(NeighborEta)>39);  // = 1 for phi=5 segments, 2 for phi=10, 4 for phi=20
	    
	    // boundary between phi=5 and phi=10 segmentation
	    if (abs(cell_eta)==20 && abs(NeighborEta)==21)
	      {
	      temp_cell_phi-=(temp_cell_phi%2); // odd cells treated as even:
	      }

	    /*
	      5 5 5
              4 4          if cellphi at eta=20 = 2, want to check larger cells 1 & 3
              3 3 3        likewise, if cellphi=3, still want to check 1 & 3
              2 2          therefore, shift back cellphi=3 index by one, check larger cells within +/-1 of this new value
              1 1 1        (even cells can remain checking +/-1 of their value)

	       Do we also need to deal with special case where cell_eta=21, NeighborEta=20?
	    */

	    // boundary between phi=10 and phi=20 segmentation
	    else if (abs(cell_eta)==39 && abs(NeighborEta)==40)
	      {
		temp_cell_phi-=(temp_cell_phi%4==1);
	      }

	    // Check just against nearest neighbors (+/-1 in eta, phi, same depth)

	    if ( (abs(neighbor->id().iphi()-temp_cell_phi))<(1+etaFactor) && 
		 (abs(neighbor->id().ieta()-cell_eta))<2)
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

	// If cell energy is less than minimum value ("hist.floor") , mark it as cool for the event
	if (_cell->energy()<hist.floor)
	  {
	    hist.NADA_cool_cell_map->Fill(cell_eta,cell_phi);
	    all.NADA_cool_cell_map->Fill(cell_eta,cell_phi);
	    hist.NADA_cool_cell_map_depth[cell_depth-1]->Fill(cell_eta,cell_phi);
	    all.NADA_cool_cell_map_depth[ cell_depth-1]->Fill(cell_eta,cell_phi);

	  }
	
	else
	  {
	    // Require at least half of neighboring cells exceed minimum difference,
	    // and that cell energy < coolcellfrac * (average of neighbors)

	    // hard-code # of neighbors -- may need to be adjusted at phi segmenation boundaries
	    if ((neighbors>4) &&
	       (_cell->energy()>0. && _cell->energy()<coolcellfrac*(1.0*neighborE/neighbors)))

	    {
	      hist.NADA_cool_cell_map->Fill(cell_eta, cell_phi);
	      all.NADA_cool_cell_map->Fill( cell_eta, cell_phi);
	      hist.NADA_cool_cell_map_depth[cell_depth-1]->Fill(cell_eta,cell_phi);
	      all.NADA_cool_cell_map_depth[ cell_depth-1]->Fill(cell_eta,cell_phi);
	    } 
 
	  } // else (_cell->energy()>=hist.floor)

      } // for (_cell=hits.begin()...)

    return;

  } // void CheckHits

} // namespace HcalDeadCellCheck


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void HcalDeadCellMonitor::reset(){}  // reset function is empty for now


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void HcalDeadCellMonitor::setup(const edm::ParameterSet& ps,
				DQMStore* dbe)
{
  /* 
     Set up the DeadCellMonitor, using user parameters (ps) from .cfg files.
  */

  HcalBaseMonitor::setup(ps,dbe); // base class setup
  
  baseFolder_ = rootFolder_+"DeadCellMonitor"; // Make subfolder
  
  // Get ps parameters here

  // Set input parameters from .cfi file
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 41.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -41.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  if (fVerbosity) 
    cout << "DeadCell eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  
  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73.5);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", -0.5);
  phiBins_ = (int)(phiMax_ - phiMin_);

  if (fVerbosity) 
    cout << "DeadCell phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;

  // if cell energy is less than this fraction of its neighbors, cell is marked as cool:
  coolcellfrac_ = ps.getUntrackedParameter<double>("coolcellfrac",0.25); 

  // The number of consecutive events for which a cell must be below (pedestal+Nsigma) to be considered dead:
  checkNevents_ = ps.getUntrackedParameter<int>("checkNevents",1000);

  // Number of sigma to use in evalutaed (pedestal+Nsigma)
  Nsigma_ = ps.getUntrackedParameter<double>("ped_Nsigma",2.);

  // Cells with total ADC counts <= this value will be marked dead in an event:
  minADCcount_ = ps.getUntrackedParameter<double>("minADCcount",0.);

  if (fVerbosity)
    {
      cout <<"DeadCell NADA coolcells must have energy fraction of <"<<coolcellfrac_<<"* (neighbors' average energy)"<<endl;
      cout <<"DeadCell cool digis are checked every "<<checkNevents_<<" events"<<endl;
      cout <<"\tCool digis must have energy <(pedestal + "<<Nsigma_<<"sigma)"<<endl;
      cout <<"DeadCell digis are considered dead if ADC count is <= "<<minADCcount_<<endl;
    }

  // Values for comparing cell energies to neighbors
  floor_ = ps.getUntrackedParameter<double>("deadcellfloor",-0.5);
  mindiff_ = ps.getUntrackedParameter<double>("deadcellmindiff",0.5);


  ievt_=0;
  if (m_dbe !=NULL) 
    {
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void HcalDeadCellMonitor::setupHists(DeadCellHists& hist,  DQMStore* dbe)
{
  /*  
      Instantiates histogram instances for all histograms in DeadCellHist object 'hist'.
      (Each subdetector gets its own DeadCellHist object, and there's a separate object for
      the total "Hcal" combination.)
  */

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
  hist.deadADC_map = m_dbe->book2D(hist.subdet+"_deadADCOccupancyMap",hist.subdet+" No ADC Count Occupancy Map",
				   etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  hist.deadADC_eta = m_dbe->book1D(hist.subdet+"_deadADCEta",hist.subdet+" No ADC Count Eta ",
				   etaBins_,etaMin_,etaMax_);
  hist.ADCdist = m_dbe->book1D(hist.subdet+"_ADCdist",hist.subdet+" ADC count distribution",
			       128,0,128);
  hist.NADA_cool_cell_map = m_dbe->book2D(hist.subdet+"_NADA_CoolCellMap",hist.subdet+" Cool Cells",
					  etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  hist.digiCheck = m_dbe->book2D(hist.subdet+"_digiCheck",hist.subdet+" Check that digi was found",
				 etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  hist.cellCheck = m_dbe->book2D(hist.subdet+"_cellCheck",hist.subdet+" Check that cell hit was found",
				 etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  hist.deadcapADC_map.push_back(m_dbe->book2D(hist.subdet+"_DeadCap0","Map of "+hist.subdet+" Events with no ADC hits for capid=0",
					      etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
  hist.deadcapADC_map.push_back(m_dbe->book2D(hist.subdet+"_DeadCap1","Map of "+hist.subdet+" Events with no ADC hits for capid=1",
					      etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
  hist.deadcapADC_map.push_back(m_dbe->book2D(hist.subdet+"_DeadCap2","Map of "+hist.subdet+" Events with no ADC hits for capid=2",
					      etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
  hist.deadcapADC_map.push_back(m_dbe->book2D(hist.subdet+"_DeadCap3","Map of "+hist.subdet+" Events with no ADC hits for capid=3",
					      etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
  string Nsig;
  stringstream out;
  out <<Nsigma_;
  Nsig=out.str();
  string consec;
  stringstream out2;
  out2<<checkNevents_;
  consec=out2.str();
  hist.above_pedestal = m_dbe->book2D(hist.subdet+"_abovePed",hist.subdet+" cells above pedestal+"+Nsig+"sigma",
				      etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  hist.coolcell_below_pedestal = m_dbe->book2D(hist.subdet+"_CoolCell_belowPed",
					       hist.subdet+" cells below (pedestal+"+Nsig+"sigma) for "+consec+" consecutive events",
					       etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  char PedTemp[256];
  sprintf(PedTemp,"%sAbovePedTemp",hist.subdet.c_str());
  hist.above_pedestal_temp = new TH2F(PedTemp,"Don't look at this!",
				      etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

  // Create histograms for each depth
  char DepthName[256];
  for (unsigned int d=0;d<4;++d)
    {
      // ADC count <= min value
      sprintf(DepthName,"%s_DeadADCmap_Depth%i",hist.subdet.c_str(),d+1);
      hist.deadADC_map_depth.push_back( m_dbe->book2D(DepthName,DepthName,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
      // Cell is cool compared to neighbors
      sprintf(DepthName,"%s_NADACoolCell_Depth%i",hist.subdet.c_str(),d+1);
      hist.NADA_cool_cell_map_depth.push_back(m_dbe->book2D(DepthName,DepthName,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
      // Cell is below pedestal+Nsigma
      sprintf(DepthName,"%s_coolcell_below_pedestal_Depth%i",hist.subdet.c_str(),d+1);
      hist.coolcell_below_pedestal_depth.push_back(m_dbe->book2D(DepthName,DepthName,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
      // Digi occupancy plot (redundant?)
      sprintf(DepthName,"%s_digiCheck_Depth%i",hist.subdet.c_str(),d+1);
      hist.digiCheck_depth.push_back(m_dbe->book2D(DepthName,DepthName,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
      // Rechit occupancy plot (redundant?)
      sprintf(DepthName,"%s_cellCheck_Depth%i",hist.subdet.c_str(),d+1);
      hist.cellCheck_depth.push_back(m_dbe->book2D(DepthName,DepthName,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
    }
  
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
  for (unsigned int depth=0;depth<4;++depth)
    {
      hist.deadADC_map_depth[depth]->setAxisTitle("i#eta", 1);
      hist.deadADC_map_depth[depth]->setAxisTitle("i#phi",2);
      hist.NADA_cool_cell_map_depth[depth]->setAxisTitle("i#eta", 1);
      hist.NADA_cool_cell_map_depth[depth]->setAxisTitle("i#phi",2);
      hist.coolcell_below_pedestal_depth[depth]->setAxisTitle("i#eta", 1);
      hist.coolcell_below_pedestal_depth[depth]->setAxisTitle("i#phi",2);
      hist.digiCheck_depth[depth]->setAxisTitle("i#eta", 1);
      hist.digiCheck_depth[depth]->setAxisTitle("i#phi",2);
      hist.cellCheck_depth[depth]->setAxisTitle("i#eta", 1);
      hist.cellCheck_depth[depth]->setAxisTitle("i#phi",2);
      
    }
  hist.above_pedestal->setAxisTitle("i#eta", 1);
  hist.above_pedestal->setAxisTitle("i#phi",2);
  hist.coolcell_below_pedestal->setAxisTitle("i#eta", 1);
  hist.coolcell_below_pedestal->setAxisTitle("i#phi",2);

  return;
} // void HcalDeadCellMonitor::setupHists(...)


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void HcalDeadCellMonitor::processEvent(const HBHERecHitCollection& hbHits, 
				       const HORecHitCollection& hoHits, 
				       const HFRecHitCollection& hfHits,
				       const HBHEDigiCollection& hbhedigi,
				       const HODigiCollection& hodigi,
				       const HFDigiCollection& hfdigi,
				       const HcalDbService& cond)
{
  /*
    Run DeadCell-checking code for all cells in an event.
     Need both digi collections (for checking ADC counts, comparing to pedestals)
     and rechit collections (for comparing energies with neighbors).
  */

  if(!m_dbe) 
    {
      if(fVerbosity) cout <<"<HcalDeadCellMonitor::processEvent>    DQMStore not instantiated!!!\n";
      return;
    }

  ievt_++;
  meEVT_->Fill(ievt_);
  if (fVerbosity) cout <<"<HcalDeadCellMonitor::processEvent>     Starting process"<<endl;
  

  // Process digis
  processEvent_digi(hbhedigi,hodigi,hfdigi,cond); // check for dead digis
  // Process rechits
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void HcalDeadCellMonitor::processEvent_digi(const HBHEDigiCollection& hbhedigi,
					    const HODigiCollection& hodigi,
					    const HFDigiCollection& hfdigi,
					    const HcalDbService& cond)
{

  /*
    Call digi-based Dead Cell monitor code ( check ADC counts,
    compare readout values vs. pedestals).
  */


  if (fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_digi     Starting process"<<endl;

  HcalCalibrationWidths widths;

  // Loop over HBHE
  try
    {
      for (HBHEDigiCollection::const_iterator j=hbhedigi.begin(); j!=hbhedigi.end(); ++j)
	{
	  const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
	  calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private. 
	  cond.makeHcalCalibrationWidth(digi.id(),&widths);

	  // HB goes out to ieta=16; ieta=16 is shared with HE
	  if (abs(digi.id().ieta())<16 || (abs(digi.id().ieta())==16 && ((HcalSubdetector)(digi.id().subdet()) == HcalBarrel)))
	    HcalDeadCellCheck::CheckForDeadDigis(digi,hbHists,hcalHists,
						 Nsigma_,minADCcount_,calibs_,widths,m_dbe,baseFolder_);
	  else 
	    HcalDeadCellCheck::CheckForDeadDigis(digi,heHists,hcalHists,
						 Nsigma_,minADCcount_,calibs_,widths,m_dbe,baseFolder_);
	}
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_digi   No HBHE Digis."<<endl;
    }

  // Loop over HO
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

  // Loop over HF
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


////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void HcalDeadCellMonitor::processEvent_hits(const HBHERecHitCollection& hbHits, 
					    const HORecHitCollection& hoHits, 
					    const HFRecHitCollection& hfHits)
{
  /* 
     Look for dead cells based on rec hit information
     (by comparing to neighboring cell energies)
  */

  if(!m_dbe) 
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits    DQMStore not instantiated!!!\n";
      return;
    }
  if (fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits     Starting process"<<endl;



  /*
    // Implement this code later, if we find that repeated looping over recHits is too slow.
    // I don't think using HcalDetIds as the map key is the best solution, though -- we'll still
    // have to loop over all keys and evaluate each key to see if it is within a "neighboring" 
    // range of the cell in question.
    // Better to use an int key (something like 1000*ieta+10*iphi+depth)?
    // Then we can just loop over neighbors (eta=celleta-1; eta<=celleta+1; ++eta),
    // and look to see if a key is present for the neighboring cell?
    // This might make things faster when there is high occupancy, but the speed increase may be
    // negligible.  -- Jeff , 17 May 2008

  // Make map of all hits found -- used in searching for neighbors
  std::map<HcalDetId, double> rechitmap;
  for (HBHERecHitCollection::const_iterator j=hbHits.begin(); j!=hbHits.end(); ++j)
    {
      //int id = 1000*j->id().ieta()+10*((j->id().iphi())%72)+j->id().depth();
      rechitmap[(HcalDetId)j->id()]=j->energy();
    }
  for (HORecHitCollection::const_iterator j=hoHits.begin(); j!=hoHits.end(); ++j)
    rechitmap[(HcalDetId)j->id()]=j->energy();
  for (HFRecHitCollection::const_iterator j=hfHits.begin(); j!=hfHits.end(); ++j)
    rechitmap[(HcalDetId)j->id()]=j->energy();
  */  

  // Loop over HB digis
  try
    {
      HcalDeadCellCheck::CheckHits(coolcellfrac_,hbHits,hbHists,hcalHists,
				   m_dbe, baseFolder_);
    }
  catch(...)
	{
	  if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HB Hits"<<endl;
	}

  // Loop over HE digis
  try
    {
      HcalDeadCellCheck::CheckHits(coolcellfrac_,hbHits,heHists,hcalHists,
				   m_dbe, baseFolder_);
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HE Hits"<<endl;
    }
  
  // Loop over HO digis
  try
    {
      HcalDeadCellCheck::CheckHits(coolcellfrac_,hoHits,hoHists,hcalHists,
				   m_dbe, baseFolder_);
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HO Hits"<<endl;
    }

  // Loop over HF digis
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

} // void HcalDeadCellMonitor::processEvent_hits(...)


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void HcalDeadCellMonitor::reset_Nevents(DeadCellHists &h)

{
  /*
    Every N events, look for cells that have been persistently below pedestal + Nsigma, and
    plot them in our ped histogram.  Reset the transient histograms that are checking that
    cells are persistently below pedestal.
  */

  if (h.check==0) return;
  
  int eta, phi; // these will store detector eta, phi
  for (int ieta=1;ieta<=etaBins_;++ieta)
    {
      // convert ieta  from histograms to eta (HCAL coordinates)
      eta=ieta+int(etaMin_)-1;
      if (eta==0) continue; // skip eta=0 bin -- unphysical
      
      // Check eta range for each subdetector
      if (h.type==1 && abs(eta)>16) continue; // skip events outside range of HB 
      else if (h.type==2 && (abs(eta)<16 || abs(eta)>29)) // skip events outside range of HE
	continue;
      else if (h.type==3 && abs(eta)>15) continue; // ho should extend to eta=15?
      else if (h.type==4 && abs(eta)<30) continue; // FIXME:  is this the correct condition for HF?

      for (int iphi=1;iphi<=phiBins_;++iphi)
	{
	  // convert iphi from histograms to phi (HCAL coordinates)
	  phi=iphi+int(phiMin_)-1;

	  if (phi<1) continue; 
	  if (phi>72) continue; // detector phi runs from 1-72

	  // At larger eta, phi segmentation is more coarse
	  if (h.type==2) 
	    if ((abs(eta)>20) && (phi%2)!=1) continue; // skip HE even-phi counters where they don't exist

	  else if (h.type==4)
	    {
	      // skip HF counters where they don't exist
	      if ((abs(eta)<40) && (phi%2)!=1) continue; 
	      if ((abs(eta)>39) && (phi%4)!=1) continue;
	    }

	  double temp=h.above_pedestal_temp->GetBinContent(ieta,iphi);

	  if (temp==0)
	    {
	      h.coolcell_below_pedestal->Fill(eta,phi);
	      hcalHists.coolcell_below_pedestal->Fill(eta,phi);
	    }
	  else
	    {
	      h.above_pedestal->Fill(eta,phi,temp);
	      hcalHists.above_pedestal->Fill(eta,phi,temp);
	    }
	} // for (int iphi=1; iphi<phiBins_+1;++iphi)
    } // for (int ieta=1;ieta<etaBins_+1;++ieta)


  h.above_pedestal_temp->Reset();
  hcalHists.above_pedestal_temp->Reset();

  return;
}
