#include "DQM/HcalMonitorTasks/interface/HcalDeadCellMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <math.h>
#include <sstream>
using namespace std;
#include "FWCore/Utilities/interface/CPUTimer.h"


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
			 float Nsigma, 
			 float mincount, // specify Nsigma for pedestal check, mincount for ADC check
			 const HcalDbService& cond,
			 DQMStore* dbe, 
			 bool pedsInFC=false)
  {

    if (!hist.check) return;

    // Timing doesn't work well on individual digis -- times almost always come out as 0.
    //edm::CPUTimer XXX;
    //XXX.reset(); XXX.start();

    string type;
    // Get subdet name associated with type value
    if(hist.type==1) type = "HB";
    else if(hist.type==2) type = "HE"; 
    else if(hist.type==3) type = "HO"; 
    else if(hist.type==4) type = "HF"; 
    else return;

    // Unnecessary -- histogram already declared
    //if(dbe) dbe->setCurrentFolder(baseFolder+"/"+type);

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
    /////all.digiCheck->Fill(digi_eta,digi_phi);


    if (hist.makeDiagnostics)
      {
	hist.digiCheck_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
	/////all.digiCheck_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
      }

    /* Update on 21 May 2008 -- code modified to compare digi values to 
       pedestals regardless of whether ped hits are in ADCs or fC.
       (hcalMonitor.PedestalsInFC user-input boolean is used to determine
        which to use.  It's assumed false (pedestals in ADC, not fC)
    */

    HcalCalibrationWidths widths;
    cond.makeHcalCalibrationWidth(digi.id(),&widths);
    HcalCalibrations calibs;
    calibs= cond.getHcalCalibrations(digi.id());  // Old method was made private. 

    const HcalQIEShape* shape = cond.getHcalShape();
    const HcalQIECoder* coder = cond.getHcalCoder(digi.id());  

    // Loop over the  time slices of the digi to find the time slice with maximum charge deposition
    // We'll assume steeply-peaked distribution, so that charge deposit occurs
    // in slices (i-1) -> (i+2) around maximum deposit time i

    float maxa=0;
    int maxi=0;
    float digival;
    float total_digival=0;
    float total_pedestal=0;
    float total_pedwidth=0;

    for(int i=0; i<digi.size(); ++i)
      {
	int thisCapid = digi.sample(i).capid();

	// Calculate charge deposited (minus pedestal) in either fC or ADC
	if (pedsInFC)
	  digival = coder->charge(*shape,digi.sample(i).adc(),digi.sample(i).capid())-calibs.pedestal(thisCapid);
	else
	  {
	    digival=digi.sample(i).adc()-calibs.pedestal(thisCapid);
	  }
	// Check to see if value is new max
	if(digival >maxa)
	  {
	    maxa=digival ;
	    maxi=i;
	  }
      } // for (int i=0;i<digi.size();++i)	

    // Now loop over 4 time slices around maximum value
    
    //for (int i=0;i<digi.size();++i) // old code ran over all time slices

    
    for (int i=max(0,maxi-1);i<=min(digi.size()-1,maxi+2);++i)
      {
	ADCsum+=digi.sample(i).adc();

	//if (ADCsum!=0) break;
	int thisCapid = digi.sample(i).capid();
	capADC[thisCapid]+=digi.sample(i).adc();

	total_pedestal+=calibs.pedestal(thisCapid);
	// Add widths in quadrature; need to account for correlations between capids at some point
	total_pedwidth+=pow(widths.pedestal(thisCapid),2);

	/* If ADC value above (pedestal+Nsigma_), fill hist.above_pedestal_temp
	   for that particular depth
	   (Cool cells will later be found by looking for empty spots in the
	   above_pedestal_temp histogram)
	*/

	if (pedsInFC)
	  digival = coder->charge(*shape,digi.sample(i).adc(),digi.sample(i).capid());
	else
	  digival = (float)digi.sample(i).adc();

	total_digival+=digival;

	/*
	  cout <<type<< "  ("<<digi_eta<<", "<<digi_phi<<","<<digi_depth<<")  "  <<i<< "  CAPID = "<<thisCapid<<"  ADC = "<<digi.sample(i).adc()<<"  PED = "<<calibs.pedestal(thisCapid)<<" +/- "<<widths.pedestal(thisCapid)<<endl;
	cout <<"\t digival = "<<digival<<endl<<endl;
	*/


	// Not yet sure if this histogram is useful, but it gives an idea of the ADC distributions
	hist.ADCdist->Fill(digi.sample(i).adc());
	/////all.ADCdist->Fill(digi.sample(i).adc());
      } // for (int i = max(0,maxi-1)...)
    

    // Compare sum around max digi value to sum of pedestals
    total_pedwidth=pow(total_pedwidth,0.5);
    if (total_digival>total_pedestal+Nsigma*total_pedwidth)
      {
	hist.above_pedestal_temp_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
	/////all.above_pedestal_temp_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
	hist.above_pedestal->Fill(digi_eta,digi_phi);
	/////all.above_pedestal->Fill(digi_eta,digi_phi);
	hist.above_pedestal_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
	/////all.above_pedestal_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
      }
    
    // If ADCsum <= mincount, cell is considered dead
    if (ADCsum<=mincount)
      {
	hist.deadADC_map->Fill(digi_eta,digi_phi)
;
	if (hist.makeDiagnostics)
	  {
	    hist.deadADC_map_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
	    /////all.deadADC_map_depth[digi_depth-1]->Fill(digi_eta,digi_phi); 
	  }
	//hist.deadADC_temp_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
	hist.deadADC_eta->Fill(digi_eta);
	/////all.deadADC_map->Fill(digi_eta,digi_phi);
	/////all.deadADC_eta->Fill(digi_eta);

	// Dead cell is potentially problematic -- add it to combined "problem cell" histogram
	hist.problemDeadCells->Fill(digi_eta,digi_phi);
	all.problemDeadCells->Fill(digi_eta,digi_phi);
	hist.problemDeadCells_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
	/////all.problemDeadCells_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
	
      }

    // look for individual dead caps
    for (int zz=0;zz<4;++zz)
      {
	if (capADC[zz]<=mincount)
	  {
	    hist.deadcapADC_map[zz]->Fill(digi_eta,digi_phi);
	    /////all.deadcapADC_map[zz]->Fill(digi_eta,digi_phi);
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
		 DQMStore* dbe)
  { 
    if (hist.check==false) 
      return;
    
    string type;
    type=hist.subdet;

    typename Hits::const_iterator CellIter;
    for (CellIter=hits.begin();
	 CellIter!=hits.end(); 
	 ++CellIter)
      {
	// Fill histogram if cell found in hist region

	//if ((CellIter->id().subdet())!=hist.type) continue; // does this cause a slowdown?  Or has Jason's fix cured this?

	HcalDetId id(CellIter->detid().rawId());
	int cell_eta=id.ieta();
	int cell_depth=id.depth();
	if (hist.type==1)
	  {
	    if (abs(cell_eta)>16)
	      continue;
	  }
	// HB starts at |eta|=17, except for one layer at |eta|=16, depth=3
	else if (hist.type==2)
	  {
	    if (abs(cell_eta)<16)
	      continue;
	    else if (abs(cell_eta)==16 && cell_depth!=3)
	      continue;
	  }
	int cell_phi=id.iphi();

	int temp_cell_phi = cell_phi;  // temporary variable for dealing with neighbors at boundaries between different phi segmentations
	

	hist.cellCheck->Fill(cell_eta,cell_phi);
	all.cellCheck->Fill(cell_eta,cell_phi);

	// Don't need these any more (or the cellCheck histos above),
	// but keep them around in case extra diagnostics are desired
	if (hist.makeDiagnostics)
	  {
	    hist.cellCheck_depth[cell_depth-1]->Fill(cell_eta,cell_phi);
	    all.cellCheck_depth[ cell_depth-1]->Fill(cell_eta,cell_phi);
	  }

	// if (id.depth()==2) continue; // skip depth=2 for now
	// if (vetoCell(id)) continue;

	// Sum energies of neighbors around cell
	
	int allneighbors=0; // all neighbors with energy>0 around cell
	int neighbors=0; // subset of allneighbors, with energy > (cell energy + mindiff)
	double neighborE=0; // total energy found from neighbors

	int etaFactor;  // correct for eta regions where phi segmentation is > 5 degrees/cell
	
	int dPhi;

	for (typename Hits::const_iterator neighbor=hits.begin();neighbor!=hits.end();++neighbor)
	  {
	    //if (vetoCell(neighbor->id())) continue;
	    // Calls to .subdet() are too slow?
	    //if  ((HcalSubdetector)(neighbor->id().subdet())!=(HcalSubdetector)(id.subdet())) continue;
	    HcalDetId neighborId(neighbor->detid().rawId());
	    int NeighborEta=neighborId.ieta();
	    // Only check nearest neighbors in eta
	    if (abs(NeighborEta-cell_eta)>=2)
	      continue;
	    
	    if (neighborId.depth()!=cell_depth) continue;
	    
	    if (hist.type==1)
	      {
		if (abs(NeighborEta)>16)
		  continue;
	      }
	    // HB starts at |eta|=17, except for one layer at |eta|=16, depth=3
	    else if (hist.type==2)
	      {
		if (abs(NeighborEta)<16)
		  continue;
		else if (abs(NeighborEta)==16 && cell_depth!=3)
		  continue;
	      }


	    etaFactor = 1+(abs(NeighborEta)>20)+2*(abs(NeighborEta)>39);  // = 1 for phi=5 segments, 2 for phi=10, 4 for phi=20
	    
	    // boundary between phi=5 and phi=10 segmentation
	    if (abs(cell_eta)==20 && abs(NeighborEta)==21)
	      {
		if (temp_cell_phi%2); --temp_cell_phi; // odd cells treated as even
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
		if ((temp_cell_phi%4)==3) ++temp_cell_phi;
	      }

	    // Check just against nearest neighbors (+/-1 in eta, phi, same depth)

	    // Find minimum distance in phi between neighbors
	    dPhi = (abs(neighborId.iphi()-temp_cell_phi));
	    if (dPhi>36)
	      dPhi=72-dPhi;

	    if (dPhi<(1+etaFactor)
		//&& 		 (abs(NeighborEta-cell_eta))<2)
		)
	      {
		// Skip neighbors with negative energy?
		//if (neighbor->energy()<0) continue;
		allneighbors++;
		if ((neighbor->energy()-CellIter->energy())>hist.mindiff)
		  {
		    neighborE+=neighbor->energy();
		    neighbors++;
		  }
	      }
	  }// for (Hits::const_iterator neighbor=hits.begin()...
	
	// Remove cell energy from neighbor calculation
	neighborE-=CellIter->energy();
	neighbors-=1;

	// If cell energy is less than minimum value ("hist.floor") , mark it as cool for the event
	if (CellIter->energy()<hist.floor)
	  {
	    hist.NADA_cool_cell_map->Fill(cell_eta,cell_phi);
	    all.NADA_cool_cell_map->Fill(cell_eta,cell_phi);
	    if (hist.makeDiagnostics)
	      {
		hist.NADA_cool_cell_map_depth[cell_depth-1]->Fill(cell_eta,cell_phi);
		all.NADA_cool_cell_map_depth[ cell_depth-1]->Fill(cell_eta,cell_phi);
	      }
	  }
	
	else
	  {
	    // Require at least half of neighboring cells exceed minimum difference,
	    // and that cell energy < coolcellfrac * (average of neighbors)

	    // hard-code # of neighbors -- may need to be adjusted at phi segmenation boundaries
	    if ((neighbors>4) &&
	       (CellIter->energy()>0. && CellIter->energy()<coolcellfrac*(1.0*neighborE/neighbors)))

	    {
	      hist.NADA_cool_cell_map->Fill(cell_eta, cell_phi);
	      all.NADA_cool_cell_map->Fill( cell_eta, cell_phi);
	      if (hist.makeDiagnostics)
		{
		  hist.NADA_cool_cell_map_depth[cell_depth-1]->Fill(cell_eta,cell_phi);
		  all.NADA_cool_cell_map_depth[ cell_depth-1]->Fill(cell_eta,cell_phi);
		}
	    } 
 
	  } // else (CellIter->energy()>=hist.floor)

      } // for (CellIter=hits.begin()...)

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
  doFCpeds_ = ps.getUntrackedParameter<bool>("PedestalsInFC", false);

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

  // Set up subdetector histograms 
  hbHists.check=ps.getUntrackedParameter<bool>("checkHB", 1); 
  heHists.check=ps.getUntrackedParameter<bool>("checkHE", 1); 
  hoHists.check=ps.getUntrackedParameter<bool>("checkHO", 1); 
  hfHists.check=ps.getUntrackedParameter<bool>("checkHF", 1); 
  hcalHists.check=(hbHists.check || heHists.check || hoHists.check || hfHists.check); 
  
  hcalHists.makeDiagnostics=ps.getUntrackedParameter<bool>("MakeDeadCellDiagnosticPlots",makeDiagnostics);
  hbHists.makeDiagnostics=hcalHists.makeDiagnostics;
  heHists.makeDiagnostics=hcalHists.makeDiagnostics; 
  hoHists.makeDiagnostics=hcalHists.makeDiagnostics; 
  hfHists.makeDiagnostics=hcalHists.makeDiagnostics; 


  ievt_=0;
  if (m_dbe !=NULL) 
    {
      m_dbe->setCurrentFolder(baseFolder_);
      
      meEVT_ = m_dbe->bookInt("DeadCell Task Event Number");    
      meEVT_->Fill(ievt_);

      meCheckN_ = m_dbe->bookInt("CheckNevents");
      meCheckN_ -> Fill(checkNevents_);
          
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

  if (hist.check==false) return;
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

  // Convert Nsigma, consecutive events to string values
  string Nsig;
  stringstream out;
  out <<Nsigma_;
  Nsig=out.str();
  string consec;
  stringstream out2;
  out2<<checkNevents_;
  consec=out2.str();


  // Set main directory showing occupancy plots
  m_dbe->setCurrentFolder(baseFolder_+"/"+hist.subdet.c_str());

  hist.problemDeadCells = m_dbe->book2D(hist.subdet+"ProblemDeadCells",
					hist.subdet+" Dead Cell rate for potentially bad cells",
					etaBins_, etaMin_, etaMax_,
					phiBins_, phiMin_, phiMax_);

  std::stringstream histname;
  std::stringstream histtitle;
  for (int depth=0;depth<4;++depth)
    {
      m_dbe->setCurrentFolder(baseFolder_+"/"+hist.subdet.c_str()+"/expertPlots");
      histname.str("");
      histtitle.str("");
      histname<<hist.subdet+"ProblemDeadCells_depth"<<depth+1;
      histtitle<<hist.subdet+" Dead Cell rate for potentially bad cells (depth "<<depth+1<<")";
      hist.problemDeadCells_depth[depth]=(m_dbe->book2D(histname.str().c_str(),histtitle.str().c_str(),
						      etaBins_,etaMin_,etaMax_,
						      phiBins_,phiMin_,phiMax_));
    } // for (int depth=0;...)

  


  char DepthName[256];
  char DepthTitle[256];


  m_dbe->setCurrentFolder(baseFolder_+"/"+hist.subdet.c_str());
  hist.deadADC_map = m_dbe->book2D(hist.subdet+"_OccupancyMap_deadADC",
				   hist.subdet+" No ADC Count Occupancy Map",
				   etaBins_,etaMin_,etaMax_,
				   phiBins_,phiMin_,phiMax_);
  hist.NADA_cool_cell_map = m_dbe->book2D(hist.subdet+"_OccupancyMap_NADA_CoolCell",
					  hist.subdet+" Cool Cells",
					  etaBins_,etaMin_,etaMax_,
					  phiBins_,phiMin_,phiMax_);
  
  hist.coolcell_below_pedestal = m_dbe->book2D(hist.subdet+"_OccupancyMap_belowPedestal",
					       hist.subdet+" cells below (pedestal+"+Nsig+"sigma) for "+consec+" consecutive events",
					       etaBins_,etaMin_,etaMax_,
					       phiBins_,phiMin_,phiMax_);

  // Put additional plots in expertPlots subfolder
  m_dbe->setCurrentFolder(baseFolder_+"/"+hist.subdet.c_str()+"/expertPlots");
  hist.deadADC_eta = m_dbe->book1D(hist.subdet+"_deadADCEta",
				   hist.subdet+" No ADC Count Eta ",
				   etaBins_,etaMin_,etaMax_);
  
  hist.ADCdist = m_dbe->book1D(hist.subdet+"_ADCdist",
			       hist.subdet+" ADC count distribution",
			       128,0,128);
  
  hist.digiCheck = m_dbe->book2D(hist.subdet+"_digiCheck",hist.subdet+" Check that digi was found",
				 etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  hist.cellCheck = m_dbe->book2D(hist.subdet+"_cellCheck",hist.subdet+" Check that cell hit was found",
				 etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  hist.deadcapADC_map[0]=(m_dbe->book2D(hist.subdet+"_DeadCap0","Map of "+hist.subdet+" Events with no ADC hits for capid=0",
					etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
  hist.deadcapADC_map[1]=(m_dbe->book2D(hist.subdet+"_DeadCap1","Map of "+hist.subdet+" Events with no ADC hits for capid=1",
					etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
  hist.deadcapADC_map[2]=(m_dbe->book2D(hist.subdet+"_DeadCap2","Map of "+hist.subdet+" Events with no ADC hits for capid=2",
					etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
  hist.deadcapADC_map[3]=(m_dbe->book2D(hist.subdet+"_DeadCap3","Map of "+hist.subdet+" Events with no ADC hits for capid=3",
					etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));

  hist.above_pedestal = m_dbe->book2D(hist.subdet+"_abovePed",hist.subdet+" cells above pedestal+"+Nsig+"sigma",
				      etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);


  // Create histograms for each depth

  for (unsigned int d=0;d<4;++d)
    {
      sprintf(DepthName,"%s/%s/expertPlots/BelowPedestal",baseFolder_.c_str(),hist.subdet.c_str()); 
      m_dbe->setCurrentFolder(DepthName); 
      

      // Always form pedestal plots for each depth
      // Cell is below pedestal+Nsigma
      sprintf(DepthName,"%s_coolcell_below_pedestal_Depth%i",hist.subdet.c_str(),d+1);
      hist.coolcell_below_pedestal_depth[d]=(m_dbe->book2D(DepthName,DepthName,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));


      // Cell above pedestal+Nsigma 
      sprintf(DepthName,"%s_cell_above_pedestal_Depth%i",hist.subdet.c_str(), 
              d+1); 
      sprintf(DepthTitle,"%s Depth%i Cell Above Pedestal + %.2f sigma",hist.subdet.c_str(), 
              d+1,Nsigma_); 
      hist.above_pedestal_depth[d]=( m_dbe->book2D(DepthName, 
                                                         DepthTitle, 
                                                         etaBins_,etaMin_,etaMax_, 
                                                         phiBins_,phiMin_,phiMax_)); 
      sprintf(DepthName,"%s_cell_above_pedestal_Depth%i_temp",hist.subdet.c_str(), 
              d+1); 
      sprintf(DepthTitle,"(Temp) %s Depth%i Cell Above Pedestal + %.2f sigma",hist.subdet.c_str(), 
              d+1,Nsigma_); 
      hist.above_pedestal_temp_depth[d]=(new TH2F(DepthName, 
						  DepthTitle, 
						  etaBins_,etaMin_,etaMax_, 
						  phiBins_,phiMin_,phiMax_)); 

      if (!hist.makeDiagnostics) continue; // skip remaining depth plots if diagnostics are off


      sprintf(DepthName,"%s/%s/Diagnostics/Depth%i",baseFolder_.c_str(),hist.subdet.c_str(),d+1);
      m_dbe->setCurrentFolder(DepthName);

      // RecHit Occupancy Plots
      sprintf(DepthName,"%s_cellCheck_Depth%i",hist.subdet.c_str(),d+1);
      hist.cellCheck_depth[d]=(m_dbe->book2D(DepthName,DepthName,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));

      // ADC count <= min value
      
      sprintf(DepthName,"%s_DeadADCmap_Depth%i",hist.subdet.c_str(),d+1);
      hist.deadADC_map_depth[d]=( m_dbe->book2D(DepthName,DepthName,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
      
      //sprintf(DepthName,"%s_DeadADCmap_Depth%i_temp",hist.subdet.c_str(),d+1);
      //hist.deadADC_temp_depth[d]=(new TH2F(DepthName,DepthName,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
      
      // Cell is cool compared to neighbors
      sprintf(DepthName,"%s_NADACoolCell_Depth%i",hist.subdet.c_str(),d+1);
      hist.NADA_cool_cell_map_depth[d]=(m_dbe->book2D(DepthName,DepthName,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
      //cout <<"NAME = "<<DepthName<<endl;

      // Digi occupancy plot (redundant?)
      sprintf(DepthName,"%s_digiCheck_Depth%i",hist.subdet.c_str(),d+1);
      hist.digiCheck_depth[d]=(m_dbe->book2D(DepthName,DepthName,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));

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
  for (unsigned int icap=0;icap<4;++icap)
    {
      hist.deadcapADC_map[icap]->setAxisTitle("i#eta", 1);
      hist.deadcapADC_map[icap]->setAxisTitle("i#phi",2);
    }
  for (unsigned int depth=0;depth<4;++depth)
    {
      if (!hist.makeDiagnostics) continue; 
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
      hist.above_pedestal_depth[depth]->setAxisTitle("i#eta", 1);
      hist.above_pedestal_depth[depth]->setAxisTitle("i#phi",2);
      hist.coolcell_below_pedestal_depth[depth]->setAxisTitle("i#eta", 1);
      hist.coolcell_below_pedestal_depth[depth]->setAxisTitle("i#phi",2);
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

      // fill HcalHists only every N events
      //fill_Nevents(hcalHists, hbHists, heHists, hoHists, hfHists);
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

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start(); 
    }

  if (fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_digi     Starting process"<<endl;

  // Loop over HBHE
  try
    {
      for (HBHEDigiCollection::const_iterator j=hbhedigi.begin(); j!=hbhedigi.end(); ++j)
	{
	  const HBHEDataFrame digi = (const HBHEDataFrame)(*j);

	  // HB goes out to ieta=16; ieta=16 is shared with HE
	  if ((HcalSubdetector)(digi.id().subdet()) == HcalBarrel)
	    HcalDeadCellCheck::CheckForDeadDigis(digi,hbHists,hcalHists,
						 Nsigma_,minADCcount_,
						 cond,m_dbe,doFCpeds_);
	  else 
	    HcalDeadCellCheck::CheckForDeadDigis(digi,heHists,hcalHists,
						 Nsigma_,minADCcount_,
						 cond,m_dbe,doFCpeds_);

	} // for (HBHEDigiCollection::const_iterator j...)
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_digi   No HBHE Digis."<<endl;
    }

  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalDeadCell DIGI HBHE-> " << cpu_timer.cpuTime() << std::endl;
  cpu_timer.reset(); cpu_timer.start(); 
    }

  // Loop over HO
  try
    {
      for (HODigiCollection::const_iterator j=hodigi.begin(); j!=hodigi.end(); ++j)
	{
	  const HODataFrame digi = (const HODataFrame)(*j);
	  HcalDeadCellCheck::CheckForDeadDigis(digi,hoHists,hcalHists,
					       Nsigma_,minADCcount_,
					       cond,m_dbe,doFCpeds_);
	}
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_digi   No HO Digis."<<endl;
    }

  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalDeadCell HO DIGI-> " << cpu_timer.cpuTime() << std::endl;
      
      cpu_timer.reset(); cpu_timer.start(); 
    }

  // Load HF
  try
    {
      for (HFDigiCollection::const_iterator j=hfdigi.begin(); j!=hfdigi.end(); ++j)
	{
	  const HFDataFrame digi = (const HFDataFrame)(*j);
	  HcalDeadCellCheck::CheckForDeadDigis(digi,hfHists,hcalHists,
					       Nsigma_,minADCcount_,
					       cond,m_dbe,doFCpeds_);
	}
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_digi   No HF Digis."<<endl;
    }

  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalDeadCell HF DIGI-> " << cpu_timer.cpuTime() << std::endl;
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

  return;  // CheckHits doesn't seem to provide much info?

  if(!m_dbe) 
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits    DQMStore not instantiated!!!\n";
      return;
    }
  if (fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits     Starting process"<<endl;


  // Loop over HB rechits
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  try
    {
      HcalDeadCellCheck::CheckHits(coolcellfrac_,hbHits,hbHists,
				   hcalHists,m_dbe);
    }
  catch(...)
	{
	  if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HB Hits"<<endl;
	}
  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER:: HcalDeadCell HB RECHITS-> " << cpu_timer.cpuTime() << std::endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  // Loop over HE Rechits
  try
    {
      HcalDeadCellCheck::CheckHits(coolcellfrac_,hbHits,heHists,
				   hcalHists,m_dbe);
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HE Hits"<<endl;
    }

  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalDeadCell HE RECHITS-> " << cpu_timer.cpuTime() << std::endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  // Loop over HO rechits
  try
    {
      HcalDeadCellCheck::CheckHits(coolcellfrac_,hoHits,hoHists,
				   hcalHists,m_dbe);
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HO Hits"<<endl;
    }

  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalDeadCell HO RECHITS-> " << cpu_timer.cpuTime() << std::endl;
      cpu_timer.reset(); cpu_timer.start();
    }
  
  // Loop over HF rechits
  try
    {
      HcalDeadCellCheck::CheckHits(coolcellfrac_,hfHits,
				   hfHists,hcalHists,m_dbe);
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalDeadCellMonitor::processEvent_hits:   Could not process HF Hits"<<endl;
    }
  
  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalDeadCell HF RECHITS-> " << cpu_timer.cpuTime() << std::endl;
    }
  return;

} // void HcalDeadCellMonitor::processEvent_hits(...)


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void HcalDeadCellMonitor::reset_Nevents(DeadCellHists &h)

{
  /*
    Every N events, look for cells that have been persistently below pedestal + Nsigma, and plot them in our ped histogram.  Reset the transient histograms that are checking that cells are persistently below pedestal.
  */

  if (fVerbosity)
    cout <<"<HcalDeadCellMonitor> Entered reset_Nevents routine"<<endl;
  if (h.check==false) return;
  
  int eta, phi; // these will store detector eta, phi
  for (int ieta=1;ieta<=etaBins_;++ieta)
    {
      // convert ieta  from histograms to eta (HCAL coordinates)
      eta=ieta+int(etaMin_)-1;
      
      if (eta==0) continue; // skip eta=0 bin -- unphysical
      if (abs(eta)>41) continue; // skip unphysical "boundary" bins in histogram
      
      // Check eta range for each subdetector
      if (h.type==1 && abs(eta)>16) continue; // skip events outside range of HB 
      else if (h.type==2 && (abs(eta)<16 || abs(eta)>29)) // skip events outside range of HE
	continue;
      else if (h.type==3 && abs(eta)>15) continue; // ho should extend to eta=15?
      else if (h.type==4 && abs(eta)<29) continue; //

      for (int iphi=1;iphi<=phiBins_;++iphi)
	{
	  // convert iphi from histograms to phi (HCAL coordinates)
	  phi=iphi+int(phiMin_)-1;

	  if (phi<1) continue; 
	  if (phi>72) continue; // detector phi runs from 1-72

	  // Now cut on physical detector boundaries

	  // At larger eta, phi segmentation is more coarse
	  if (h.type==2) 
	    {
	      if ((abs(eta)>20) && (phi%2)!=1) continue; // skip HE even-phi counters where they don't exist
	    }
	  else if (h.type==4)
	    {
	      // skip HF counters where they don't exist
	      if ((abs(eta)<40) && (phi%2)!=1) continue; 
	      if ((abs(eta)>39) && (phi%4)!=3) continue; // starting at eta=39, values are 3, 7, 11, ...
	    }

	  double temp;

	  for (int d=1;d<5;++d)
	    {
	      if (h.type==1) // HB -- runs from eta=1-16
		{
		  if (d>2) 
		    continue;  //HB only has two depths
		  if (d==2 && abs(eta)<15)
		    continue; // depth=2 only for eta=15,16
		}
	      if (h.type==2) // HE -- runs from eta=16-29
		{
		  if (d==4)
		    continue; // HE only has 3 depths
		  
		  if (d==3)
		    {
		      if (abs(eta)!=16 && abs(eta)!=27 && abs(eta)!=28)
			continue; // HE has depth=3 only for eta=16,27,28
		    }
		  if (abs(eta)==16 && d!=3)
		    continue; // one layer only for HE at eta=16 -- depth=3
		  if (abs(eta)==17 && d!=1) 
		    continue; // one layer only for HE at eta=17 -- depth=1
		} // if h.type==2


	      if (h.type==3 && d<4)
		continue;  // HO -- only has depth=4

	      if (h.type==4 && d>2) 
		continue;  // HF -- only has depth=1,2

	      // Check last N events to see which cells were above pedestal
	      temp=h.above_pedestal_temp_depth[d-1]->GetBinContent(ieta,iphi);
	      //if (h.type==3) cout <<"\t\ttemp = "<<temp<<endl;
	      if (temp==0)
		{
		  //if (h.cellCheck_depth[d-1]->getBinContent(ieta,iphi)!=0) // no longer require a rechit for the cell -- zero suppression means not all cells will have hits
		    
		    {
		      h.coolcell_below_pedestal->Fill(eta,phi,checkNevents_);
		      /////hcalHists.coolcell_below_pedestal->Fill(eta,phi,checkNevents_);
		      h.coolcell_below_pedestal_depth[d-1]->Fill(eta,phi,checkNevents_);
		      /////hcalHists.coolcell_below_pedestal_depth[d-1]->Fill(eta,phi,checkNevents_);
		      // Cells consistently below pedestal go to combined "problem cell" histogram
		      hcalHists.problemDeadCells->Fill(eta,phi,checkNevents_);
		      h.problemDeadCells->Fill(eta,phi,checkNevents_);


		      /////hcalHists.problemDeadCells_depth[d-1]->Fill(eta,phi,checkNevents_);
		      h.problemDeadCells_depth[d-1]->Fill(eta,phi,checkNevents_);


		    }
		}
	    } // for (int d=1;d<5;++d)

	} // for (int iphi=1; iphi<phiBins_+1;++iphi)
    } // for (int ieta=1;ieta<etaBins_+1;++ieta)


  for (int d=0;d<4;++d)
    {
      h.above_pedestal_temp_depth[d]->Reset();
    }

  return;
} // reset_Nevents(...)


void HcalDeadCellMonitor::fill_Nevents(DeadCellHists& hcal, 
				       DeadCellHists& hb, DeadCellHists& he,
				       DeadCellHists& ho, DeadCellHists& hf)
{
  // JEFF
  // Idead is to fill overall HcalPlots only ever N events.  However, this seems to take a lot of time, so I'm disabling the hcal fills for now. -- 4 July 2008

  int eta,phi;
  float newval;

  hcal.digiCheck->Reset();
  for (int ieta=1;ieta<=etaBins_;++ieta)
    {
      for (int iphi=1;iphi<=phiBins_;++iphi)
	{
	  newval= hb.digiCheck->getBinContent(ieta,iphi)+he.digiCheck->getBinContent(ieta,iphi)+ho.digiCheck->getBinContent(ieta,iphi)+hf.digiCheck->getBinContent(ieta,iphi);
	  if (newval==0) continue; // ignore bins with no entries
	  eta=ieta+int(etaMin_)-1;
	  phi=iphi+int(phiMin_)-1;
	
	  hcal.digiCheck->Fill(eta,phi,newval);
	} // int iphi=1;...
    } // int ieta=1;...

}//void HcalDeadCellMonitor::fill_Nevents(...)
				       


void HcalDeadCellMonitor::clearME()
{
  // override base class function to clear DeadCellMonitor-specific directories
  if(m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_);
      m_dbe->removeContents();    

      std::vector< string > subdets;
      subdets.push_back("HCAL");
      subdets.push_back("HB");
      subdets.push_back("HE");
      subdets.push_back("HO");
      subdets.push_back("HF");

      char depthName[256];

      for (unsigned int i=0;i<subdets.size();++i)
	{
	  m_dbe->setCurrentFolder(baseFolder_+"/"+subdets[i].c_str());
	  m_dbe->removeContents();
	  m_dbe->setCurrentFolder(baseFolder_+"/expertPlots");
	  m_dbe->removeContents();

	  for (int d=1;d<5;++d)
	    {
	      sprintf(depthName,"%s/%s/Diagnostics/Depth%i",baseFolder_.c_str(),subdets[i].c_str(), d);
	      m_dbe->setCurrentFolder(depthName);
	      m_dbe->removeContents();
	    }
	} // for (int i=0;i<subdets.size();++i)

    } // if (m_dbe)
  return;
} // void HcalDeadCellMonitor::clearME()


void HcalDeadCellMonitor::done()
{
  int eta,phi;
  float binval;

  if (fVerbosity)
    cout <<"<HcalDeadCellMonitor>  Summary of Dead Cells in Run: "<<endl;

  for (int ieta=1;ieta<=etaBins_;++ieta)
    {
      for (int iphi=1;iphi<=phiBins_;++iphi)
	{
	  eta=ieta+int(etaMin_)-1;
	  phi=iphi+int(phiMin_)-1;
	  
	  for (int d=0;d<4;++d)
	    {
	      binval=hbHists.problemDeadCells_depth[d]->getBinContent(ieta,iphi);
	      if (fVerbosity && binval>0) cout <<"Dead Cell "<<"HB("<<eta<<", "<<phi<<", "<<d+1<<") in "<<binval<<"/"<<ievt_<<" events"<<endl;
	    }
	  for (int d=0;d<4;++d)
	    {
	      binval=heHists.problemDeadCells_depth[d]->getBinContent(ieta,iphi);
	      if (fVerbosity && binval>0) cout <<"Dead Cell "<<"HE("<<eta<<", "<<phi<<", "<<d+1<<") in "<<binval<<"/"<<ievt_<<" events"<<endl;
	    }
	  for (int d=0;d<4;++d)
	    {
	      binval=hoHists.problemDeadCells_depth[d]->getBinContent(ieta,iphi);
	      if (fVerbosity && binval>0) cout <<"Dead Cell "<<"HO("<<eta<<", "<<phi<<", "<<d+1<<") in "<<binval<<"/"<<ievt_<<" events"<<endl;
	    }
	  for (int d=0;d<4;++d)
	    {
	      binval=hfHists.problemDeadCells_depth[d]->getBinContent(ieta,iphi);
	      if (fVerbosity && binval>0) cout <<"Dead Cell "<<"HF("<<eta<<", "<<phi<<", "<<d+1<<") in "<<binval<<"/"<<ievt_<<" events"<<endl;
	    }


	  
	} // for (int iphi=1...)
    } // for (int ieta = 1...)


} // void HcalDeadCellMonitor::done()
