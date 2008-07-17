#include "DQM/HcalMonitorTasks/interface/HcalHotCellMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <math.h>

// Use for stringstream
#include <iostream>
#include <iomanip>

namespace HcalHotCellCheck
{
  // Make my own copy of vetoCell in HcalHotCellCheck namespace?
  // Surely there's a better way...
  bool vetoCell(HcalDetId id, vector<string> vetoCells_)
  {
    if(vetoCells_.size()==0) return false;
    for(unsigned int i = 0; i< vetoCells_.size(); i++)
      {
	unsigned int badc = atoi(vetoCells_[i].c_str());
	if(id.rawId() == badc) return true;
      }
    return false;
  } // bool vetoCell(...)


  template<class Digi>
  void CheckDigi(const Digi& digi, HotCellHists& h, 
		 HotCellHists& hcal,
		 const HcalDbService& cond,
		 DQMStore* dbe,
		 bool pedsInFC=false)
  {

    if (!h.check) return;

    if (h.fVerbosity) cout <<"Entered CheckDigi for type = "<<h.type<<endl;
    int digi_eta=digi.id().ieta();
    int digi_phi=digi.id().iphi();
    int digi_depth=digi.id().depth();
    
    HcalCalibrationWidths widths;
    cond.makeHcalCalibrationWidth(digi.id(),&widths);
    HcalCalibrations calibs;
    calibs= cond.getHcalCalibrations(digi.id());  // Old method was made private. 

    const HcalQIEShape* shape = cond.getHcalShape();
    const HcalQIECoder* coder = cond.getHcalCoder(digi.id());  

    // Loop over the time slices of the digi to find the time slice with maximum charge deposition
    // We'll assume steeply-peaked distribution, so that charge deposit occurs
    // in slices (i-1) -> (i+2) around maximum deposit time i

    float maxa=0;
    int maxi=0;
    float digival=0;

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
    float total_digival=0; 
    float total_pedestal=0; 
    float total_pedwidth=0; 

    //for (int i=0;i<digi.size();++i) // old code ran over all 10 slices
    for (int i=max(0,maxi-1);i<=min(digi.size()-1,maxi+2);++i)
      {
	int thisCapid = digi.sample(i).capid();

	total_pedestal+=calibs.pedestal(thisCapid);
	// Add widths in quadrature; need to account for correlations between capids at some point
	total_pedwidth+=pow(widths.pedestal(thisCapid),2);
	if (pedsInFC)
	  {
	    digival = coder->charge(*shape,digi.sample(i).adc(),digi.sample(i).capid());
	  }
	else
	  digival = (float)digi.sample(i).adc();

	total_digival+=digival;
      } //for (int i=max(0,maxi-1)...)

    // protect against dividing by zero
    if (total_pedwidth==0)
      total_pedwidth=0.00000001;

    total_pedwidth=pow(total_pedwidth,0.5);

    // Diagnostic plot shows digi energy value / pedestal width
    if (h.makeDiagnostics)
      {

	// Diagnostic plots of ped values will only give sensible results for non-ZS runs
	h.pedestalValues_depth[digi_depth-1]->Fill(digi_eta,digi_phi,total_pedestal);
	h.pedestalWidths_depth[digi_depth-1]->Fill(digi_eta,digi_phi,total_pedwidth);
	h.DigiEnergyDist->Fill((total_digival-total_pedestal)/total_pedwidth);
	//////hcal.DigiEnergyDist->Fill((total_digival-total_pedestal)/total_pedwidth);
      }
    if (h.makeDiagnostics)
      h.hotcellsigma->Fill((total_digival-total_pedestal)/total_pedwidth);
    if (total_digival-total_pedestal>h.hotDigiSigma*total_pedwidth)
      {
	h.abovePedSigma->Fill(digi_eta,digi_phi);
	/////hcal.abovePedSigma->Fill(digi_eta,digi_phi);

	h.problemHotCells->Fill(digi_eta,digi_phi);
	hcal.problemHotCells->Fill(digi_eta,digi_phi);

	h.problemHotCells_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
	/////hcal.problemHotCells_depth[digi_depth-1]->Fill(digi_eta,digi_phi);
      }

    /*
      // these are pretty space-intensive; skip them until needed
    if (h.makeDiagnostics)
      {
	for (int sigma=0;sigma<=4;++sigma)
	  {
	    if (total_digival-total_pedestal>sigma*total_pedwidth)
	      {
		h.digiPedestalPlots[sigma]->Fill(digi_eta,digi_phi);
		hcal.digiPedestalPlots[sigma]->Fill(digi_eta,digi_phi);
		h.digiPedestalPlots_depth[sigma][digi_depth-1]->Fill(digi_eta,digi_phi);
		hcal.digiPedestalPlots_depth[sigma][digi_depth-1]->Fill(digi_eta,digi_phi);
	      }
	    
	  } // for (int sigma = 0)
      } // if (h.makeDiagnostics)
    */
    return;
  } // void CheckDigi(...)




  template<class Hits>
  void threshCheck(const Hits& hits, HotCellHists& h, HotCellHists& hcal)
  {
    if (!h.check) return;

    // Initialize values of max-energy cell 
    h.enS=-1000., h.tS=0., h.etaS=0, h.phiS=0, h.depthS=0;
    h.idS=0;

    if (h.fVerbosity)
      cout <<"<HcalHotCellMonitor> Looping over HCAL subdetector "<<h.name.c_str()<<endl; 
    
    if(hits.size()>0)
      {
	// Loop over all hits
	typename Hits::const_iterator CellIter;
	for (CellIter=hits.begin(); 
	     CellIter!=hits.end(); 
	     ++CellIter) 
	  { 
	    HcalDetId id(CellIter->detid().rawId());
	    // Check that subdetector region is correct
	    if ((int)(id.subdet())!=h.type) continue;
	    
	    // Skip vetoed cells
	    // vetoCell(id not available in this namespace.  Make my own copy of vetoCell?

	    if(h.vetoCells.size()>0 && vetoCell(id,h.vetoCells))
	      {
		if (h.fVerbosity) cout <<"Vetoed cell with id = "<<id<<endl;
		continue;
	      }

	    double cellenergy=CellIter->energy();
	    int celldepth = id.depth();
	    int celleta = id.ieta();
	    int cellphi = id.iphi();

	    //Diagnostic plot show energy distribution of recHits
	    if (h.makeDiagnostics)
	      {
		h.RecHitEnergyDist->Fill(cellenergy);
		/////hcal.RecHitEnergyDist->Fill(cellenergy);
		h.RecHitEnergyDist_depth[celldepth-1]->Fill(cellenergy);
		/////hcal.RecHitEnergyDist_depth[celldepth-1]->Fill(cellenergy);
	      }

	    // First threshold is used for ID'ing problem cells
	    if (cellenergy>h.thresholds[0])
	      {
		h.problemHotCells->Fill(celleta,cellphi);
		hcal.problemHotCells->Fill(celleta,cellphi);
		h.problemHotCells_depth[celldepth-1]->Fill(celleta,cellphi);
		/////hcal.problemHotCells_depth[celldepth-1]->Fill(celleta,cellphi);

	      }

	    for (int k=0;k<int(h.thresholds.size());++k)
	      {
		if (cellenergy>h.thresholds[k])
		  {
		    if (h.threshOccMap[k]!=0)
		      {
			h.threshOccMap[k]->Fill(celleta,
						cellphi);
			/////hcal.threshOccMap[k]->Fill(celleta,cellphi);
		      }
		    if (h.threshEnergyMap[k]!=0)
		      {
			h.threshEnergyMap[k]->Fill(celleta,
						   cellphi,
						   cellenergy);
			/*
			hcal.threshEnergyMap[k]->Fill(celleta,
						      cellphi,
						      cellenergy);
			*//////
		      }
		    // Fill histograms for individual layers
		    if (h.makeDiagnostics)
		      {
			if (h.threshOccMap_depth[k][celldepth-1]!=0)
			  {
			    h.threshOccMap_depth[k][celldepth-1]->Fill(celleta,
								       cellphi);
			    /*
			      hcal.threshOccMap_depth[k][celldepth-1]->Fill(celleta,
									  cellphi);
			    *//////
			  }
			if (h.threshEnergyMap_depth[k][celldepth-1]!=0)
			  {
			    h.threshEnergyMap_depth[k][celldepth-1]->Fill(celleta,
									  cellphi,
									  cellenergy);
			    /*
			      hcal.threshEnergyMap_depth[k][celldepth-1]->Fill(celleta,
									     cellphi,
									     cellenergy);
			    *//////
			  }
		      } // if (h.makeDiagnostics)
		  } // if (cellenergy>h.thresholds[k])
	      } // for (int k=0;k<int(h.thresholds.size()...)
	      
	  
	    // Store info from highest-energy cell
	    if(cellenergy>h.enS)
	      {
		h.enS = cellenergy;
		h.tS = CellIter->time();
		h.etaS = celleta;
		h.phiS = cellphi;
		h.idS = 1000*h.etaS;
		h.depthS = celldepth; // change depth before altering idS?
		if(h.idS<0) h.idS -= (10*h.phiS+h.depthS);
		else h.idS += (10*h.phiS+h.depthS);
		//depth = cell->id().depth();
	      }
	    
	  } // loop over all hits
	
	// Fill histogram with info from subdetector cell with largest energy 
	if (h.enS>-1000.)
	  {
	    h.maxCellEnergy->Fill(h.enS);
	    h.maxCellTime->Fill(h.tS);
	    h.maxCellOccMap->Fill(h.etaS,h.phiS);
	    h.maxCellEnergyMap->Fill(h.etaS,h.phiS,h.enS);
	    h.maxCellID->Fill(h.idS);
	    
	    if(h.enS > hcal.enS)
	      {
		hcal.enS = h.enS;
		hcal.tS = h.tS;
		hcal.etaS = h.etaS;
		hcal.phiS = h.phiS;
		hcal.depthS = h.depthS;
	      }
	  } // if (h.enS>-1000.)

      } // if (hits.size()>0)
    return;
  } // void threshcheck



  template<class Hits>
  void nadaCheck(const Hits& hits, HotCellHists& h, HotCellHists& hcal)
  {
    if (!h.check) return;
    h.numhotcells=0;
    h.numnegcells=0;

    // Get nominal cube size 
    int cubeSize = (2*h.nadaMaxDeltaDepth+1)*(2*h.nadaMaxDeltaEta+1)*(2*h.nadaMaxDeltaPhi+1)-1;
    
    if (hits.size()>0)
      {
      typename Hits::const_iterator CellIter;
      
      // Copying NADA algorithm from D0 Note 4057.
      // The implementation needs to be optimized -- double looping over iterators is not an efficient approach?
      
      // Ecube is total energy in cube around cell
      float  Ecube=0;
      
      // Some cuts are variable, depending on cell energy
      float ECubeCut=0;
      float ECellCut=0;

      // Store coordinates of candidate hot cells
      int CellDepth=-1;
      int CellEta=-1000;
      int CellPhi=-1000;

      int vetosize=h.vetoCells.size();
      
      float cellenergy=0;

      if (h.fVerbosity) cout <<"Checking NADA for subdetector "<<h.name.c_str()<<endl;
      for (CellIter=hits.begin(); CellIter!=hits.end(); ++CellIter)
	{
	  HcalDetId id(CellIter->detid().rawId());
	  if (id.subdet()!=h.type) continue;
	  // CellIter points to the current hot cell candidate
	  Ecube=0; // reset Ecube energy counter
	  CellDepth=id.depth();
	  CellEta=id.ieta();
	  CellPhi=id.iphi();

	  if (vetosize>0 && vetoCell(id, h.vetoCells)) continue;
	  
	  cellenergy=CellIter->energy();

	  h.nadaEnergy->Fill(cellenergy);
	  /////hcal.nadaEnergy->Fill(cellenergy);
	  

	  if (h.fVerbosity==2) cout <<"<HcalHotCellMonitor:nadaCheck> Cell Energy = "<<cellenergy<<" at position ("<<CellEta<<", "<<CellPhi<<")"<<endl;

	  // --------------------------- 
	  // Case 1:  E<negative cutoff or E> max cutoff:  Each counts as hot cell
	  
	  if (cellenergy<h.nadaNegCandCut || cellenergy>h.nadaEnergyCandCut2)
	    {
	      // Case 1a:  E< negative cutoff
	      if (cellenergy<h.nadaNegCandCut) 
		{ 
		  if (h.fVerbosity==2) cout <<"<HcalHotCellMonitor:nadaCheck> WARNING:  NEGATIVE "<<h.name.c_str()<<" CELL ENERGY>  Energy = "<<cellenergy<<" at position ("<<CellEta<<", "<<CellPhi<<")"<<endl;
		  h.numnegcells++;
		  hcal.numnegcells++;

		  h.nadaNegOccMap->Fill(CellEta,CellPhi);
		  /////hcal.nadaNegOccMap->Fill(CellEta,CellPhi);

		  // Fill with -1*E to make plotting easier (large negative values appear as peaks rather than troughs, etc.)
		  h.nadaNegEnergyMap->Fill(CellEta,CellPhi,-1*cellenergy);
		  /////hcal.nadaNegEnergyMap->Fill(CellEta,CellPhi,-1*cellenergy);

		  // Fill individual depth histograms
		  if (h.makeDiagnostics)
		    {
		      h.nadaNegOccMap_depth[CellDepth-1]->Fill(CellEta,CellPhi);
		      h.nadaNegEnergyMap_depth[CellDepth-1]->Fill(CellEta,CellPhi,
								  -1*cellenergy);
		      
		      /////hcal.nadaNegOccMap_depth[CellDepth-1]->Fill(CellEta,CellPhi);
			/*hcal.nadaNegEnergyMap_depth[CellDepth-1]->Fill(CellEta,CellPhi,
			  -1*cellenergy);
			*//////
		    } // if (h.makeDiagnostics)

		} // cellenergy < negative cutoff

	      // Case 1b:  E>maximum
	      else
		{
		  h.problemHotCells->Fill(CellEta,CellPhi);
		  hcal.problemHotCells->Fill(CellEta,CellPhi);
		  h.problemHotCells_depth[CellDepth-1]->Fill(CellEta,CellPhi);
		  /////hcal.problemHotCells_depth[CellDepth-1]->Fill(CellEta,CellPhi);

		  h.numhotcells++;
		  hcal.numhotcells++;
		  h.nadaOccMap->Fill(CellEta,CellPhi);
		  if (h.fVerbosity==2) cout <<"<HcalHotCellMonitor:nadaCheck> NADA ENERGY > MAX FOR ("<<CellEta<<","<<CellPhi<<"):  "<<cellenergy<<" GeV"<<endl;
		  h.nadaEnergyMap->Fill(CellEta,CellPhi,cellenergy);
		  /////hcal.nadaOccMap->Fill(CellEta,CellPhi);
		  
		    /////hcal.nadaEnergyMap->Fill(CellEta,CellPhi,cellenergy);
		} // else
	      // Cells marked as hot; no need to complete remaining code
	      continue;
	      
	    } // cell < negative cutoff or > maximum

	  // -------------------------------
	  // Case 2:  Energy is > negative cutoff, but less than minimum threshold -- skip the cell
	  
	  // Comment this line out if you want to plot distribution of all cells vs surroundings
	  else if (cellenergy<=h.nadaEnergyCandCut0)
	    continue;
	  
	  // -------------------------------
	  // Case 3:  Set thresholds according to input variables

	  // Case 3A:  If Cut0<E<Cut1, set thresholds based on default values
	  else if (cellenergy>h.nadaEnergyCandCut0 && cellenergy<h.nadaEnergyCandCut1) 
	    {
	      ECubeCut=h.nadaEnergyCubeCut;
	      ECellCut=h.nadaEnergyCellCut;
	    }
	  
	  // Case 3A: Cut1<=E<=Cut2, set thresholds based on fraction of energy in candidate cell
	  else if (cellenergy>h.nadaEnergyCandCut1 && cellenergy<h.nadaEnergyCandCut2) 
	    {
	      ECubeCut=h.nadaEnergyCubeFrac*cellenergy;
	      ECellCut=h.nadaEnergyCellFrac*cellenergy;
	    }
	  
	  // Form cube of nearest neighbor cells around CellIter

	  if (h.fVerbosity==2) cout <<"****** Candidate Cell Energy: "<<cellenergy<<endl;
	  typename Hits::const_iterator NeighborIter;

	  if (cubeSize<=0) continue; // no NADA cells can be found if the number of neighboring cells is zero

	  int etaFactor=1;  // correct for eta regions where phi segmentation is > 5 degrees/cell
	  int dPhi;
	  int NeighborEta;

	  int cubeComp=-1; // count up number of cells composing cube -- start at -1 because original cell is subtracted

	  int temp_cell_phi = CellPhi; // temporary variable for dealing with neighbors at boundaries between different phi segmentations
	  for ( NeighborIter = hits.begin();NeighborIter!=hits.end();++NeighborIter)
	    // Form cube centered on CellIter.  This needs to be looked at more carefully to deal with boundary conditions.  Should Ecube constraints change at the boundaries?
	    // Also need to deal with regions where phi segmentation changes
	    {
	      if (vetosize>0 && vetoCell(NeighborIter->id(),h.vetoCells)) continue; 
	      HcalDetId NeighborId(NeighborIter->detid().rawId());
	      if (NeighborId.subdet()!=h.type) continue;

	      NeighborEta=NeighborId.ieta();
	      // etafactor works to expand box size in regions where detectors cover more than 5 degrees in phi
	    
	      if (abs(NeighborId.depth()-CellDepth)>h.nadaMaxDeltaDepth) continue;
	      if (abs(NeighborId.ieta()-CellEta)>h.nadaMaxDeltaEta) continue;
	      etaFactor = 1+(abs(NeighborEta)>20)+2*(abs(NeighborEta)>39);

	      // Deal with changes in segmentation at eta boundaries
	      if (abs(CellEta)==20 && abs(NeighborEta)==21)
		{
		  temp_cell_phi-=(temp_cell_phi%2); // odd cells treated as even:
		}
	      else if (abs(CellEta)==39 && abs(NeighborEta)==40)
	      {
		temp_cell_phi+=(temp_cell_phi%4==3);
	      }

	      dPhi = (abs(NeighborId.iphi()-temp_cell_phi));
	      if (dPhi>36)
		dPhi=72-dPhi;
	      if (dPhi>=h.nadaMaxDeltaPhi*(1+etaFactor)) continue;
	      cubeComp++;

	      if (h.fVerbosity==2) cout <<"\t Neighbor energy = "<<NeighborIter->energy()<< "  "<<NeighborId<<endl;	  
	      if (NeighborIter->energy()>ECellCut)
		{
		  if (h.fVerbosity==2) cout <<"\t     ABOVE ENERGY CUT!"<<endl;

		  Ecube+=NeighborIter->energy();
		  if (h.fVerbosity==2) cout <<"\t\t Cube energy = "<<Ecube<<endl;
		}
	    } // for (cell_iter NeighborIter = c.begin()...)
	  
	  //Remove energy due to _cell
	  Ecube -=cellenergy;
	  
	  if (h.makeDiagnostics)
	    {
	      h.diagnostic[0]->Fill(cellenergy,Ecube);
	      /////hcal.diagnostic[0]->Fill(cellenergy,Ecube);


	      // Diagnostic plot of cell energy vs cube energy
	      h.EnergyVsNADAcube->Fill(Ecube, cellenergy);
	      /////hcal.EnergyVsNADAcube->Fill(Ecube, cellenergy);
	    }

	  if (h.fVerbosity==2) 
	    {
	      cout <<"\t\t\t\t Final Cube energy = "<<Ecube<<endl;
	      cout <<"\t\t\t\t ENERGY CUBE CUT = "<<ECubeCut<<endl;
	    }

	  // Compare cube energy to cube cut
	  // scale cube cut to handle cells at subdetector boundaries

	  if (h.fVerbosity && Ecube <=(ECubeCut*cubeComp/cubeSize))
	    {
	      cout <<"NADA Hot Cell found!"<<endl;
	      cout <<"\t NADA Ecube energy: "<<Ecube<<endl;
	      cout <<"\t NADA Ecell energy: "<<cellenergy<<endl;
	      cout <<"\t NADA Cell position: "<<id<<endl;
	    }
	  
	  // Hot cells found -- Identify hot cells by value of Ecube
	  if (Ecube <= ECubeCut*cubeComp/cubeSize)
	    {   
	      h.problemHotCells->Fill(CellEta,CellPhi);
	      hcal.problemHotCells->Fill(CellEta,CellPhi);
	      h.problemHotCells_depth[CellDepth-1]->Fill(CellEta,CellPhi);
	      /////hcal.problemHotCells_depth[CellDepth-1]->Fill(CellEta,CellPhi);
	      
	      if (h.makeDiagnostics)
		h.diagnostic[1]->Fill(CellDepth, 1.0*cubeComp/cubeSize);
	      if (h.fVerbosity==2) cout <<"Found NADA hot cell in "<<h.name.c_str()<<":  Ecube energy = "<<Ecube<<endl;
	      h.numhotcells++;
	      hcal.numhotcells++;
	      h.nadaOccMap->Fill(CellEta,CellPhi);
	      h.nadaEnergyMap->Fill(CellEta,CellPhi,cellenergy);
	      /////hcal.nadaOccMap->Fill(CellEta,CellPhi);
		/////hcal.nadaEnergyMap->Fill(CellEta,CellPhi,cellenergy);
	      // Fill histograms for each depth level of hcal
	      if (h.makeDiagnostics)
		{
		  h.nadaOccMap_depth[CellDepth-1]->Fill(CellEta,CellPhi);
		  h.nadaEnergyMap_depth[CellDepth-1]->Fill(CellEta,CellPhi,cellenergy);
		  /////hcal.nadaOccMap_depth[CellDepth-1]->Fill(CellEta,CellPhi);
		    /////hcal.nadaEnergyMap_depth[CellDepth-1]->Fill(CellEta,CellPhi,cellenergy);
		  
		  h.HOT_EnergyVsNADAcube->Fill(Ecube, cellenergy);
		  /////hcal.HOT_EnergyVsNADAcube->Fill(Ecube, cellenergy);
		} // if (h.makeDiagnostics)
	    } // if (Ecube <=EcubeCut*...)
	} //for (CellIter=c.begin(); CellIter!=c.end(); CellIter++)
      if (h.fVerbosity) cout <<"Filling "<<h.name.c_str()<<" NADA NumHotCell histo"<<endl;
      h.nadaNumHotCells->Fill(h.numhotcells);
      h.nadaNumNegCells->Fill(h.numnegcells);

    } // if (hits.size()>0)
    

  return;

  } //void nadaCheck
  
} // namespace HcalHotCellCheck


///////////////////////////////////////////////////////////////

// Default constructor
HcalHotCellMonitor::HcalHotCellMonitor() 
{
  ievt_=0;
}

// Destructor
HcalHotCellMonitor::~HcalHotCellMonitor() 
{
}

void HcalHotCellMonitor::reset(){}

void HcalHotCellMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe)
{
  HcalBaseMonitor::setup(ps,dbe); // set up base class

  baseFolder_ = rootFolder_+"HotCellMonitor";
  
  // Global bool for disabling NADA tests (slow, not very useful in cosmics runs)
  usenada_=ps.getUntrackedParameter<bool>("useNADA", true);

  // All subdetector values will be set to hcalHists values, unless 
  // explicitly stated otherwise in .cfi file
  hcalHists.etaMax=ps.getUntrackedParameter<double>("MaxEta", 29.5);
  hcalHists.etaMin=ps.getUntrackedParameter<double>("MinEta", -29.5);
  hcalHists.phiMax = ps.getUntrackedParameter<double>("MaxPhi", 73);
  hcalHists.phiMin = ps.getUntrackedParameter<double>("MinPhi", 0);

  if (hcalHists.etaMax<hcalHists.etaMin && fVerbosity)
    cout <<"<HcalHotCellMonitor> WARNING IN hcalhists!  etaMax is less than etaMin!  Swapping max and min!"<<endl;
  hcalHists.etaBins=(int)(fabs(hcalHists.etaMax-hcalHists.etaMin));
  if (hcalHists.phiMax<hcalHists.phiMin && fVerbosity)
    cout <<"<HcalHotCellMonitor> WARNING IN hcalhists!  phiMax is less than phiMin!  Swapping max and min!"<<endl;
  hcalHists.phiBins=(int)(fabs(hcalHists.phiMax-hcalHists.phiMin));
  if (fVerbosity) 
    {
      cout << "HotCell eta min/max set to " << hcalHists.etaMin << "/" << hcalHists.etaMax << endl;
      cout << "HotCell phi min/max set to " << hcalHists.phiMin << "/" << hcalHists.phiMax << endl;
    }

  // Pedestals in femtoCoulombs, rather than ADC counts
  doFCpeds_ = ps.getUntrackedParameter<bool>("PedestalsInFC", false);

  // Get thresholds to use for checking hot cell.  First threshold is used for ID'ing hot cells; remainder provide expert-level info
  hcalHists.thresholds = ps.getUntrackedParameter<vector <double> >("thresholds");

  // NADA values:
  /*  <=CandCut0 :  cells below this cut aren't considered hot
      CandCut0 < cell < CandCut1:  neighbors compared to fixed value (neighbors < X GeV means cell is hot) 
      CandCut1 <= cell < CandCut2:  neighbors compared to fraction of cell's energy
      cell >= CandCut2:  cell considered hot
  */

  hcalHists.nadaEnergyCandCut0 = ps.getUntrackedParameter<double>("NADA_Ecand_cut0",1.);
  hcalHists.nadaEnergyCandCut1 = ps.getUntrackedParameter<double>("NADA_Ecand_cut1",5.);
  hcalHists.nadaEnergyCandCut2 = ps.getUntrackedParameter<double>("NADA_Ecand_cut2",500.);
  hcalHists.nadaEnergyCubeCut = ps.getUntrackedParameter<double>("NADA_Ecube_cut",.1);
  hcalHists.nadaEnergyCellCut = ps.getUntrackedParameter<double>("NADA_Ecell_cut",.1);

  // Changed negative cut from D0 default of -1 GeV to -1.5 GeV based on CMS run 24934
  hcalHists.nadaNegCandCut = ps.getUntrackedParameter<double>("NADA_NegCand_cut",-1.5);
  hcalHists.nadaEnergyCubeFrac = ps.getUntrackedParameter<double>("NADA_Ecube_frac",0.02);
  hcalHists.nadaEnergyCellFrac = ps.getUntrackedParameter<double>("NADA_Ecell_frac",0.02);
  hcalHists.nadaMaxDeltaDepth = ps.getUntrackedParameter<int>("NADA_maxdepth",0);
  hcalHists.nadaMaxDeltaEta = ps.getUntrackedParameter<int>("NADA_maxeta",1);
  hcalHists.nadaMaxDeltaPhi = ps.getUntrackedParameter<int>("NADA_maxphi",1);

  // If  (digi-pedestal)/ped_Width > hotDigiSigma, cell is considered hot
  hcalHists.hotDigiSigma=ps.getUntrackedParameter<double>("HotCellDigiSigma",3.);
  hcalHists.makeDiagnostics=ps.getUntrackedParameter<bool>("MakeHotCellDiagnosticPlots",makeDiagnostics);
  
  hcalHists.name="HCAL";
  hcalHists.type=10;
  hcalHists.vetoCells=hotCells_;
  hcalHists.subdetOn=true;
  hcalHists.fVerbosity=fVerbosity;
  hcalHists.numhotcells=0;
  hcalHists.numnegcells=0;
  hcalHists.check=true;
  setupHists(hcalHists,dbe);
  // ID which subdetectors should be checked

  hbHists.check=ps.getUntrackedParameter<bool>("checkHB","true");
  if (hbHists.check)
    {
      setupVals(hbHists,1,hcalHists,ps);
      setupHists(hbHists,dbe);
    }
  heHists.check=ps.getUntrackedParameter<bool>("checkHE","true");
  if (heHists.check)
    {
      setupVals(heHists,2,hcalHists,ps);
      setupHists(heHists,dbe);
    }
  hoHists.check=ps.getUntrackedParameter<bool>("checkHO","true");
  if (hoHists.check)
    {
      setupVals(hoHists,3,hcalHists,ps);
      setupHists(hoHists,dbe);
    }
  hfHists.check=ps.getUntrackedParameter<bool>("checkHF","true");
  if (hfHists.check)
    {
      setupVals(hfHists,4,hcalHists,ps);
      setupHists(hfHists,dbe);
    }


  ievt_=0;
  
  if ( m_dbe !=0 ) {    

    m_dbe->setCurrentFolder(baseFolder_);

    meEVT_ = m_dbe->bookInt("HotCell Task Event Number");    
    meEVT_->Fill(ievt_);
  }

  return;
}

void HcalHotCellMonitor::setupVals(HotCellHists& h,int type,HotCellHists& base, const edm::ParameterSet& ps)
{
  // All subdetector values will be set to hcalHists values, unless 
  // explicitly stated otherwise in .cfi file

  if (!h.check)
    return;

  h.type=type;
  if (h.type==1)
    h.name="HB";
  else if (h.type==2)
    h.name="HE";
  else if (h.type==3)
    h.name="HO";
  else if (h.type==4)
    h.name="HF";
  else if (h.type==10)
    h.name="HCAL";
  else
    h.name="UNKNOWN";

  h.subdetOn=true;
  h.vetoCells=hotCells_;

  h.enS=-1000.;
  h.numhotcells=0;
  h.numnegcells=0;


  // Allow for each subdetector to set its own values, but default to base values if not specified
  char tag[256];
  sprintf(tag,"%sMaxEta",h.name.c_str());
  h.etaMax=ps.getUntrackedParameter<double>(tag, base.etaMax);
  sprintf(tag,"%sMinEta",h.name.c_str());
  h.etaMin=ps.getUntrackedParameter<double>(tag, base.etaMin);
  sprintf(tag,"%sMaxPhi",h.name.c_str());
  h.phiMax = ps.getUntrackedParameter<double>(tag,base.phiMax);
  sprintf(tag,"%sMinPhi",h.name.c_str());
  h.phiMin = ps.getUntrackedParameter<double>(tag,base.phiMin);
  // Allow for comments to be restricted to a single subdetector
  sprintf(tag,"%sdebug",h.name.c_str());
  h.fVerbosity=ps.getUntrackedParameter<int>(tag,base.fVerbosity);
  sprintf(tag,"%sHotCellDigiSigma",h.name.c_str());
  h.hotDigiSigma=ps.getUntrackedParameter<double>(tag,base.hotDigiSigma);
  sprintf(tag,"%sMakeHotCellDiagnosticPlots",h.name.c_str());
  h.makeDiagnostics=ps.getUntrackedParameter<double>(tag,base.makeDiagnostics);


  if (h.etaMax<h.etaMin && h.fVerbosity)
    cout <<"<HcalHotCellMonitor> WARNING IN setupVals for "<<h.name.c_str()<<"!  etaMax is less than etaMin!  Swapping max and min!"<<endl;
  h.etaBins=(int)(fabs(h.etaMax-h.etaMin));
  if (h.phiMax<h.phiMin && h.fVerbosity)
    cout <<"<HcalHotCellMonitor> WARNING IN setupVals for "<<h.name.c_str()<<"!  phiMax is less than phiMin!  Swapping max and min!"<<endl;
  h.phiBins=(int)(fabs(h.phiMax-h.phiMin));
  if (h.fVerbosity) 
    {
      cout << h.name.c_str()<<"HotCell eta min/max set to " << h.etaMin << "/" << h.etaMax << endl;
      cout << h.name.c_str()<<"HotCell phi min/max set to " << h.phiMin << "/" << h.phiMax << endl;
    }

  sprintf(tag,"%sthresholds",h.name.c_str());
  h.thresholds = ps.getUntrackedParameter<vector <double> >(tag,base.thresholds);
  sprintf(tag,"%sNADA_Ecand_cut0",h.name.c_str());
  h.nadaEnergyCandCut0 = ps.getUntrackedParameter<double>(tag,base.nadaEnergyCandCut0);
  sprintf(tag,"%sNADA_Ecand_cut1",h.name.c_str());
  h.nadaEnergyCandCut1 = ps.getUntrackedParameter<double>(tag,base.nadaEnergyCandCut1);
  sprintf(tag,"%sNADA_Ecand_cut2",h.name.c_str());
  h.nadaEnergyCandCut2 = ps.getUntrackedParameter<double>(tag,base.nadaEnergyCandCut2);
  sprintf(tag,"%sNADA_Ecube_cut",h.name.c_str());
  h.nadaEnergyCubeCut = ps.getUntrackedParameter<double>(tag,base.nadaEnergyCubeCut);
    sprintf(tag,"%sNADA_Ecell_cut",h.name.c_str());
  h.nadaEnergyCellCut = ps.getUntrackedParameter<double>(tag,base.nadaEnergyCellCut);
  // Changed negative cut from D0 default of -1 GeV to -1.5 GeV based on CMS run 24934
  sprintf(tag,"%sNADA_NegCand_cut",h.name.c_str());
  h.nadaNegCandCut = ps.getUntrackedParameter<double>(tag,base.nadaNegCandCut);
  sprintf(tag,"%sNADA_Ecube_frac",h.name.c_str());
  h.nadaEnergyCubeFrac = ps.getUntrackedParameter<double>(tag,base.nadaEnergyCubeFrac);
  sprintf(tag,"%sNADA_Ecell_frac",h.name.c_str());
  h.nadaEnergyCellFrac = ps.getUntrackedParameter<double>(tag,base.nadaEnergyCellFrac);
  sprintf(tag,"%sNADA_maxdepth",h.name.c_str());
  h.nadaMaxDeltaDepth = ps.getUntrackedParameter<int>(tag,base.nadaMaxDeltaDepth);
  sprintf(tag,"%sNADA_maxdeta",h.name.c_str());
  h.nadaMaxDeltaEta = ps.getUntrackedParameter<int>(tag,base.nadaMaxDeltaEta);
  sprintf(tag,"%sNADA_maxphi",h.name.c_str());
  h.nadaMaxDeltaPhi = ps.getUntrackedParameter<int>(tag,base.nadaMaxDeltaPhi);
  if (h.fVerbosity)
    { 
      cout <<"NADA parameters for subdetector "<<h.name.c_str()<<":"<<endl;
      cout <<"\tEcand cut #0 = "<<h.nadaEnergyCandCut0<<endl;
      cout <<"\tEcand cut #1 = "<<h.nadaEnergyCandCut1<<endl;
      cout <<"\tEcand cut #2 = "<<h.nadaEnergyCandCut2<<endl;
             
      cout <<"\tEcube cut = "<<h.nadaEnergyCubeCut<<endl;
      cout <<"\tEcell cut = "<<h.nadaEnergyCellCut<<endl;
      cout <<"\tNegCand cut = "<<h.nadaNegCandCut<<endl;
      cout <<"\tEcube_frac = "<<h.nadaEnergyCubeFrac<<endl;
      cout <<"\tEcell_frac = "<<h.nadaEnergyCellFrac<<endl;
      cout <<"\tMax delta depth = "<<h.nadaMaxDeltaDepth<<endl;
      cout <<"\tMax delta phi = "<<h.nadaMaxDeltaPhi<<endl;
      cout <<"\tMax delta eta = "<<h.nadaMaxDeltaEta<<endl;
    }
  return;
}

void HcalHotCellMonitor::setupHists(HotCellHists& h, DQMStore* dbe)

{
  if (!h.check)
    return;

  string subdet = h.name;
  m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str());
  
  // Create histogram for showing all hot cells in eta-phi
  h.problemHotCells = m_dbe->book2D(subdet+"ProblemHotCells", 
				    subdet+" Hot Cell rate for potentially bad cells",
				    h.etaBins,h.etaMin,h.etaMax,
				    h.phiBins,h.phiMin,h.phiMax);

  std::stringstream histname;
  std::stringstream histtitle;
  for (int depth=0;depth<4;++depth)
    {
      // Even though they are individual "Depth" histograms, put the 
      // ProblemHotCells_depth histos into expertPlots subdirectory
      m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str()+"/expertPlots");
      histname.str("");
      histtitle.str("");
      histname<<subdet+"ProblemHotCells_depth"<<depth+1;
      histtitle<<subdet+" Hot Cell rate for potentially bad cells (depth "<<depth+1<<")";
      h.problemHotCells_depth[depth]=(m_dbe->book2D(histname.str().c_str(),histtitle.str().c_str(),
						    h.etaBins,h.etaMin,h.etaMax,
						    h.phiBins,h.phiMin,h.phiMax));
    } // for (int depth=0;...)

  // Histograms for hottest cell in subdetector
  h.maxCellOccMap  = m_dbe->book2D(subdet+"_OccupancyMap_MaxCell",subdet+" HotCell Occupancy Map, Max Cell",h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax);
  
  m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str()+"/expertPlots/MaxCell");
  h.maxCellEnergy =  m_dbe->book1D(subdet+"HotCellEnergyMaxCell",subdet+" HotCell Max. Cell Energy",2000,0,20);
  h.maxCellTime =  m_dbe->book1D(subdet+"HotCellTimeMaxCell",subdet+" HotCell Max. Cell Time",200,-50,300);
  h.maxCellID =  m_dbe->book1D(subdet+"HotCellIDMaxCell",subdet+" HotCell Max. Cell ID",36000,-18000,18000);
  h.maxCellEnergyMap  = m_dbe->book2D(subdet+"_HotCell_EnergyMap_MaxCell",subdet+" HotCell Energy Map, Max Cell",h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax);

  // Add axis labels
  h.maxCellEnergy->setAxisTitle("Energy",1);
  h.maxCellEnergy->setAxisTitle("Counts",2);
  h.maxCellTime->setAxisTitle("Time",1);
  h.maxCellTime->setAxisTitle("Counts",2);
  h.maxCellID->setAxisTitle("ID",1);
  h.maxCellID->setAxisTitle("Counts",2);
  h.maxCellOccMap->setAxisTitle("i#eta",1);
  h.maxCellOccMap->setAxisTitle("i#phi",2);
  h.maxCellEnergyMap->setAxisTitle("i#eta",1);
  h.maxCellEnergyMap->setAxisTitle("i#phi",2);

  // Histograms for thresholds
  //std::vector<MonitorElement*> thresholdval;

  for (int k=0;k<int(h.thresholds.size());++k)
    {
      if (h.name!="HCAL")
	{
	  m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str());
  	  std::stringstream threshval;
	  threshval<<subdet+"HotCellThreshold"<<k;
	  //cout <<"THRESHVAL = "<<threshval.str().c_str()<<"  "<<h.thresholds[k]<<endl;

	  MonitorElement* me;
	  me = m_dbe->bookFloat(threshval.str().c_str());
	  me->Fill(h.thresholds[k]);
	  threshval.str(""); // clear stringstream object
	}

      std::stringstream myoccname;
      myoccname<<subdet+"_OccupancyMap_HotCell_Threshold"<<k;
      //const char *occname=myoccname.str().c_str();
      std::stringstream myocctitle;
      if (h.name=="HCAL")
	myocctitle<<subdet+" Hot Cell Occupancy, Cells > Threshold #"<<k;
      else
	myocctitle<<subdet+" Hot Cell Occupancy, Cells > "<<h.thresholds[k]<<" GeV";
      if (k==0)
	m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str());
      else
	m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str()+"/expertPlots/Thresholds");
      h.threshOccMap.push_back(m_dbe->book2D(myoccname.str().c_str(),myocctitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));

      m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str()+"/expertPlots/Thresholds");
      std::stringstream myenergyname;
      myenergyname<<subdet+"_HotCell_EnergyMap_Thresh"<<k;
      std::stringstream myenergytitle;
      if (h.name=="HCAL")
	myenergytitle<<subdet+" Hot Cell Energy, Cells > Threshold #"<<k;
      else
	myenergytitle<<subdet+" Hot Cell Energy, Cells > "<<h.thresholds[k]<<" GeV";
      h.threshEnergyMap.push_back(m_dbe->book2D(myenergyname.str().c_str(),myenergytitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));
      h.threshOccMap[k]->setAxisTitle("i#eta",1);
      h.threshOccMap[k]->setAxisTitle("i#phi",2);
      h.threshEnergyMap[k]->setAxisTitle("i#eta",1);
      h.threshEnergyMap[k]->setAxisTitle("i#phi",2);


      // Depth histograms
      if (h.makeDiagnostics)
	{
	  std::vector<MonitorElement*> occDepthHist;
	  std::vector<MonitorElement*> enDepthHist;
	  
	  for (int l=1;l<5;++l)
	    {
	      std::stringstream depthFoldername;
	      depthFoldername<<baseFolder_+"/"+subdet.c_str()+"/Diagnostics/Depth"<<l;
	      m_dbe->setCurrentFolder(depthFoldername.str().c_str());
	      depthFoldername.str("");
	      std::stringstream occdepthname;
	      occdepthname<<subdet+"_OccupancyMap_HotCell_Threshold"<<k<<"Depth"<<l;
	      std::stringstream occdepthtitle;
	      if (h.name=="HCAL")
		occdepthtitle<<subdet+"Hot Cell Occupancy for Depth "<<l<<", Cells > Threshold #"<<k;
	      else
		occdepthtitle<<subdet+"Hot Cell Occupancy for Depth "<<l<<", Cells > "<<h.thresholds[k]<<" GeV";
	      occDepthHist.push_back(m_dbe->book2D(occdepthname.str().c_str(),occdepthtitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));
	      
	      std::stringstream endepthname;
	      endepthname<<subdet+"_HotCell_EnergyMap_Thresh"<<k<<"Depth"<<l;
	      std::stringstream endepthtitle;
	      endepthtitle<<subdet+"Hot Cell Energy for Depth "<<l<<", Cells > "<<h.thresholds[k]<<" GeV";
	      enDepthHist.push_back(m_dbe->book2D(endepthname.str().c_str(),endepthtitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));
	      
	      // l starts at 1; shift by 1 to start with histogram 0
	      occDepthHist[l-1]->setAxisTitle("i#eta",1);
	      occDepthHist[l-1]->setAxisTitle("i#phi",2);
	      enDepthHist[l-1]->setAxisTitle("i#eta",1);
	      enDepthHist[l-1]->setAxisTitle("i#phi",2);
	      
	    }
	  h.threshOccMap_depth.push_back(occDepthHist);
	  h.threshEnergyMap_depth.push_back(enDepthHist);
	  m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str());
	} // if (h.makeDiagnostics)	
    } // for (int k=0;k<h.thresholds.size();++k)

  // Check against Digi Pedestals
  histname.str("");
  histtitle.str("");
  histname<<subdet+"_OccupancyMap_HotCell_Digi";
  histtitle<<subdet+" Digi > "<<h.hotDigiSigma<<" #sigma above pedestal";
  m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str());
  h.abovePedSigma=m_dbe->book2D(histname.str().c_str(),histtitle.str().c_str(),  
				h.etaBins,h.etaMin,h.etaMax,
				h.phiBins,h.phiMin,h.phiMax);

  if (h.makeDiagnostics)
    {
      m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str()+"/Diagnostics/DigiPedestals");
      for (int sigma=0;sigma<=4;++sigma)
	{
	  histname.str("");
	  histtitle.str("");
	  histname<<subdet+"HotCellDigiPedestalSigma"<<sigma;
	  histtitle<<subdet+" Digi > "<<sigma<<" sigma above pedestal";
	  h.digiPedestalPlots.push_back(m_dbe->book2D(histname.str().c_str(),histtitle.str().c_str(),
						      h.etaBins,h.etaMin,h.etaMax,
						      h.phiBins,h.phiMin,h.phiMax));
	
	  // Skip individual depth plots for now
	  /*
	  std::vector<MonitorElement*> digiDepthHist;
	  for (int depth=0;depth<4;++depth)
	    {
	      histname.str("");
	      histtitle.str("");
	      histname<<subdet+"HotCellDigiPedestalSigma"<<sigma<<"Depth"<<depth+1;
	      histtitle<<subdet+" Digi > "<<sigma<<" #sigma above pedestal for Depth= "<<depth+1;
	      digiDepthHist.push_back(m_dbe->book2D(histname.str().c_str(),histtitle.str().c_str(),
						      h.etaBins,h.etaMin,h.etaMax,
						      h.phiBins,h.phiMin,h.phiMax));
	    } // for (int depth=0;...)
	  h.digiPedestalPlots_depth.push_back(digiDepthHist);
	  */
	} // for (int sigma=0;...)
    } // if (h.makeDiagnostics)

  // NADA algorithm
  m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str());
  h.nadaOccMap = m_dbe->book2D(subdet+"_OccupancyMap_NADA",subdet+" NADA Occupancy",h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax);

  m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str()+"/expertPlots/NADA");
  h.nadaEnergyMap = m_dbe->book2D(subdet+"nadaEnergyMap",subdet+" NADA Energy",h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax);
  
  h.nadaOccMap->setAxisTitle("i#eta",1);
  h.nadaOccMap->setAxisTitle("i#phi",2);
  h.nadaEnergyMap->setAxisTitle("i#eta",1);
  h.nadaEnergyMap->setAxisTitle("i#phi",2);



  h.nadaNumHotCells = m_dbe->book1D(subdet+"nadaNumHotCells",subdet+" # of NADA Hot Cells Per Event",1000,0,1000);
  h.nadaEnergy = m_dbe->book1D(subdet+"Energy",subdet+" Energy for all cells",1000,-10,90);
  h.nadaNumNegCells = m_dbe->book1D(subdet+"nadaNumNegCells",subdet+" # of NADA Negative-Energy Cells Per Event",1000,0,1000);
  
  h.nadaNumHotCells->setAxisTitle("# of NADA Hot Cells",1);
  h.nadaNumHotCells->setAxisTitle("# of Events",2);
  h.nadaEnergy->setAxisTitle("Energy",1);
  h.nadaEnergy->setAxisTitle("# of Events",2);
  h.nadaNumNegCells->setAxisTitle("# of NADA Negative-Energy Cells",1);
  h.nadaNumNegCells->setAxisTitle("# of Events",2);

  h.nadaNegOccMap = m_dbe->book2D(subdet+"nadaNegOccMap",subdet+" NADA Negative Energy Cell Occupancy",h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax);
  h.nadaNegEnergyMap = m_dbe->book2D(subdet+"nadaNegEnergyMap",subdet+" NADA Negative Cell Energy",h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax);

  h.nadaNegOccMap->setAxisTitle("i#eta",1);
  h.nadaNegOccMap->setAxisTitle("i#phi",2);
  h.nadaNegEnergyMap->setAxisTitle("i#eta",1);
  h.nadaNegEnergyMap->setAxisTitle("i#phi",2);

  if (h.makeDiagnostics)
    {
      for (int l=1;l<5;++l)
	{
	  
	  std::stringstream depthFoldername;
	  depthFoldername<<baseFolder_+"/"+subdet.c_str()+"/Diagnostics/Depth"<<l;
	  m_dbe->setCurrentFolder(depthFoldername.str().c_str());
	  depthFoldername.str("");
	  std::stringstream occdepthname;
	  occdepthname<<subdet+"nadaOccMap"<<"Depth"<<l;
	  std::stringstream occdepthtitle;
	  occdepthtitle<<subdet+" Nada Hot Cell Occupancy for Depth "<<l;
	  h.nadaOccMap_depth[l-1]=(m_dbe->book2D(occdepthname.str().c_str(),occdepthtitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));
	  
	  std::stringstream endepthname;
	  endepthname<<subdet+"nadaEnergyMap"<<"Depth"<<l;
	  std::stringstream endepthtitle;
	  endepthtitle<<subdet+"Nada Hot Cell Energy for Depth "<<l;
	  h.nadaEnergyMap_depth[l-1]=(m_dbe->book2D(endepthname.str().c_str(),endepthtitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));
	  
	  // l starts at 1; shift by 1 to start with histogram 0
	  h.nadaOccMap_depth[l-1]->setAxisTitle("i#eta",1);
	  h.nadaOccMap_depth[l-1]->setAxisTitle("i#phi",2);
	  h.nadaEnergyMap_depth[l-1]->setAxisTitle("i#eta",1);
	  h.nadaEnergyMap_depth[l-1]->setAxisTitle("i#phi",2);
	  
	  std::stringstream negoccdepthname;
	  negoccdepthname<<subdet+"nadaNegOccMap"<<"Depth"<<l;
	  std::stringstream negoccdepthtitle;
	  negoccdepthtitle<<subdet+" Nada Negative Cell Occupancy for Depth "<<l;
	  h.nadaNegOccMap_depth[l-1]=(m_dbe->book2D(negoccdepthname.str().c_str(),negoccdepthtitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));
	  std::stringstream negendepthname;
	  negendepthname<<subdet+"nadaNegEnergyMap"<<"Depth"<<l;
	  std::stringstream negendepthtitle;
	  endepthtitle<<subdet+"Nada Negative Cell Energy for Depth "<<l;
	  h.nadaNegEnergyMap_depth[l-1]=(m_dbe->book2D(negendepthname.str().c_str(),negendepthtitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));
	  
	  // l starts at 1; shift by 1 to start with histogram 0
	  h.nadaNegOccMap_depth[l-1]->setAxisTitle("i#eta",1);
	  h.nadaNegOccMap_depth[l-1]->setAxisTitle("i#phi",2);
	  h.nadaNegEnergyMap_depth[l-1]->setAxisTitle("i#eta",1);
	  h.nadaNegEnergyMap_depth[l-1]->setAxisTitle("i#phi",2);
	} // for (int l=1;l<5;++l)
    } // if (h.makeDiagnostics)

  m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str()); 
    
  // Diagnostic histograms

  if ( h.makeDiagnostics)
    {
      std::stringstream diagFoldername;
      diagFoldername<<baseFolder_+"/"+subdet.c_str()+"/Diagnostics";
      m_dbe->setCurrentFolder(diagFoldername.str().c_str());
      
      h.hotcellsigma=m_dbe->book1D(subdet+"diagnostic_pedestal","(value - pedestal)/sigma(pedestal)",100,-10,10);

      h.diagnostic.push_back(m_dbe->book2D(subdet+"diagnostic_NADA","NADA cube energy vs. NADA cell energy",200,0,20,200,0,20));
      h.diagnostic.push_back(m_dbe->book2D(subdet+"diagnostic_depth","Cube size/Nominal vs. depth",4,0,4,100,0,1.1));

      h.diagnostic[0]->setAxisTitle("NADA cell energy",1);
      h.diagnostic[0]->setAxisTitle("NADA cube energy",2);
      h.diagnostic[1]->setAxisTitle("NADA cube size/nominal",1);
      h.diagnostic[1]->setAxisTitle("NADA depth",2);
      
      h.RecHitEnergyDist=m_dbe->book1D(subdet+"RecHitEnergyDist",
				       "Energy Distribution of Rec Hits",
				       200,0,20);

      for (int depth=1;depth<=4;++depth)
	{
	  diagFoldername.str("");
	  diagFoldername<<baseFolder_+"/"+subdet.c_str()+"/Diagnostics/Depth"<<depth;
	  m_dbe->setCurrentFolder(diagFoldername.str().c_str());
	  diagFoldername.str("");
	  std::stringstream tempname;
	  tempname<<subdet+"RecHitEnergyDist_Depth"<<depth;
	  h.RecHitEnergyDist_depth[depth-1]=(m_dbe->book1D(tempname.str().c_str(), 
							   "Energy Distribution of Rec Hits",
							   200,0,20));
	  tempname.str(""); // resets tempname
	  tempname<<subdet+"PedestalValue_Depth"<<depth;
	  h.pedestalValues_depth[depth-1]=(m_dbe->book2D(tempname.str().c_str(),
						       "Pedestal value for each cell (need to scale by 1/# of events run!)",
						       h.etaBins,h.etaMin,h.etaMax,
						       h.phiBins,h.phiMin,h.phiMax));
	  tempname.str(""); // resets tempname
	  tempname<<subdet+"PedestalWidth_Depth"<<depth;
	  h.pedestalWidths_depth[depth-1]=(m_dbe->book2D(tempname.str().c_str(),
						       "Pedestal width for each cell (need to scale by 1/# of events run!)",
						       h.etaBins,h.etaMin,h.etaMax,
						       h.phiBins,h.phiMin,h.phiMax));
	  tempname.str(""); //resets tempname

	} // for (int depth =1 ; depth<=4; ++depth)

      diagFoldername<<baseFolder_+"/"+subdet.c_str()+"/Diagnostics";
      m_dbe->setCurrentFolder(diagFoldername.str().c_str());
      h.DigiEnergyDist=m_dbe->book1D(subdet+"DigiEnergyDist","Digi Energy/#sigma_{pedestal}",100,-10,10);
      h.EnergyVsNADAcube=m_dbe->book2D(subdet+"EnergyVsNADAcube","Cell energy vs surrounding cube energy",300,-10,20,300,-10,20);
      h.EnergyVsNADAcube->setAxisTitle("Cell Energy",2);
      h.EnergyVsNADAcube->setAxisTitle("Cube Energy",1);
      
      h.HOT_EnergyVsNADAcube=m_dbe->book2D(subdet+"HOT_EnergyVsNADAcube","Cell energy vs surrounding cube energy",300,-10,20,300,-10,20);
      h.HOT_EnergyVsNADAcube->setAxisTitle("Cell Energy",2);
      h.HOT_EnergyVsNADAcube->setAxisTitle("Cube Energy",1);
    } // if (h.makeDiagnostics)
  return;

} // void HcalHotCellMonitor::setupHists(HotCellHists& h, DQMStore* dbe)


void HcalHotCellMonitor::processEvent( const HBHERecHitCollection& hbHits,
				       const HORecHitCollection& hoHits, 
				       const HFRecHitCollection& hfHits,
				       const HBHEDigiCollection& hbhedigi,
				       const HODigiCollection& hodigi,
				       const HFDigiCollection& hfdigi,
				       const HcalDbService& cond)
{
  if(!m_dbe) 
    { 
      cout<<"HcalHotCellMonitor::processEvent   DQMStore not instantiated!!!"<<endl;  
      return; 
    }

  ievt_++;
  meEVT_->Fill(ievt_);

  // Loop over digis
  processEvent_digi(hbhedigi,hodigi,hfdigi,cond); // check for hot digis


  if (fVerbosity) cout <<"HcalHotCellMonitor::processEvent   Starting process"<<endl;

  // Reset overall hcalHists max cell energy to default values
  hcalHists.enS=-1000.;
  hcalHists.tS=0.;
  hcalHists.etaS=0, hcalHists.phiS=0, hcalHists.depthS=0;
  hcalHists.idS=0;
  hcalHists.numhotcells=0, hcalHists.numnegcells=0;

  if (showTiming)
    {
      cpu_timer.reset();
      cpu_timer.start();
    }

  HcalHotCellCheck::threshCheck(hbHits, hbHists, hcalHists);
  if (showTiming)
    {
      cpu_timer.stop();
      cout <<"TIMER:: HcalHotCell RECHIT Threshold HB -> "<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset();cpu_timer.start();
    }

  HcalHotCellCheck::threshCheck(hbHits, heHists, hcalHists);
  if (showTiming)
    {
      cpu_timer.stop();
      cout <<"TIMER:: HcalHotCell RECHIT Threshold HE -> "<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }
  
  HcalHotCellCheck::threshCheck(hoHits, hoHists, hcalHists);
  if (showTiming)
    {
      cpu_timer.stop();
      cout <<"TIMER:: HcalHotCell RECHIT Threshold HO -> "<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }
  HcalHotCellCheck::threshCheck(hfHits, hfHists, hcalHists);


  if (showTiming) 
    {
      cpu_timer.stop();
      cout <<"TIMER:: HcalHotCell RECHIT Threshold HF -> "<<cpu_timer.cpuTime()<<endl;
    }

  if (usenada_)
    {
      if (showTiming) 
	{
	  cpu_timer.reset();cpu_timer.start();
	}

      HcalHotCellCheck::nadaCheck(hbHits, hbHists, hcalHists);
      if (showTiming)
	{
	  cpu_timer.stop();
	  cout <<"TIMER:: HcalHotCell RECHIT NADA HB -> "<<cpu_timer.cpuTime()<<endl;
	  cpu_timer.reset(); cpu_timer.start();
	}
      HcalHotCellCheck::nadaCheck(hbHits, heHists, hcalHists);
      if (showTiming)
	{
	  cpu_timer.stop();
	  cout <<"TIMER:: HcalHotCell RECHIT NADA HE -> "<<cpu_timer.cpuTime()<<endl;
	  cpu_timer.reset(); cpu_timer.start();
	}
      HcalHotCellCheck::nadaCheck(hoHits, hoHists, hcalHists);
      if (showTiming)
	{
	  cpu_timer.stop();
	  cout <<"TIMER:: HcalHotCell RECHIT NADA HO-> "<<cpu_timer.cpuTime()<<endl;
	  cpu_timer.reset(); cpu_timer.start();
	}
      // HF nada check is much slower than other subdets -- investigate later!  -- JT, 30/06/08
      //HcalHotCellCheck::nadaCheck(hfHits, hfHists, hcalHists);
      if (showTiming)	
	{
	  cpu_timer.stop();
	  cout <<"TIMER:: HcalHotCell RECHIT NADA HF-> "<<cpu_timer.cpuTime()<<endl;
	}
    }

  // After checking over all subdetectors, fill hcalHist maximum histograms:


  // At some point, change this code so that we only fill every N events?
  if (hcalHists.enS>-1000.)
    {
      hcalHists.maxCellEnergy->Fill(hcalHists.enS);
      hcalHists.maxCellTime->Fill(hcalHists.tS);
      hcalHists.maxCellOccMap->Fill(hcalHists.etaS,hcalHists.phiS);
      hcalHists.maxCellEnergyMap->Fill(hcalHists.etaS,hcalHists.phiS,hcalHists.enS);
      hcalHists.maxCellID->Fill(hcalHists.idS);
    }
  hcalHists.nadaNumHotCells->Fill(hcalHists.numhotcells);
  hcalHists.nadaNumNegCells->Fill(hcalHists.numnegcells);


  return;
}


void HcalHotCellMonitor::processEvent_digi(const HBHEDigiCollection& hbhedigi,
					    const HODigiCollection& hodigi,
					    const HFDigiCollection& hfdigi,
					    const HcalDbService& cond)
{

  /*
    Call digi-based Hot Cell monitor code ( check ADC counts,
    compare readout values vs. pedestals).
  */


  if (fVerbosity) cout <<"HcalHotCellMonitor::processEvent_digi     Starting process"<<endl;

  // Loop over HBHE

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  try
    {
      for (HBHEDigiCollection::const_iterator j=hbhedigi.begin(); j!=hbhedigi.end(); ++j)
	{
	  const HBHEDataFrame digi = (const HBHEDataFrame)(*j);

	  // HB goes out to ieta=16; ieta=16 is shared with HE
	  if (abs(digi.id().ieta())<16 || (abs(digi.id().ieta())==16 && ((HcalSubdetector)(digi.id().subdet()) == HcalBarrel)))
	    HcalHotCellCheck::CheckDigi(digi,hbHists,hcalHists,
					cond,m_dbe,doFCpeds_);
	  else 
	    HcalHotCellCheck::CheckDigi(digi,heHists,hcalHists,
					cond,m_dbe,doFCpeds_);
	}
    }
  catch (range_error)
    {
      if(fVerbosity) cout <<"HcalHotCellMonitor::processEvent_digi   No HBHE Digis."<<endl;
    }

  if (showTiming)
    {
      cpu_timer.stop();
      cout <<" TIMER:: HcalHotCell DIGI HBHE-> "<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  // Loop over HO
  try
    {
      for (HODigiCollection::const_iterator j=hodigi.begin(); j!=hodigi.end(); ++j)
	{
	  const HODataFrame digi = (const HODataFrame)(*j);
	  HcalHotCellCheck::CheckDigi(digi,hoHists,hcalHists,
				      cond,m_dbe,doFCpeds_);
	}
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalHotCellMonitor::processEvent_digi   No HO Digis."<<endl;
    }

  if (showTiming)
    {
      cpu_timer.stop();
      cout <<"TIMER:: HcalHotCell DIGI HO-> "<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  // Loop over HF
  try
    {
      for (HFDigiCollection::const_iterator j=hfdigi.begin(); j!=hfdigi.end(); ++j)
	{
	  const HFDataFrame digi = (const HFDataFrame)(*j);
	  HcalHotCellCheck::CheckDigi(digi,hfHists,hcalHists,
				      cond,m_dbe,doFCpeds_);
	}
    }
  catch(...)
    {
      if(fVerbosity) cout <<"HcalHotCellMonitor::processEvent_digi   No HF Digis."<<endl;
    }
  
  if (showTiming)
    {
      cpu_timer.stop();
      cout <<"TIMER:  HcalHotCell DIGI HF-> "<<cpu_timer.cpuTime()<<endl;
    }

  return;

} // void HcalHotCellMonitor::processEvent_digi
