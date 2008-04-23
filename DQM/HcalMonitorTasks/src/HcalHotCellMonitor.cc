#include "DQM/HcalMonitorTasks/interface/HcalHotCellMonitor.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <map>
#include <math.h>

// Use for stringstream
#include <iostream>
#include <iomanip>

namespace hotcells
{
  // Make my own copy of vetoCell in hotcells namespace?
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
  }

  template<class Hits>
  void threshCheck(const Hits& hits, HotCellHists& h, HotCellHists& hcal)
  {
    // Initialize values of max-energy cell 
    h.enS=-1000., h.tS=0., h.etaS=0, h.phiS=0, h.depthS=0;
    h.idS=0;

    if (h.fVerbosity)
      cout <<"<HcalHotCellMonitor> Looping over HCAL subdetector "<<h.name.c_str()<<endl; 
    
    if(hits.size()>0)
      {
	// Loop over all hits
	typename Hits::const_iterator _cell;
	for (_cell=hits.begin(); 
	     _cell!=hits.end(); 
	     ++_cell) 
	  { 
	    // Check that subdetector region is correct
	    if ((int)(_cell->id().subdet())!=h.type) continue;
	    
	    // Skip vetoed cells
	    // vetoCell(_cell->id() not available in this namespace.  Make my own copy of vetoCell?


	    if(h.vetoCells.size()>0 && vetoCell(_cell->id(),h.vetoCells))
	      {
		if (h.fVerbosity) cout <<"Vetoed cell with id = "<<_cell->id()<<endl;
		continue;
	      }


	    double cellenergy=_cell->energy();
	    int celldepth = _cell->id().depth();
	    
	    for (int k=0;k<int(h.thresholds.size());++k)
	      {
		if (cellenergy>h.thresholds[k])
		  {
		    if (h.threshOccMap[k]!=0)
		      {
			h.threshOccMap[k]->Fill(_cell->id().ieta(),
						_cell->id().iphi());
			hcal.threshOccMap[k]->Fill(_cell->id().ieta(),
						   _cell->id().iphi());
		      }
		    if (h.threshEnergyMap[k]!=0)
		      {
			h.threshEnergyMap[k]->Fill(_cell->id().ieta(),
						   _cell->id().iphi(),
						   cellenergy);
			hcal.threshEnergyMap[k]->Fill(_cell->id().ieta(),
						      _cell->id().iphi(),
						      cellenergy);
		      }
		    // Fill histograms for individual layers
		    // (FIXME::  I think layer counting starts at 1, not 0.
		    //  double check!! -- 30 Nov 2007)
		    if (h.threshOccMapDepth[k][celldepth-1]!=0)
		      {
			h.threshOccMapDepth[k][celldepth-1]->Fill(_cell->id().ieta(),
								  _cell->id().iphi());
			hcal.threshOccMapDepth[k][celldepth-1]->Fill(_cell->id().ieta(),
								     _cell->id().iphi());
		      }
		    if (h.threshEnergyMapDepth[k][celldepth-1]!=0)
		      {
			h.threshEnergyMapDepth[k][celldepth-1]->Fill(_cell->id().ieta(),
								     _cell->id().iphi(),
								   cellenergy);
			hcal.threshEnergyMapDepth[k][celldepth-1]->Fill(_cell->id().ieta(),
									_cell->id().iphi(),
									cellenergy);
		      }
		  } // if (cellenergy>h.thresholds[k])
	      } // for (int k=0;k<int(h.thresholds.size()...)
	    
	    // Store info from highest-energy cell
	    if(cellenergy>h.enS)
	      {
		h.enS = cellenergy;
		h.tS = _cell->time();
		h.etaS = _cell->id().ieta();
		h.phiS = _cell->id().iphi();
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
    h.numhotcells=0;
    h.numnegcells=0;

    // Get nominal cube size 
    int cubeSize = (2*h.nadaMaxDeltaDepth+1)*(2*h.nadaMaxDeltaEta+1)*(2*h.nadaMaxDeltaPhi+1)-1;

    if (hits.size()>0)
    {
      
      typename Hits::const_iterator _cell;
      
      // Copying NADA algorithm from D0 Note 4057.
      // The implementation needs to be optimized -- double looping over iterators is not an efficient approach.

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
      for (_cell=hits.begin(); _cell!=hits.end(); ++_cell)
	{
	  if (_cell->id().subdet()!=h.type) continue;
	  if (vetosize>0 && vetoCell(_cell->id(),h.vetoCells)) continue;
	  
	  cellenergy=_cell->energy();

	  h.nadaEnergy->Fill(cellenergy);
	  hcal.nadaEnergy->Fill(cellenergy);
	  
	  // _cell points to the current hot cell candidate
	  Ecube=0; // reset Ecube energy counter
	  CellDepth=_cell->id().depth();
	  CellEta=_cell->id().ieta();
	  CellPhi=_cell->id().iphi();

	  if (h.fVerbosity) cout <<"<HcalHotCellMonitor:nadaCheck> Cell Energy = "<<cellenergy<<" at position ("<<CellEta<<", "<<CellPhi<<")"<<endl;

	  // --------------------------- 
	  // Case 1:  E< -1 GeV or E>500 GeV:  Each counts as hot cell
	  
	  if (cellenergy<h.nadaNegCandCut || cellenergy>h.nadaEnergyCandCut2)
	    {
	      // Case 1a:  E< negative cutoff
	      if (cellenergy<h.nadaNegCandCut) 
		{ 
		  if (h.fVerbosity) cout <<"<HcalHotCellMonitor:nadaCheck> WARNING:  NEGATIVE "<<h.name.c_str()<<" CELL ENERGY>  Energy = "<<cellenergy<<" at position ("<<CellEta<<", "<<CellPhi<<")"<<endl;
		  h.numnegcells++;
		  hcal.numnegcells++;

		  h.nadaNegOccMap->Fill(CellEta,CellPhi);
		  hcal.nadaNegOccMap->Fill(CellEta,CellPhi);

		  // Fill with -1*E to make plotting easier (large negative values appear as peaks rather than troughs, etc.)
		  h.nadaNegEnergyMap->Fill(CellEta,CellPhi,-1*cellenergy);
		  hcal.nadaNegEnergyMap->Fill(CellEta,CellPhi,-1*cellenergy);

		  // Fill individual depth histograms
		  h.nadaNegOccMapDepth[CellDepth-1]->Fill(CellEta,CellPhi);
		  h.nadaNegEnergyMapDepth[CellDepth-1]->Fill(CellEta,CellPhi,
							     -1*cellenergy);

		  hcal.nadaNegOccMapDepth[CellDepth-1]->Fill(CellEta,CellPhi);
		  hcal.nadaNegEnergyMapDepth[CellDepth-1]->Fill(CellEta,CellPhi,
								-1*cellenergy);
		} // cellenergy < negative cutoff

	      // Case 1b:  E>maximum
	      else
		{
		  h.numhotcells++;
		  hcal.numhotcells++;
		  h.nadaOccMap->Fill(CellEta,CellPhi);
		  if (h.fVerbosity) cout <<"<HcalHotCellMonitor:nadaCheck> NADA ENERGY > MAX FOR ("<<CellEta<<","<<CellPhi<<"):  "<<cellenergy<<" GeV"<<endl;
		  h.nadaEnergyMap->Fill(CellEta,CellPhi,cellenergy);
		  hcal.nadaOccMap->Fill(CellEta,CellPhi);
		  
		  hcal.nadaEnergyMap->Fill(CellEta,CellPhi,cellenergy);
		}
	      // Cells marked as hot; no need to complete remaining code
	      continue;
	      
	    } // cell < negative cutoff or > maximum

	  // -------------------------------
	  // Case 2:  Energy is > negative cutoff, but less than minimum threshold -- skip the cell
	  
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
	  
	  // Form cube of nearest neighbor cells around _cell

	  if (h.fVerbosity) cout <<"****** Candidate Cell Energy: "<<cellenergy<<endl;
	  typename Hits::const_iterator _neighbor;

	  if (cubeSize<=0) return; // no NADA cells can be found if the number of neighboring cells is zero

	  int etaFactor=1;  // correct for eta regions where phi segmentation is > 5 degrees/cell
	  int dPhi;
	  int NeighborEta;

	  int cubeComp=-1; // count up number of cells composing cube -- start at -1 because original cell is subtracted

	  for ( _neighbor = hits.begin();_neighbor!=hits.end();++_neighbor)
	    // Form cube centered on _cell.  This needs to be looked at more carefully to deal with boundary conditions.  Should Ecube constraints change at the boundaries?
	    // Also need to deal with regions where phi segmentation changes
	    {
	      if (vetosize>0 && vetoCell(_neighbor->id(),h.vetoCells)) continue; 
	      if (_neighbor->id().subdet()!=h.type) continue;

	      NeighborEta=_neighbor->id().ieta();
	      // etafactor works to expand box size in regions where detectors cover more than 5 degrees in phi
	    
	      if (abs(_neighbor->id().depth()-CellDepth)>h.nadaMaxDeltaDepth) continue;
	      if (abs(_neighbor->id().ieta()-CellEta)>h.nadaMaxDeltaEta) continue;
	      etaFactor = 1+(abs(NeighborEta)>20)+2*(abs(NeighborEta)>39);
	      dPhi = (abs(_neighbor->id().iphi()-CellPhi));
	      if (dPhi>36)
		dPhi=72-dPhi;
	      if (dPhi>h.nadaMaxDeltaPhi*etaFactor) continue;
	      cubeComp++;

	      if (h.fVerbosity) cout <<"\t Neighbor energy = "<<_neighbor->energy()<< "  "<<_neighbor->id()<<endl;	  
	      if (_neighbor->energy()>ECellCut)
		{
		  if (h.fVerbosity) cout <<"\t     ABOVE ENERGY CUT!"<<endl;

		  Ecube+=_neighbor->energy();
		  if (h.fVerbosity) cout <<"\t\t Cube energy = "<<Ecube<<endl;
		}
	    } // for (cell_iter _neighbor = c.begin()...)
	  
	  //Remove energy due to _cell
	  Ecube -=cellenergy;
	  
	  h.diagnostic[0]->Fill(cellenergy,Ecube);
	  hcal.diagnostic[0]->Fill(cellenergy,Ecube);
	  if (h.fVerbosity) 
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
	      cout <<"\t NADA Cell position: "<<_cell->id()<<endl;
	    }
	  
	  // Identify hot cells by value of Ecube
	  if (Ecube <= ECubeCut*cubeComp/cubeSize)
	    {   
	      h.diagnostic[1]->Fill(CellDepth, 1.0*cubeComp/cubeSize);
	      if (h.fVerbosity) cout <<"Found NADA hot cell in "<<h.name.c_str()<<":  Ecube energy = "<<Ecube<<endl;
	      h.numhotcells++;
	      hcal.numhotcells++;
	      h.nadaOccMap->Fill(CellEta,CellPhi);
	      h.nadaEnergyMap->Fill(CellEta,CellPhi,cellenergy);
	      hcal.nadaOccMap->Fill(CellEta,CellPhi);
	      hcal.nadaEnergyMap->Fill(CellEta,CellPhi,cellenergy);
	      // Fill histograms for each depth level of hcal
	      h.nadaOccMapDepth[CellDepth-1]->Fill(CellEta,CellPhi);
	      h.nadaEnergyMapDepth[CellDepth-1]->Fill(CellEta,CellPhi,cellenergy);
	      hcal.nadaOccMapDepth[CellDepth-1]->Fill(CellEta,CellPhi);
	      hcal.nadaEnergyMapDepth[CellDepth-1]->Fill(CellEta,CellPhi,cellenergy);
	    }
	} //for (_cell=c.begin(); _cell!=c.end(); _cell++)
      if (h.fVerbosity) cout <<"Filling "<<h.name.c_str()<<" NADA NumHotCell histo"<<endl;
      h.nadaNumHotCells->Fill(h.numhotcells);
      h.nadaNumNegCells->Fill(h.numnegcells);
      
    } // if hits.size()>0
  return;

  } //void nadaCheck
  
} // namespace hotcells


HcalHotCellMonitor::HcalHotCellMonitor() {
  ievt_=0;
}

HcalHotCellMonitor::~HcalHotCellMonitor() {
}
void HcalHotCellMonitor::reset(){}

void HcalHotCellMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  baseFolder_ = rootFolder_+"HotCellMonitor";

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
  hcalHists.thresholds = ps.getUntrackedParameter<vector <double> >("thresholds");
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
  hcalHists.name="HCAL";
  hcalHists.type=10;
  hcalHists.vetoCells=hotCells_;
  hcalHists.subdetOn=true;
  hcalHists.fVerbosity=fVerbosity;
  hcalHists.numhotcells=0;
  hcalHists.numnegcells=0;
  setupHists(hcalHists,dbe);

  // ID which subdetectors should be checked
  bool temp;
  temp=ps.getUntrackedParameter<bool>("checkHB","true");
  if (temp)
    {
      setupVals(hbHists,1,hcalHists,ps);
      setupHists(hbHists,dbe);
    }
  temp=ps.getUntrackedParameter<bool>("checkHE","true");
  if (temp)
    {
      setupVals(heHists,2,hcalHists,ps);
      setupHists(heHists,dbe);
    }
  temp=ps.getUntrackedParameter<bool>("checkHO","true");
  if (temp)
    {
      setupVals(hoHists,3,hcalHists,ps);
      setupHists(hoHists,dbe);
    }
  temp=ps.getUntrackedParameter<bool>("checkHF","true");
  if (temp)
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
  h.fVerbosity=ps.getUntrackedParameter<bool>(tag,fVerbosity);
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
  string subdet = h.name;
  m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str());
  
  // Histograms for hottest cell in subdetector
  h.maxCellEnergy =  m_dbe->book1D(subdet+"HotCellEnergyMaxCell",subdet+" HotCell Max. Cell Energy",2000,0,20);
  h.maxCellTime =  m_dbe->book1D(subdet+"HotCellTimeMaxCell",subdet+" HotCell Max. Cell Time",200,-50,300);
  h.maxCellID =  m_dbe->book1D(subdet+"HotCellIDMaxCell",subdet+" HotCell Max. Cell ID",36000,-18000,18000);
  
  h.maxCellOccMap  = m_dbe->book2D(subdet+"HotCellOccMapMaxCell",subdet+" HotCell Geo Occupancy Map, Max Cell",h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax);
  h.maxCellEnergyMap  = m_dbe->book2D(subdet+"HotCellEnergyMapMaxCell",subdet+" HotCell Geo Energy Map, Max Cell",h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax);

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
  std::vector<MonitorElement*> thresholdval;
  
  for (int k=0;k<int(h.thresholds.size());++k)
    {
      if (h.name!="HCAL")
	{
	  std::stringstream threshval;
	  threshval<<subdet+"Threshold"<<k;
	  thresholdval.push_back(m_dbe->bookFloat(threshval.str().c_str()));
	  thresholdval[k]->Fill(h.thresholds[k]);
	}

      std::stringstream myoccname;
      myoccname<<subdet+"HotCellOccMapThresh"<<k;
      //const char *occname=myoccname.str().c_str();
      std::stringstream myocctitle;
      if (h.name=="HCAL")
	myocctitle<<subdet+" Hot Cell Occupancy, Cells > Threshold #"<<k;
      else
	myocctitle<<subdet+" Hot Cell Occupancy, Cells > "<<h.thresholds[k]<<" GeV";
      h.threshOccMap.push_back(m_dbe->book2D(myoccname.str().c_str(),myocctitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));
      
      std::stringstream myenergyname;
      myenergyname<<subdet+"HotCellEnergyMapThresh"<<k;
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

      std::vector<MonitorElement*> occDepthHist;
      std::vector<MonitorElement*> enDepthHist;
      
      for (int l=1;l<5;++l)
	{
	  std::stringstream depthFoldername;
	  depthFoldername<<baseFolder_+"/"+subdet.c_str()+"/"+"Depth"<<l;
	  m_dbe->setCurrentFolder(depthFoldername.str().c_str());
	  std::stringstream occdepthname;
	  occdepthname<<subdet+"HotCellOccMapThresh"<<k<<"Depth"<<l;
	  std::stringstream occdepthtitle;
	  if (h.name=="HCAL")
	    occdepthtitle<<subdet+"Hot Cell Occupancy for Depth "<<l<<", Cells > Threshold #"<<k;
	  else
	    occdepthtitle<<subdet+"Hot Cell Occupancy for Depth "<<l<<", Cells > "<<h.thresholds[k]<<" GeV";
	  occDepthHist.push_back(m_dbe->book2D(occdepthname.str().c_str(),occdepthtitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));

	  std::stringstream endepthname;
	  endepthname<<subdet+"HotCellEnergyMapThresh"<<k<<"Depth"<<l;
	  std::stringstream endepthtitle;
	  endepthtitle<<subdet+"Hot Cell Energy for Depth "<<l<<", Cells > "<<h.thresholds[k]<<" GeV";
	  enDepthHist.push_back(m_dbe->book2D(endepthname.str().c_str(),endepthtitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));

	  // l starts at 1; shift by 1 to start with histogram 0
	  occDepthHist[l-1]->setAxisTitle("i#eta",1);
	  occDepthHist[l-1]->setAxisTitle("i#phi",2);
	  enDepthHist[l-1]->setAxisTitle("i#eta",1);
	  enDepthHist[l-1]->setAxisTitle("i#phi",2);

	}
      h.threshOccMapDepth.push_back(occDepthHist);
      h.threshEnergyMapDepth.push_back(enDepthHist);
      m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str());
    }
  
  h.nadaOccMap = m_dbe->book2D(subdet+"nadaOccMap",subdet+" NADA Occupancy",h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax);
  h.nadaEnergyMap = m_dbe->book2D(subdet+"nadaEnergyMap",subdet+" NADA Energy",h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax);
  
  h.nadaOccMap->setAxisTitle("i#eta",1);
  h.nadaOccMap->setAxisTitle("i#phi",2);
  h.nadaEnergyMap->setAxisTitle("i#eta",1);
  h.nadaEnergyMap->setAxisTitle("i#phi",2);



  h.nadaNumHotCells = m_dbe->book1D(subdet+"nadaNumHotCells",subdet+" # of NADA Hot Cells/Event",1000,0,1000);
  h.nadaTestPlot = m_dbe->book1D(subdet+"nadaTestcell",subdet+" Energy for test cell",1000,-10,90);
  h.nadaEnergy = m_dbe->book1D(subdet+"Energy",subdet+" Energy for all cells",1000,-10,90);
  h.nadaNumNegCells = m_dbe->book1D(subdet+"nadaNumNegCells",subdet+" # of NADA Negative-Energy Cells/Event",1000,0,1000);

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

  for (int l=1;l<5;++l)
    {

      std::stringstream depthFoldername;
      depthFoldername<<baseFolder_+"/"+subdet.c_str()+"/"+"Depth"<<l;
      m_dbe->setCurrentFolder(depthFoldername.str().c_str());
      std::stringstream occdepthname;
      occdepthname<<subdet+"nadaOccMap"<<"Depth"<<l;
      std::stringstream occdepthtitle;
      occdepthtitle<<subdet+" Nada Hot Cell Occupancy for Depth "<<l;
      h.nadaOccMapDepth.push_back(m_dbe->book2D(occdepthname.str().c_str(),occdepthtitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));
      
      std::stringstream endepthname;
      endepthname<<subdet+"nadaEnergyMap"<<"Depth"<<l;
      std::stringstream endepthtitle;
      endepthtitle<<subdet+"Nada Hot Cell Energy for Depth "<<l;
      h.nadaEnergyMapDepth.push_back(m_dbe->book2D(endepthname.str().c_str(),endepthtitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));

      // l starts at 1; shift by 1 to start with histogram 0
      h.nadaOccMapDepth[l-1]->setAxisTitle("i#eta",1);
      h.nadaOccMapDepth[l-1]->setAxisTitle("i#phi",2);
      h.nadaEnergyMapDepth[l-1]->setAxisTitle("i#eta",1);
      h.nadaEnergyMapDepth[l-1]->setAxisTitle("i#phi",2);

      std::stringstream negoccdepthname;
      negoccdepthname<<subdet+"nadaNegOccMap"<<"Depth"<<l;
      std::stringstream negoccdepthtitle;
      negoccdepthtitle<<subdet+" Nada Negative Cell Occupancy for Depth "<<l;
      h.nadaNegOccMapDepth.push_back(m_dbe->book2D(negoccdepthname.str().c_str(),negoccdepthtitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));
      std::stringstream negendepthname;
      negendepthname<<subdet+"nadaNegEnergyMap"<<"Depth"<<l;
      std::stringstream negendepthtitle;
      endepthtitle<<subdet+"Nada Negative Cell Energy for Depth "<<l;
      h.nadaNegEnergyMapDepth.push_back(m_dbe->book2D(negendepthname.str().c_str(),negendepthtitle.str().c_str(),h.etaBins,h.etaMin,h.etaMax,h.phiBins,h.phiMin,h.phiMax));

      // l starts at 1; shift by 1 to start with histogram 0
      h.nadaNegOccMapDepth[l-1]->setAxisTitle("i#eta",1);
      h.nadaNegOccMapDepth[l-1]->setAxisTitle("i#phi",2);
      h.nadaNegEnergyMapDepth[l-1]->setAxisTitle("i#eta",1);
      h.nadaNegEnergyMapDepth[l-1]->setAxisTitle("i#phi",2);

    }

  //m_dbe->setCurrentFolder(baseFolder_+"/"+subdet.c_str()); // uncomment this if adding more histograms below this point
    
  // Diagnostic histogram
  std::stringstream diagFoldername;
  diagFoldername<<baseFolder_+"/"+subdet.c_str()+"/"+"Diagnostics";
  m_dbe->setCurrentFolder(diagFoldername.str().c_str());
  h.diagnostic.push_back(m_dbe->book2D(subdet+"diagnostic_NADA","NADA cube energy vs. NADA cell energy",200,0,20,200,0,20));
  h.diagnostic.push_back(m_dbe->book2D(subdet+"diagnostic_depth","Cube size/Nominal vs. depth",4,0,4,100,0,1.1));

  h.diagnostic[0]->setAxisTitle("NADA cell energy",1);
  h.diagnostic[0]->setAxisTitle("NADA cube energy",2);
  h.diagnostic[1]->setAxisTitle("NADA cube size/nominal",1);
  h.diagnostic[1]->setAxisTitle("NADA depth",2);
  
  return;

} // setupHists



void HcalHotCellMonitor::processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits){

  if(!m_dbe) 
    { 
      cout<<"HcalHotCellMonitor::processEvent   DQMStore not instantiated!!!"<<endl;  
      return; 
    }

  ievt_++;
  meEVT_->Fill(ievt_);


  if (fVerbosity) cout <<"HcalHotCellMonitor::processEvent   Starting process"<<endl;

  // Reset overall hcalHists max cell energy to default values
  hcalHists.enS=-1000., hcalHists.tS=0.;
  hcalHists.etaS=0, hcalHists.phiS=0, hcalHists.depthS=0;
  hcalHists.idS=0;
  hcalHists.numhotcells=0, hcalHists.numnegcells=0;

  hotcells::threshCheck(hbHits, hbHists, hcalHists);
  hotcells::threshCheck(hbHits, heHists, hcalHists);
  hotcells::threshCheck(hoHits, hoHists, hcalHists);
  hotcells::threshCheck(hfHits, hfHists, hcalHists);

  hotcells::nadaCheck(hbHits, hbHists, hcalHists);
  hotcells::nadaCheck(hbHits, heHists, hcalHists);
  hotcells::nadaCheck(hoHits, hoHists, hcalHists);
  hotcells::nadaCheck(hfHits, hfHists, hcalHists);

  // After checking over all subdetectors, fill hcalHist maximum histograms:

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
