#ifndef DQM_HCALMONITORTASKS_HCALHOTCELLMONITOR_H
#define DQM_HCALMONITORTASKS_HCALHOTCELLMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include <map>

/** \class HcalHotCellMonitor
  *  
  * $Date: 2007/04/02 13:19:38 $
  * $Revision: 1.3 $
  * \author W. Fisher - FNAL
  */

struct HistList{
  MonitorElement* meOCC_MAP_GEO_Thr0;
  MonitorElement* meEN_MAP_GEO_Thr0;
  MonitorElement* meOCC_MAP_GEO_Thr1;
  MonitorElement* meEN_MAP_GEO_Thr1;
  MonitorElement* meOCC_MAP_GEO_Max;
  MonitorElement* meEN_MAP_GEO_Max;
  MonitorElement* meMAX_E;
  MonitorElement* meMAX_T;
  MonitorElement* meMAX_ID;
};

struct NADAHistList{
  MonitorElement* NADA_OCC_MAP;
  MonitorElement* NADA_EN_MAP;
  MonitorElement* NADA_NumHotCells;
  MonitorElement* NADA_testcell;
  MonitorElement* NADA_Energy;
  
};

class HcalHotCellMonitor: public HcalBaseMonitor {
public:
  HcalHotCellMonitor(); 
  ~HcalHotCellMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits);
  void clearME();


  template<class CellCollection> 
    void FindHotCell(const CellCollection& c, HistList& h, int hb=0)
  {
    /* This function finds hot cells from a collection of reco'd cells 'c'.
       It fills them into the appropriate histogram in HistList 'h'.
       If hb = 1, cells are required to come from the HB barrel detector.
       If hb = 2, cells are required to come from the HE endcap.
       Otherwise, no requirement is made on the cells (sorting by region
       is assumed to take place when the original collection 'c' is made).
    */

    if(c.size()>0)
      {
	// reset subdetector counters
	enS=0; 
	tS=0; 
	etaS=0; phiS=0;
	idS=-1;
	
	// Define CellCollection iterator type
	typedef typename CellCollection::const_iterator cell_iter;
	cell_iter _cell;

	// loop over all hits
	for (_cell=c.begin(); _cell!=c.end(); _cell++) 
	  { 
	
	    // hb==1:  loop over barrels only
	    if (hb==1 && (HcalSubdetector)(_cell->id().subdet())!=HcalBarrel) continue;
	    // hb==2:  loop over endcap only
	    else if (hb==2 && (HcalSubdetector)(_cell->id().subdet())!=HcalEndcap) continue;

	    // check whether cell is above lower threshold
	    if(_cell->energy()>occThresh0_)
	      {
		if(vetoCell(_cell->id())) continue; // skip vetoed cells
		h.meEN_MAP_GEO_Thr0->Fill(_cell->id().ieta(),_cell->id().iphi(),_cell->energy());
		h.meOCC_MAP_GEO_Thr0->Fill(_cell->id().ieta(),_cell->id().iphi());
		// check whether cell is above upper threshold
		if(_cell->energy()>occThresh1_)
		  {
		    h.meEN_MAP_GEO_Thr1->Fill(_cell->id().ieta(),_cell->id().iphi(),_cell->energy());
		    h.meOCC_MAP_GEO_Thr1->Fill(_cell->id().ieta(),_cell->id().iphi());
		  }
		// Compare cell energy to largest energy found thus far
		// in the subdetector.  
		if(_cell->energy()>enS)
		  {
		    enS = _cell->energy();
		    tS = _cell->time();
		    etaS = _cell->id().ieta();
		    phiS = _cell->id().iphi();
		    idS = 1000*etaS;
		    if(idS<0) idS -= (10*phiS+depth);
		    else idS += (10*phiS+depth);
		    
		    depth = _cell->id().depth();
		  }
	      } // if (_cell->energy()>occThresh0)
	  } // for (_cell=c.begin()...
	
	// Plot largest energy
	if(enS>0)
	  {
	    h.meMAX_E->Fill(enS);
	    h.meMAX_T->Fill(tS);
	    h.meOCC_MAP_GEO_Max->Fill(etaS,phiS);
	    h.meEN_MAP_GEO_Max->Fill(etaS,phiS,enS);
	    h.meMAX_ID->Fill(idS);
	  }
      } // if (c.size()>0)
    
    // Compare largest subdetector energy to largest 
    // overall energy.
    if(enS>enA)
      {
	enA = enS;
	tA = tS;
	etaA = etaS;
	phiA = phiS;
      }

    return;
  }


  template<class CellCollection>
    void NADAFinder(const CellCollection& c, NADAHistList& h, int hb=0)
  {
    if (c.size()>0)
      {
	// Define CellCollection iterator type
	typedef typename CellCollection::const_iterator cell_iter;
	cell_iter _cell;
	
	// Copying NADA algorithm from D0 Note 4057.
	// The implementation needs to be optimized -- double looping over iterators is not an efficient approach.
	float  Ecube=0;
	int numhotcells=0;
	int CellPhi=-1, CellEta=-1, CellDepth=-1;

	for (_cell=c.begin(); _cell!=c.end(); _cell++)
	  {
	    // hb==1:  loop over barrels only
	    if (hb==1 && (HcalSubdetector)(_cell->id().subdet())!=HcalBarrel) continue;
	    // hb==2:  loop over endcap only
	    else if (hb==2 && (HcalSubdetector)(_cell->id().subdet())!=HcalEndcap) continue;

	    if (hb==3)
	      h.NADA_Energy->Fill(_cell->energy()-HF_offsets[_cell->id().ieta()-29][(_cell->id().iphi()-1)/2][_cell->id().depth()]);
	    else
	      h.NADA_Energy->Fill(_cell->energy());
	    // _cell points to the current hot cell candidate
	    Ecube=0; // reset Ecube energy counter
	    
	    // Case 1:  E< -1 GeV or E>500 GeV
	    if (_cell->energy()<-25 || _cell->energy()>500)
	      {
		if (vetoCell(_cell->id()))continue;
		numhotcells++;
		h.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		if (hb==3)
		  h.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),_cell->energy()-HF_offsets[_cell->id().ieta()-29][(_cell->id().iphi()-1)/2][_cell->id().depth()]);
		else
		  h.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),_cell->energy());
	      
	      }
	    
	    // Case 2:  Sum around neighbors
	    else if (_cell->energy()>1) // minimum threshold energy of 1 GeV; make this mutable
	      {
		if (vetoCell(_cell->id())) continue;
		CellPhi = _cell->id().iphi();
		CellEta = _cell->id().ieta();
		CellDepth = _cell->id().depth();
		
		// Form cube of nearest neighbor cells around _cell
		
		for (cell_iter _neighbor = c.begin();_neighbor!=c.end();_neighbor++)
		  // Form cube centered on _cell.  This needs to be looked at more carefully to deal with boundary conditions.  Should Ecube constraints change at the boundaries?
		  {
		    if (vetoCell(_neighbor->id())) continue;
		    // An HE cell can be a neighbor of an HB cell, right?
		    // I think we want to allow these, though conditions may
		    // need to change for the overlaps.
		    /*
		    // hb==1:  loop over barrels only
		    if (hb==1 && (HcalSubdetector)(_neighbor->id().subdet())!=HcalBarrel) continue;
		    // hb==2:  loop over endcap only
		    else if (hb==2 && (HcalSubdetector)(_neighbor->id().subdet())!=HcalEndcap) continue;
		    */

		    if (abs(_neighbor->id().iphi()-CellPhi)<2 && (abs(_neighbor->id().ieta()-CellEta))%72<2 && abs(_neighbor->id().depth()-CellDepth)<2)
		      {
			if (_cell->energy()<=5 && _neighbor->energy()>0.1) // another threshold to be made mutable
			  Ecube+=_neighbor->energy();
			else if (_neighbor->energy()>0.02*_cell->energy())
			  Ecube+=_neighbor->energy();
		      }
		  } // for (cell_iter _neighbor = c.begin()...)
		
		// Remove energy due to _cell
		Ecube -=_cell->energy();
		
		// What if Ecube = zero here?  That should count as a hot cell, right?  (Only _cell was found to have energy > 1 GeV)
		// Identify hot cells by value of Ecube
		if ((_cell->energy()<=5 && Ecube < 0.1)||(_cell->energy()>5 && Ecube<0.020*_cell->energy()))// more mutable thresholds
		  {   
		    numhotcells++;
		    h.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		    if (hb==3)
		      h.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),_cell->energy()-HF_offsets[_cell->id().ieta()-29][(_cell->id().iphi()-1)/2][_cell->id().depth()]);
		    else
		      h.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),_cell->energy());
		  }
	      } // if _cell->energy()>1	  
	  } //for (_cell=c.begin(); _cell!=c.end(); _cell++)
	h.NADA_NumHotCells->Fill(numhotcells);
	hotcells_+=numhotcells;
      } // if c.size()>0
    return;
  }




private:  ///Monitoring elements

  int ievt_;
  double occThresh0_;
  double occThresh1_;

  // These are now accessed in multiple methods; make them class variables
  // (Add "_" suffix to names?)
  float enS, tS, etaS, phiS, idS;
  float enA, tA, etaA, phiA;
  int depth;
  int hotcells_;

  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;
  
  
  // HF offsets take form [eta][phi][depth]
  // eta : 29-41
  // phi:  1-71
  // depth: 1-2
  float HF_offsets[13][36][2];



  // Should we make this something like 
  // struct HistList{ ....}
  // HistList hbHists, heHists,...
  // Then we can call routines witha HistList argument?

  HistList hbHists,heHists,hfHists,hoHists;
  NADAHistList NADA_hbHists,NADA_heHists, NADA_hfHists, NADA_hoHists;


  MonitorElement* meOCC_MAP_L1;
  MonitorElement* meEN_MAP_L1;
  MonitorElement* meOCC_MAP_L2;
  MonitorElement* meEN_MAP_L2;
  MonitorElement* meOCC_MAP_L3;
  MonitorElement* meEN_MAP_L3;
  MonitorElement* meOCC_MAP_L4;
  MonitorElement* meEN_MAP_L4;

  MonitorElement* meOCC_MAP_all;
  MonitorElement* meEN_MAP_all;

  MonitorElement* meMAX_E_all;
  MonitorElement* meMAX_T_all;
  MonitorElement* meEVT_;

  MonitorElement* NADA_NumHotCells;

};

#endif
