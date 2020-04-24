// Package:    GenFilters
// Class:      ComphepSingletopFilterPy8
// 
/*
 class ComphepSingletopFilterPy8 ComphepSingletopFilterPy8.cc GeneratorInterface/GenFilters/src/ComphepSingletopFilterPy8.cc

 Description: a filter to match LO/NLO in comphep-generated singletop

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vladimir Molchanov
//         Created:  Wed Mar 25 19:43:12 CET 2009
// $Id: ComphepSingletopFilterPy8.cc,v 1.3 2009/12/15 10:29:32 fabiocos Exp $
// Author:  Alexey Baskakov
//         Created:  Oct 2014
//  ComphepSingletopFilterPy8.cc,v 2.1 
//

// system include files
#include <vector>
#include <boost/format.hpp>

// user include files
#include "GeneratorInterface/GenFilters/interface/ComphepSingletopFilterPy8.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "CLHEP/Vector/LorentzVector.h"

#include "HepMC/IO_GenEvent.h"



using namespace std;
using namespace HepMC;

 ComphepSingletopFilterPy8::ComphepSingletopFilterPy8(const edm::ParameterSet& iConfig):
 token_(consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared")))
{	
	ptsep = iConfig.getParameter<double>("pTSep");
}

ComphepSingletopFilterPy8::~ComphepSingletopFilterPy8() {}

void ComphepSingletopFilterPy8::beginJob() 
{ 
    read22 = read23 = 0;
    pass22 = pass23 = 0;
    hardLep = 23; //identifies the "hard part" in Pythia8
 }

void ComphepSingletopFilterPy8::endJob() 
{
    cout << "Proc:     2-->2     2-->3     Total" << endl;
    cout << boost::format("Read: %9d %9d %9d") % read22 % read23 % (read22+read23)
         << endl;
    cout << boost::format("Pass: %9d %9d %9d") % pass22 % pass23 % (pass22+pass23)
         << endl;
}


bool ComphepSingletopFilterPy8::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  
edm::Handle<edm::HepMCProduct> evt;
//iEvent.getByLabel("generator","unsmeared", evt);
iEvent.getByToken(token_, evt);
const HepMC::GenEvent * myEvt = evt->GetEvent();

int id_bdec=0, id_lJet=0, id_b_from_top=0, id_lep = 0;
//vars for lepton top
const   GenParticle * gp_clep = NULL;
GenVertex * gv_hard = NULL;

const   GenParticle * gp_lep_FSR = NULL;
GenVertex * gv_lep = NULL;
vector<const GenParticle *> vgp_lep;

//vars for add b from top
GenVertex * gv = NULL;
const GenParticle * gp = NULL;
vector<const GenParticle *> vgp_bsec;

//vars for light Jet (light q)
GenVertex * gv_lJet= NULL;
const GenParticle * gplJet= NULL;
vector<const GenParticle *> vgp_lJet;

//vars for b from top
GenVertex * gv_b_t= NULL;
const GenParticle * gp_b_t= NULL;
vector<const GenParticle *> vgp_b_t;


//Run through all particles in myEvt, if lepton found saves it gp_clep
    for (GenEvent::particle_const_iterator it = myEvt->particles_begin(); it != myEvt->particles_end(); ++it)
   {
    if (abs((*it)->status()) == hardLep) 
      {
            int abs_id = abs((*it)->pdg_id());//11=e, -11=E, 13=mu, -13=Mu, 15=l(tau), -15=L
            if (abs_id==11 || abs_id==13 || abs_id==15) 
            {
			gp_clep = *it;
			id_lep = (*it)->pdg_id();
			gv_lep=(*it)->production_vertex();
	    // Lepton FSR
		 while (gv_lep) 
		  {  
		      gp_lep_FSR = NULL;
		  for (GenVertex::particle_iterator it = gv_lep->particles_begin(children); it != gv_lep->particles_end(children); ++it) 
		    {
			if ((*it)->pdg_id() == id_lep) 
			{
			      if (!gp_lep_FSR || (*it)->momentum().perp2() > gp_lep_FSR->momentum().perp2()) 
			      {
				  gp_lep_FSR = *it;
			      }
			  }
		      }      
		      if (gp_lep_FSR) 
		      {
			  gv_lep = gp_lep_FSR->end_vertex();
			  vgp_lep.push_back(gp_lep_FSR); //vertex of 
		      } 
		      else 
		      {
			  gv_lep = NULL; //exits the "while" loop
		      }
		  }
               break;
            }
        }
   }
// Goes to lepton production vertex
    gv_hard = gp_clep->production_vertex(); 
    
    if (! gp_clep) 
    {
        cout << "ERROR: ComphepSingletopFilterPy8: no charged lepton" << endl;
        return false;
    }


//Run through lepton production_vertex 
    for (GenVertex::particle_iterator it = gv_hard->particles_begin(children); it != gv_hard->particles_end(children); ++it) 
   {
        int pdg_id = (*it)->pdg_id(); // KF in Pythia
		int abs_id = abs(pdg_id);

//selection of light quark among particles in primary vertex
	if(abs_id<5) 
	{
		id_lJet = (*it)->pdg_id();
		gv_lJet=(*it)->production_vertex();
		
		 while (gv_lJet) 
		  {  
		      gplJet = NULL;
		  for (GenVertex::particle_iterator it = gv_lJet->particles_begin(children); it != gv_lJet->particles_end(children); ++it) 
		    {
			if ((*it)->pdg_id() == id_lJet) 
			{
			      if (!gplJet || (*it)->momentum().perp2() > gplJet->momentum().perp2()) 
			      {
				  gplJet = *it;
			      }
			  }
		      }
		      
		      if (gplJet) 
		      {
			  gv_lJet = gplJet->end_vertex();
			  vgp_lJet.push_back(gplJet); //vertex of 
		      } 
		      else 
		      {
			  gv_lJet = NULL; //exits the "while" loop
		      }
		  }		  
	}

    if (abs(pdg_id) == 5) // 5 = b
       {
            if (pdg_id * (gp_clep->pdg_id()) < 0) 
           { //b is from top 
                id_bdec = pdg_id;

		id_b_from_top = (*it)->pdg_id();
		gv_b_t = (*it)->production_vertex();
		
		 while (gv_b_t) 
		  {  
		      gp_b_t = NULL;
		  for (GenVertex::particle_iterator it = gv_b_t->particles_begin(children); it != gv_b_t->particles_end(children); ++it) 
		    {
			if ((*it)->pdg_id() == id_b_from_top) 
				{
			      if (!gp_b_t || (*it)->momentum().perp2() > gp_b_t->momentum().perp2()) 
			      {
				  gp_b_t = *it;
			      }
				}	
		    }
		      
		      if (gp_b_t) 
		      {
			  gv_b_t = gp_b_t->end_vertex();
			  vgp_b_t.push_back(gp_b_t); //vertex of 
		      } 
		      else 
		      {
			  gv_b_t = NULL; //exits the "while" loop
		      }
		  }  
           } 
           else  
           {//If process 2-3, then aditional b in the initial state fills
                vgp_bsec.push_back(*it); 
           }
        }
    }


bool process22 = (vgp_bsec.size() == 0); //if there is no aditional b-quark in primary vexrtes, then it is tq-process (2->2) 

    if (process22) 
    {
        for (GenVertex::particle_iterator it = gv_hard->particles_begin(parents); it != gv_hard->particles_end(parents); ++it)  //Among parents of lepton production vertex(primary vertex) we find b-quark.
       {

            if ((*it)->pdg_id() == id_bdec) 
            {
                gv = (*it)->production_vertex();
               break;
            }
        }
        if (! gv) 
        {
            cerr << "ERROR: ComphepSingletopFilterPy8: HepMC inconsistency (! gv)" << endl;
            myEvt->print();
            return false;
        }
    } 
   else 
    {
        gv = vgp_bsec.back()->end_vertex();
    }

//##Correction for GV vertex, because of additional gluons in ISR
bool WeFoundAdditional_b_quark = false;
int loopCount = 0, gv_loopCount = 0;
    while (WeFoundAdditional_b_quark != true) 
    {     
////we go through b or B quark (not from top, top parent) production vertex
        for (GenVertex::particle_iterator it = gv->particles_begin(children); it != gv->particles_end(children); ++it) 
       {
////if we found b, but anti to ours, than it is additional b quark from ISR
           if ((*it)->pdg_id() == -id_bdec) 
           { //we found right vertex of ISR gluon splitting!
	     WeFoundAdditional_b_quark = true;//gv = (*it)->production_vertex();
           }
       }
       if (WeFoundAdditional_b_quark == false)
       {
////If we don't find add b quark, we need to go to parents vertex, to find it there
	    for (GenVertex::particle_iterator it = gv->particles_begin(parents); it != gv->particles_end(parents); ++it) 
	     {
	       if ((*it)->pdg_id() == id_bdec) 
		{
		  gv = (*it)->production_vertex();

		}
	     }
       }
      loopCount++;

      if (loopCount > 100)//loop protection, nothing more 
      {
cerr << "ERROR: ComphepSingletopFilterPy8: HepMC inconsistency (No add b vertex found)" << endl;
	break;
      }
    }



    while (gv) 
    {
        gp = NULL;

        for (GenVertex::particle_iterator it = gv->particles_begin(children); it != gv->particles_end(children); ++it) 
       {
           if ((*it)->pdg_id() == -id_bdec) 
           {
                if (!gp || (*it)->momentum().perp2() > gp->momentum().perp2()) 
                {
                    gp = *it;
		    
				}
            }
        }
        if (gp) 
        {
            gv = gp->end_vertex();
            vgp_bsec.push_back(gp); //vertex of 
        } 
        else 
        {
            gv = NULL; //exits the "while" loop
        }
        gv_loopCount++;
    }


    if (vgp_bsec.size() == 0) 
    {
        cerr << "ERROR: ComphepSingletopFilterPy8: HepMC inconsistency (vgp_bsec.size() == 0)" << endl;
        return false;
    }
    
    double pt = vgp_bsec.back()->momentum().perp();
   // double eta = vgp_bsec.back()->momentum().eta();  
   bool pass;
    if (process22) 
   {
        read22 += 1;
        pass = pt < ptsep;
        if (pass) pass22 += 1;
    }
    else
    {
        read23 += 1;
        pass = ptsep <= pt;
        if (pass) pass23 += 1;
    }
 return pass;
}


