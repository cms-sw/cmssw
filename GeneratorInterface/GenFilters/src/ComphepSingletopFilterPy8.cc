
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

#include "TFile.h"
#include "TH1F.h"
#include "TTree.h"

#include "HepMC/IO_GenEvent.h"

#include "TLorentzVector.h"


using namespace std;
using namespace HepMC;
//using namespace edm;

 ComphepSingletopFilterPy8::ComphepSingletopFilterPy8(const edm::ParameterSet& iConfig)
{	
	ptsep = iConfig.getParameter<double>("pTSep");
	iWriteFile = iConfig.getParameter<bool>("writefile");
	outputFileName = iConfig.getParameter<std::string>("outputFileName");
	isPythia8 = iConfig.getParameter<bool>("isPythia8");
}

ComphepSingletopFilterPy8::~ComphepSingletopFilterPy8() {}

void ComphepSingletopFilterPy8::beginJob() 
{ 
//    cout <<"### beginJob ###"<<endl;
    read22 = read23 = 0;
    pass22 = pass23 = 0;
       
    if (iWriteFile)
    {
    if (isPythia8)
    {
      hardLep = 23; //identifes the "hard part" in Pythia8
    }
    else 
    {
      hardLep = 3; //identifes the "hard part" of the interaction https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookGenParticleCandidate
    }
    moutFile = new TFile(outputFileName,"RECREATE");
    moutFile->cd();
    Matching  = new TTree ("M", "Matching",1);
    pt_add_b=0.;
    pt_add_b_FSR =0.;
    eta_add_b=0.;
    eta_add_b_FSR=0.;
    pt_t_b=0.;
    eta_t_b=0.;
    pt_l=0.;
    eta_l=0.;
   Matching->Branch ("pt_l",&pt_l,"pt_l/D");
   Matching->Branch ("eta_l",&eta_l,"eta_l/D");
   Matching->Branch ("pt_l_FSR",&pt_l_FSR,"pt_l_FSR/D");
   Matching->Branch ("eta_l_FSR",&eta_l_FSR,"eta_l_FSR/D");
   
   Matching->Branch ("pt_t_b",&pt_t_b,"pt_t_b/D");
   Matching->Branch ("eta_t_b",&eta_t_b,"eta_t_b/D");
   Matching->Branch ("pt_t_b_FSR",&pt_t_b_FSR,"pt_t_b_FSR/D");
   Matching->Branch ("eta_t_b_FSR",&eta_t_b_FSR,"eta_t_b_FSR/D");


   Matching->Branch ("pt_add_b",&pt_add_b,"pt_add_b/D");
   Matching->Branch ("eta_add_b",&eta_add_b,"eta_add_b/D");
   Matching->Branch ("pt_add_b_FSR",&pt_add_b_FSR ,"pt_add_b_FSR/D");
   Matching->Branch ("eta_add_b_FSR",&eta_add_b_FSR,"eta_add_b_FSR/D");

   Matching->Branch ("pt_light_q",&pt_light_q,"pt_light_q/D");
   Matching->Branch ("eta_light_q",&eta_light_q,"eta_light_q/D");
   Matching->Branch ("pt_light_q_FSR",&pt_light_q_FSR,"pt_light_q_FSR/D");
   Matching->Branch ("eta_light_q_FSR",&eta_light_q_FSR,"eta_light_q_FSR/D");

   Matching->Branch ("pt_top",&pt_top,"pt_top/D");
   Matching->Branch ("eta_top",&eta_top,"eta_top/D");
   Matching->Branch ("pt_top_FSR",&pt_top_FSR,"pt_top/D");
   Matching->Branch ("eta_top_FSR",&eta_top_FSR,"eta_top_FSR/D");


   Matching->Branch ("Cos_CargedLep_LJet_noFSR",&Cos_CargedLep_LJet);
   Matching->Branch ("Cos_CargedLep_LJet_FSR",&Cos_CargedLep_LJet_FSR);
   
    }
 }

void ComphepSingletopFilterPy8::endJob() 
{
    using namespace std;

    cout << "Proc:     2-->2     2-->3     Total" << endl;
    cout << boost::format("Read: %9d %9d %9d") % read22 % read23 % (read22+read23)
         << endl;
    cout << boost::format("Pass: %9d %9d %9d") % pass22 % pass23 % (pass22+pass23)
         << endl;

if (iWriteFile)
{
  moutFile->Write();
  moutFile->Close();
}
}

bool ComphepSingletopFilterPy8::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
//   cout <<"###  Filter Start 1 ###"<<endl;
edm::Handle<edm::HepMCProduct> evt;
iEvent.getByLabel("generator", evt);
const HepMC::GenEvent * myEvt = evt->GetEvent();
// HepMC::GenEvent * myEvt_test = evt->GetEvent();
//   myEvt->print();  // to print the record


//This four-vectors are used to plot cos between charged lepton and light jet(spectator jet (quark))
HepMC::FourVector p4Nu_hep, p4B1_hep, p4Lep_hep, p4Lep_FSR_hep, lJetP4_hep, lJet_FSR_P4_hep,p4B2_hep;
TLorentzVector p4W, p4Nu, p4B1, p4Lep, p4Lep_FSR, p4Top, lJetP4, lJet_FSR_P4, p4B2 ;



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
		pt_l=(*it)->momentum().perp();
 		eta_l=(*it)->momentum().eta();
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
		  p4Lep_FSR_hep=vgp_lep.back()->momentum();
		  p4Lep_FSR.SetXYZT(p4Lep_FSR_hep.x(),p4Lep_FSR_hep.y(),p4Lep_FSR_hep.z(),p4Lep_FSR_hep.t());		 
		  pt_l_FSR=vgp_lep.back()->momentum().perp();
		  eta_l_FSR=vgp_lep.back()->momentum().eta();
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

	if(abs_id==12 || abs_id==14 || abs_id==16)
	{
		p4Nu_hep=(*it)->momentum();
		p4Nu.SetXYZT(p4Nu_hep.x(),p4Nu_hep.y(),p4Nu_hep.z(),p4Nu_hep.t());
		p4Lep_hep = gp_clep->momentum();
		p4Lep.SetXYZT(p4Lep_hep.x(),p4Lep_hep.y(),p4Lep_hep.z(),p4Lep_hep.t());
		
	}
//selection of light quark among particles in primary vertex
	if(abs_id<5) 
	{
		lJetP4_hep=(*it)->momentum();
		lJetP4.SetXYZT(lJetP4_hep.x(),lJetP4_hep.y(),lJetP4_hep.z(),lJetP4_hep.t());

		id_lJet = (*it)->pdg_id();

    		eta_light_q=(*it)->momentum().eta();
		pt_light_q=(*it)->momentum().perp();
		

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
		  lJet_FSR_P4_hep=vgp_lJet.back()->momentum();
		  lJet_FSR_P4.SetXYZT(lJet_FSR_P4_hep.x(),lJet_FSR_P4_hep.y(),lJet_FSR_P4_hep.z(),lJet_FSR_P4_hep.t());
		  eta_light_q_FSR=vgp_lJet.back()->momentum().eta();
		  pt_light_q_FSR=vgp_lJet.back()->momentum().perp();
	}

    if (abs(pdg_id) == 5) // 5 = b
       {
            if (pdg_id * (gp_clep->pdg_id()) < 0) 
           { //b is from top 
                id_bdec = pdg_id;
 		pt_t_b=(*it)->momentum().perp();
 		eta_t_b=(*it)->momentum().eta();

		p4B1_hep = (*it)->momentum();
		p4B1.SetXYZT(p4B1_hep.x(),p4B1_hep.y(),p4B1_hep.z(),p4B1_hep.t());


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
		p4B2_hep = vgp_b_t.back()->momentum();
		p4B2.SetXYZT(p4B2_hep.x(),p4B2_hep.y(),p4B2_hep.z(),p4B2_hep.t());
		pt_t_b_FSR=vgp_b_t.back()->momentum().perp();
		eta_t_b_FSR=vgp_b_t.back()->momentum().eta();
           } 
           else  
           {//If process 2-3, then aditional b in the initial state fills
                vgp_bsec.push_back(*it); 
 		pt_add_b=(*it)->momentum().perp();
 		eta_add_b=(*it)->momentum().eta();

           }
        }
    }

p4W = p4Nu + p4Lep;
p4Top = p4W + p4B1;
pt_top=p4Top.Pt();
eta_top=p4Top.Eta();
TVector3 boostV = p4Top.BoostVector();
TLorentzVector boostedLep = p4Lep;
boostedLep.Boost(-boostV);
TVector3 p3Lepton = boostedLep.Vect();
TLorentzVector boostedLJet = lJetP4;
boostedLJet.Boost(-boostV);
TVector3 p3LJet = boostedLJet.Vect();
Cos_CargedLep_LJet= p3Lepton.Dot(p3LJet) / p3Lepton.Mag() / p3LJet.Mag();

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
		    
		   if (gv_loopCount==0)
		   {
		     pt_add_b = (*it)->momentum().perp();
		     eta_add_b=(*it)->momentum().eta();			     
		  }
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
p4W = p4Nu + p4Lep_FSR;
p4Top = p4W + p4B2;
pt_top_FSR=p4Top.Pt();
eta_top_FSR=p4Top.Eta();
boostV  = p4Top.BoostVector();
boostedLep = p4Lep;
boostedLep.Boost(-boostV);
p3Lepton = boostedLep.Vect();
boostedLJet = lJet_FSR_P4;
boostedLJet.Boost(-boostV);
p3LJet = boostedLJet.Vect();
Cos_CargedLep_LJet_FSR= p3Lepton.Dot(p3LJet) / p3Lepton.Mag() / p3LJet.Mag();


    if (vgp_bsec.size() == 0) 
    {
        cerr << "ERROR: ComphepSingletopFilterPy8: HepMC inconsistency (vgp_bsec.size() == 0)" << endl;
        return false;
    }
    
    double pt = vgp_bsec.back()->momentum().perp();
    double eta = vgp_bsec.back()->momentum().eta();  

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

    if (pass) 
    {

       pt_add_b_FSR = pt;      
       eta_add_b_FSR=eta;	
       moutFile->cd();
       Matching->Fill();
    }	

 return pass;
}


