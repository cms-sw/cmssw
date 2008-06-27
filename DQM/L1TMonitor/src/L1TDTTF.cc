/*
 * \file L1TDTTF.cc
 *
 * $Date: 2008/06/10 18:01:55 $
 * $Revision: 1.15 $
 * \author J. Berryhill
 *
 * $Log: L1TDTTF.cc,v $
 * Revision 1.15  2008/06/10 18:01:55  lorenzo
 * reduced n histos
 *
 * Revision 1.14  2008/05/09 16:42:27  ameyer
 * *** empty log message ***
 *
 * Revision 1.13  2008/04/30 08:44:21  lorenzo
 * new dttf source, not based on gmt record
 *
 * Revision 1.20  2008/03/20 19:38:25  berryhil
 *
 *
 * organized message logger
 *
 * Revision 1.19  2008/03/14 20:35:46  berryhil
 *
 *
 * stripped out obsolete parameter settings
 *
 * rpc tpg restored with correct dn access and dbe handling
 *
 * Revision 1.18  2008/03/12 17:24:24  berryhil
 *
 *
 * eliminated log files, truncated HCALTPGXana histo output
 *
 * Revision 1.17  2008/03/10 09:29:52  lorenzo
 * added MEs
 *
 * Revision 1.16  2008/03/01 00:40:00  lat
 * DQM core migration.
 *
 * $Log: L1TDTTF.cc,v $
 * Revision 1.15  2008/06/10 18:01:55  lorenzo
 * reduced n histos
 *
 * Revision 1.14  2008/05/09 16:42:27  ameyer
 * *** empty log message ***
 *
 * Revision 1.13  2008/04/30 08:44:21  lorenzo
 * new dttf source, not based on gmt record
 *
 * Revision 1.20  2008/03/20 19:38:25  berryhil
 *
 *
 * organized message logger
 *
 * Revision 1.19  2008/03/14 20:35:46  berryhil
 *
 *
 * stripped out obsolete parameter settings
 *
 * rpc tpg restored with correct dn access and dbe handling
 *
 * Revision 1.18  2008/03/12 17:24:24  berryhil
 *
 *
 * eliminated log files, truncated HCALTPGXana histo output
 *
 * Revision 1.17  2008/03/10 09:29:52  lorenzo
 * added MEs
 *
 * Revision 1.15  2008/01/22 18:56:01  muzaffar
 * include cleanup. Only for cc/cpp files
 *
 * Revision 1.14  2007/12/21 17:41:20  berryhil
 *
 *
 * try/catch removal
 *
 * Revision 1.13  2007/11/19 15:08:22  lorenzo
 * changed top folder name
 *
 * Revision 1.12  2007/08/15 18:56:25  berryhil
 *
 *
 * split histograms by bx; add Maiken's bx classifier plots
 *
 * Revision 1.11  2007/07/26 09:37:09  berryhil
 *
 *
 * set verbose false for all modules
 * set verbose fix for DTTPG tracks
 *
 * Revision 1.10  2007/07/25 09:03:58  berryhil
 *
 *
 * conform to DTTFFEDReader input tag.... for now
 *
 * Revision 1.9  2007/07/12 16:06:18  wittich
 * add simple phi output track histograms.
 * note that the label of this class is different than others
 * from the DTFFReader creates.
 *
 */
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/L1TMonitor/interface/L1TDTTF.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace std;
using namespace edm;

L1TDTTF::L1TDTTF(const ParameterSet& ps)
  : dttpgSource_( ps.getParameter< InputTag >("dttpgSource") )
{

  l1tinfofolder = ps.getUntrackedParameter<string>("l1tInfoFolder", "L1T/EventInfo") ;
  l1tsubsystemfolder = ps.getUntrackedParameter<string>("l1tSystemFolder", "L1T/L1TDTTF") ;
  
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TDTTF: constructor...." << endl;


  dbe = NULL;
  if ( ps.getUntrackedParameter<bool>("DQMStore", false) ) 
    {
      dbe = Service<DQMStore>().operator->();
      dbe->setVerbose(0);
    }
  
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }


  if ( dbe !=NULL ) {
//    dbe->setCurrentFolder("L1T/L1TDTTF");
    dbe->setCurrentFolder(l1tsubsystemfolder);
  }


}

L1TDTTF::~L1TDTTF()
{
}

void L1TDTTF::beginJob(const EventSetup& c)
{

  nev_ = 0;
  
  // get hold of back-end interface
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();
/*  if ( dbe ) {
    dbe->setCurrentFolder("L1T/L1TDTTF");
    dbe->rmdir("L1T/L1TDTTF");
  }
*/

  if ( dbe ) 
    {
      //  error summary histograms
      //    dbe->setCurrentFolder(l1tinfofolder);
      
      //  error summary segments
      //    string suberrfolder = l1tinfofolder + "/reportSummaryContents" ;
      //    dbe->setCurrentFolder(suberrfolder);
      //    dttpgphmap = dbe->book2D("DT_TPG_phi_map","Map of triggers per station",20,1,21,12,0,12);
      //    setMapPhLabel(dttpgphmap);
      
      //    string dttf_phi_folder = l1tsubsystemfolder+"/DTTF_PHI";
      //    string dttf_theta_folder = l1tsubsystemfolder+"/DTTF_THETA";
      string dttf_trk_folder = l1tsubsystemfolder+"/DTTF_TRACKS";
      char hname[40];
      //DTTF Output (6 wheels)
      
      dbe->setCurrentFolder(dttf_trk_folder);
      
      for(int ibx=0;ibx<3;ibx++){
	int tbx=ibx-1; 
      
	dbe->setCurrentFolder(dttf_trk_folder);
      
	for(int iwh=0;iwh<6;iwh++){
	  int twh=iwh-3;
	  if(iwh>=3) twh+=1;
	  
	  ostringstream whnum;
	  whnum << iwh;
	  string whn;
	  whn = whnum.str();
	
	  string dttf_trk_folder_wheel = dttf_trk_folder + "/WHEEL_" + whn;
	  dbe->setCurrentFolder(dttf_trk_folder_wheel);
	
	  for(int ise=0;ise<12;ise++){
	  
	    
	    ostringstream senum;
	    senum << ise;
	    string sen;
	    sen = senum.str();
	    string dttf_trk_folder_sector = dttf_trk_folder_wheel + "/SECTOR_" + sen;
	    dbe->setCurrentFolder(dttf_trk_folder_sector);
	    
	    sprintf(hname,"dttf_p_phi_bx%d_wh%d_se%d",tbx,twh,ise);
	    //dttf_p_phi[ibx][iwh][ise] = dbe->book1D(hname,hname,256,-0.5,255.5);
	    dttf_p_phi[ibx][iwh][ise] = dbe->book1D(hname,hname,144,-0.5,143.5);//according to Ref Manual, phi (packed) has range of 0->143
	    
	    sprintf(hname,"dttf_p_qual_bx%d_wh%d_se%d",tbx,twh,ise);
	    dttf_p_qual[ibx][iwh][ise] = dbe->book1D(hname,hname,8,-0.5,7.5);
	    
	    sprintf(hname,"dttf_p_q_bx%d_wh%d_se%d",tbx,twh,ise);
	    dttf_p_q[ibx][iwh][ise] = dbe->book1D(hname,hname,2,-0.5,1.5);
	    
	    sprintf(hname,"dttf_p_pt_bx%d_wh%d_se%d",tbx,twh,ise);
	    dttf_p_pt[ibx][iwh][ise]= dbe->book1D(hname,hname,32,-0.5,31.5);
	    
	  }

	  //track occupancy info - for each wheel
	  dbe->setCurrentFolder(dttf_trk_folder_wheel);
	  sprintf(hname,"dttf_p_phi_eta_wh%d",twh);
	  dttf_p_phi_eta[iwh] = dbe->book2D(hname,hname,144,-0.5,143.5,100,-0.5,99.5);

	}
      }	

      //integrated values
      string dttf_trk_folder_integrated = dttf_trk_folder + "/INTEG";
      dbe->setCurrentFolder(dttf_trk_folder_integrated);

      //packed values
      sprintf(hname,"dttf_p_phi_integ");
      dttf_p_phi_integ = dbe->book1D(hname,hname,144,-0.5,143.5);
	    
      sprintf(hname,"dttf_p_eta_integ");
      dttf_p_eta_integ = dbe->book1D(hname,hname,2,-0.5,1.5);//what is the eta_packed range??
	    
      sprintf(hname,"dttf_p_pt_integ");
      dttf_p_pt_integ  = dbe->book1D(hname,hname,32,-0.5,31.5);

      sprintf(hname,"dttf_p_qual_integ");
      dttf_p_qual_integ  = dbe->book1D(hname,hname,8,-0.5,7.5);


      //physical values
      sprintf(hname,"dttf_phys_phi_integ");
      dttf_phys_phi_integ = dbe->book1D(hname,hname,144,-0.5,6.5);
	    
      sprintf(hname,"dttf_phys_eta_integ");
      dttf_phys_eta_integ = dbe->book1D(hname,hname,100,-0.5,99.5);//what is max eta value?
	    
      sprintf(hname,"dttf_phys_pt_integ");
      dttf_phys_pt_integ  = dbe->book1D(hname,hname,100,-0.5,99.5);//what is max pt value?

      //track occupancy info - everything
      sprintf(hname,"dttf_p_phi_eta");
      dttf_p_phi_eta_integ = dbe->book2D(hname,hname,144,-0.5,143.5,100,-0.5,99.5);

    } 
}


void L1TDTTF::endJob(void)
{
  if(verbose_) cout << "L1TDTTF: end job...." << endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events"; 

  if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

  return;
}

void L1TDTTF::analyze(const Event& e, const EventSetup& c)
{

  nev_++; 
  if(verbose_) cout << "L1TDTTF: analyze...." << endl;
  

  edm::Handle<L1MuDTTrackContainer > myL1MuDTTrackContainer;

  
    std::string trstring;
    trstring = dttpgSource_.label()+":"+"DATA"+":"+dttpgSource_.process();
    edm::InputTag trInputTag(trstring);
    e.getByLabel(trInputTag,myL1MuDTTrackContainer);

  if (!myL1MuDTTrackContainer.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1MuDTTrackContainer with label "
                               << dttpgSource_.label() ;
    return;
  }

  L1MuDTTrackContainer::TrackContainer *t =  myL1MuDTTrackContainer->getContainer();



   for ( L1MuDTTrackContainer::TrackContainer::const_iterator i 
 	  = t->begin(); i != t->end(); ++i ) {
     if ( verbose_ ) {
       std::cout << "bx = " << i->bx() 
 		<< std::endl;
       std::cout << "quality (packed) = " << i->quality_packed() 
 		<< std::endl;
       std::cout << "pt      (packed) = " << i->pt_packed() << "  , pt  (GeV) = " << i->ptValue()
 		<< std::endl;
       std::cout << "phi     (packed) = " << i->phi_packed() << " , phi (rad) = " << i->phiValue()
 		<< std::endl;
       std::cout << "charge  (packed) = " << i->charge_packed() 
 		<< std::endl;
     }


     int bxindex = i->bx() + 1;
     int wh = i->whNum();//wh has possible values {-3,-2,-1,1,2,3}
     int se = i->scNum();

     int wh2;//make wh2 go from 0 to 5
     if(wh<0)wh2=wh+3;
     else wh2=wh+2;
     

     //std::cout << "whNum = " << wh << "  bxindex = " << bxindex << std::endl;
     //std::cout << "wh2 = " << wh2 << std::endl;
     //std::cout << "se = " << se << std::endl;
     //std::cout << "eta_packed = " << i->eta_packed() << std::endl;


     dttf_p_phi[bxindex][wh2][se]->Fill(i->phi_packed());
     dttf_p_qual[bxindex][wh2][se]->Fill(i->quality_packed());
     dttf_p_pt[bxindex][wh2][se]->Fill(i->pt_packed());
     dttf_p_q[bxindex][wh2][se]->Fill(i->charge_packed());

     dttf_p_phi_integ->Fill(i->phi_packed());
     dttf_p_pt_integ->Fill(i->pt_packed());
     dttf_p_eta_integ->Fill(i->eta_packed());
     dttf_p_qual_integ->Fill(i->quality_packed());

     dttf_phys_phi_integ->Fill(i->phiValue());
     dttf_phys_pt_integ->Fill(i->ptValue());
     dttf_phys_eta_integ->Fill(i->etaValue());

     dttf_p_phi_eta[wh2]->Fill(i->phi_packed(),i->eta_packed());
     dttf_p_phi_eta_integ->Fill(i->phi_packed(),i->eta_packed());

   }
}

void L1TDTTF::setMapPhLabel(MonitorElement *me)
{

  me->setAxisTitle("DTTF Sector",2);
      for(int i=0;i<5;i++){
	ostringstream wheel;
	wheel << i-2;
	me->setBinLabel(1+i*4,"Wheel "+ wheel.str(),1);
      }
  
}

void L1TDTTF::setMapThLabel(MonitorElement *me)
{

  me->setAxisTitle("DTTF Sector",2);
      for(int i=0;i<5;i++){
	ostringstream wheel;
	wheel << i-2;
	me->setBinLabel(1+i*3,"Wheel "+ wheel.str(),1);
      }
  
}
