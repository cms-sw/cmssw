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

      /*
      //  error summary histograms
      dbe->setCurrentFolder(l1tinfofolder);
      
      //  error summary segments
      string suberrfolder = l1tinfofolder + "/reportSummaryContents" ;
      dbe->setCurrentFolder(suberrfolder);
      dttpgphmap = dbe->book2D("DT_TPG_phi_map","Map of triggers per station",20,1,21,12,0,12);
      setMapPhLabel(dttpgphmap);
      
      string dttf_phi_folder = l1tsubsystemfolder+"/DTTF_PHI";
      string dttf_theta_folder = l1tsubsystemfolder+"/DTTF_THETA";
      */

      string dttf_trk_folder = l1tsubsystemfolder+"/DTTF_TRACKS";

      char hname[40];

      /*

      dbe->setCurrentFolder(dttf_phi_folder);
    

      //hist1[0]
      dttpgphbx[0] = dbe->book1D("BxEncoding_PHI",
				 "Bunch encoding DTTF Phi",11,0,11);
      //hist1[1]
      dttpgphbx[1] = dbe->book1D("BxEncoding_OUT",
				 "Bunch encoding DTTF Output",11,0,11);

      for(int i=0;i<2;i++){
	dttpgphbx[i]->setBinLabel(1,"None");
	dttpgphbx[i]->setBinLabel(3,"Only bx=-1");
	dttpgphbx[i]->setBinLabel(4,"Only bx= 0");
	dttpgphbx[i]->setBinLabel(5,"Only bx=+1");
	dttpgphbx[i]->setBinLabel(7,"Bx=-1,0");
	dttpgphbx[i]->setBinLabel(8,"Bx=-1,1");
	dttpgphbx[i]->setBinLabel(9,"Bx= 0,1");
	dttpgphbx[i]->setBinLabel(11,"All bx");
      }
   

      dttpgphbxcomp = dbe->book2D("BxEncoding_PHI_OUT",
				  "Bunch encoding: DTTF Phi vs. Output",
				  11,0,11,11,0,11);
      dttpgphbxcomp->setAxisTitle("DTTF (output)",1);
      dttpgphbxcomp->setAxisTitle("PHI-TF",2);
      for(int i=1;i<=2;i++){
	dttpgphbxcomp->setBinLabel(1,"None",i);
	dttpgphbxcomp->setBinLabel(3,"Only bx=-1",i);
	dttpgphbxcomp->setBinLabel(4,"Only bx= 0",i);
	dttpgphbxcomp->setBinLabel(5,"Only bx=+1",i);
	dttpgphbxcomp->setBinLabel(7,"Bx=-1,0",i);
	dttpgphbxcomp->setBinLabel(8,"Bx=-1,1",i);
	dttpgphbxcomp->setBinLabel(9,"Bx= 0,1",i);
	dttpgphbxcomp->setBinLabel(11,"All bx",i);
      }

      //hist1[2]
      dttpgphbx[2] = dbe->book1D("NumberOfSegmentsPHI_BunchNeg1",
				 "Number of segments for bunch -1 Dttf Phi",
				 20,0,20);
      //hist1[3]
      dttpgphbx[3] = dbe->book1D("NumberOfSegmentsPHI_Bunch0",
				 "Number of segments for bunch 0 Dttf Phi",
				 20,0,20);
      //hist1[4]
      dttpgphbx[4] = dbe->book1D("NumberOfSegmentsPHI_Bunch1",
				 "Number of segments for bunch 1 Dttf Phi",
				 20,0,20);
      //hist1[5]
      dttpgphbx[5] = dbe->book1D("NumberOfSegmentsOUT_BunchNeg1",
				 "Number of segments for bunch -1 Dttf Output",
				 20,0,20);
      //hist1[6] 
      dttpgphbx[6] = dbe->book1D("NumberOfSegmentsOUT_Bunch0",
				 "Number of segments for bunch 0 Dttf Output",
				 20,0,20);
      //hist1[7]
      dttpgphbx[7] = dbe->book1D("NumberOfSegmentsOUT_Bunch1",
				 "Number of segments for bunch 1 Dttf Output",
				 20,0,20);

      dttpgphntrack = dbe->book1D("DT_TPG_phi_ntrack", 
				  "DT TPG phi ntrack", 20, -0.5, 19.5 ) ;  

      dttpgthntrack = dbe->book1D("DT_TPG_theta_ntrack", 
				  "DT TPG theta ntrack", 20, -0.5, 19.5 ) ;  


      

// DTTF INPUT Phi
      for(int ibx=0;ibx<3;ibx++){
         int tbx=ibx-1; 
	 
	 dbe->setCurrentFolder(dttf_phi_folder);
	 
	 sprintf(hname,"DT_TPG_phi_bx%d_wheel_number",tbx);
	 dttpgphwheel[ibx] = dbe->book1D(hname,hname, 5, -2.5, 2.5 ) ;  
         
	 for(int iwh=0;iwh<5;iwh++){
	   int twh=iwh-2;
	    
	   ostringstream whnum;
	   whnum << iwh;
	   string whn;
	   whn = whnum.str();

	   string dttf_phi_folder_wheel = dttf_phi_folder + "/WHEEL_" + whn;
	   dbe->setCurrentFolder(dttf_phi_folder_wheel);
 
	   sprintf(hname,"DT_TPG_phi_bx%d_wh%d_sector_number",tbx,twh);
	   dttpgphsector[ibx][iwh] = dbe->book1D(hname,hname, 12, -0.5, 11.5 ) ;  
            
	   for(int ise=0;ise<12;ise++){

	     ostringstream senum;
	     senum << ise;
	     string sen;
	     sen = senum.str();
	     string dttf_phi_folder_sector = dttf_phi_folder_wheel + "/SECTOR_" + sen;
	     dbe->setCurrentFolder(dttf_phi_folder_sector);
	     
	     sprintf(hname,"DT_TPG_phi_bx%d_wh%d_se%d_station_number",tbx,twh,ise);
	     dttpgphstation[ibx][iwh][ise] = dbe->book1D(hname,hname, 5, -0.5, 4.5 ) ;  
	     
	     for(int ist=0;ist<5;ist++){
	       
	       ostringstream stnum;
	       stnum << ist;
	       string stn;
	       stn = stnum.str();
	       
	       string dttf_phi_folder_station = dttf_phi_folder_sector + "/STATION_" + stn;
	       dbe->setCurrentFolder(dttf_phi_folder_station);
	       
	       sprintf(hname,"DT_TPG_phi_Seg1_phi_bx%d_wh%d_se%d_st%d",tbx,twh,ise,ist);
	       dttpgphsg1phiAngle[ibx][iwh][ise][ist] = dbe->book1D(hname,hname,256,-0.5,255.5);
		    
	       sprintf(hname,"DT_TPG_phi_Seg1_phiBanding_bx%d_wh%d_se%d_st%d",tbx,twh,ise,ist);
	       dttpgphsg1phiBandingAngle[ibx][iwh][ise][ist] = dbe->book1D(hname,hname,256,-0.5,255.5);
		    
	       sprintf(hname,"DT_TPG_phi_Seg1_quality_bx%d_wh%d_se%d_st%d",tbx,twh,ise,ist);
	       dttpgphsg1quality[ibx][iwh][ise][ist] = dbe->book1D(hname,hname,8,-0.5,7.5);
	            
	       sprintf(hname,"DT_TPG_phi_Seg2_phi_bx%d_wh%d_se%d_st%d",tbx,twh,ise,ist);
	       dttpgphsg2phiAngle[ibx][iwh][ise][ist] = dbe->book1D(hname,hname,256,-0.5,255.5);
		    
	       sprintf(hname,"DT_TPG_phi_Seg2_phiBanding_bx%d_wh%d_se%d_st%d",tbx,twh,ise,ist);
	       dttpgphsg2phiBandingAngle[ibx][iwh][ise][ist] = dbe->book1D(hname,hname,256,-0.5,255.5);
		    
	       sprintf(hname,"DT_TPG_phi_Seg2_quality_bx%d_wh%d_se%d_st%d",tbx,twh,ise,ist);
	       dttpgphsg2quality[ibx][iwh][ise][ist] = dbe->book1D(hname,hname,8,-0.5,7.5);
	            
	       sprintf(hname,"DT_TPG_phi_Ts2Tag_bx%d_wh%d_se%d_st%d",tbx,twh,ise,ist);
	       dttpgphts2tag[ibx][iwh][ise][ist]= dbe->book1D(hname,hname,2,-0.5,1.5);
	            
		    
	     }
	   }
         }
      }	 

      dbe->setCurrentFolder(dttf_theta_folder);
      
// DTTF INPUT Theta
      for(int ibx=0;ibx<3;ibx++){
	int tbx=ibx-1; 
	    
	dbe->setCurrentFolder(dttf_theta_folder);

	sprintf(hname,"DT_TPG_theta_bx%d_wheel_number",tbx);
	dttpgthwheel[ibx] = dbe->book1D(hname,hname, 5, -2.5, 2.5 ) ;  
	
	for(int iwh=0;iwh<5;iwh++){
	  int twh=iwh-2;
	    
	  ostringstream whnum;
	  whnum << iwh;
	  string whn;
	  whn = whnum.str();

	  string dttf_theta_folder_wheel = dttf_theta_folder + "/WHEEL_" + whn;
	  dbe->setCurrentFolder(dttf_theta_folder_wheel);
	    
	  sprintf(hname,"DT_TPG_theta_bx%d_wh%d_sector_number",tbx,twh);
	  dttpgthsector[ibx][iwh] = dbe->book1D(hname,hname, 12, -0.5, 11.5 ) ;  
            
	  for(int ise=0;ise<12;ise++){

	    ostringstream senum;
	    senum << ise;
	    string sen;
	    sen = senum.str();
	    string dttf_theta_folder_sector = dttf_theta_folder_wheel + "/SECTOR_" + sen;
	    dbe->setCurrentFolder(dttf_theta_folder_sector);
 	       
	    sprintf(hname,"DT_TPG_theta_bx%d_wh%d_se%d_station_number",tbx,twh,ise);
	    dttpgthstation[ibx][iwh][ise] = dbe->book1D(hname,hname, 5, -0.5, 4.5 ) ;  
               
	    for(int ist=0;ist<4;ist++){
	      
	      ostringstream stnum;
	      stnum << ist;
	      string stn;
	      stn = stnum.str();
	      
	      string dttf_theta_folder_station = dttf_theta_folder_sector + "/STATION_" + stn;
	      dbe->setCurrentFolder(dttf_theta_folder_station);
	      
	      sprintf(hname,"DT_TPG_theta_quality_bx%d_wh%d_se%d_st%d",tbx,twh,ise,ist);
	      dttpgthquality[ibx][iwh][ise][ist] = dbe->book1D(hname,hname,8,-0.5,7.5);
	      
	      sprintf(hname,"DT_TPG_theta_theta_bx%d_wh%d_se%d_st%d",tbx,twh,ise,ist);
	      dttpgththeta[ibx][iwh][ise][ist]= dbe->book1D(hname,hname,2,-0.5,1.5);      
    
	    }
	  }
	}
      }	 
      
	 
      for(int iwh=0;iwh<5;iwh++){					      
	int twh=iwh-2;
      									      
	for(int ise=0;ise<12;ise++){					      
            								      
	  for(int ist=0;ist<5;ist++){
	    sprintf(hname,"bxNumber_wh%d_se%d_st%d",twh,ise,ist);	      
	    bxnumber[iwh][ise][ist]= dbe->book1D(hname,hname,3,-1.5,1.5);   
	  }								      
	}								      
      }

      */
      
//DTTF Output (6 wheels)
      
     
      
      dbe->setCurrentFolder(dttf_trk_folder);
      
      for(int iwh=0;iwh<6;iwh++){
	int twh=iwh-3;
	if(iwh>=3) twh+=1;
	
	//ostringstream whnum;
	//whnum << iwh;
	string whn;
	//whn = whnum.str();
	
	switch(iwh){
	case 0:  whn = "N2";  break;
	case 1:  whn = "N1";  break;
	case 2:  whn = "N0";  break;
	case 3:  whn = "P0";  break;
	case 4:  whn = "P1";  break;
	case 5:  whn = "P2";  break;   
	}
	
	string dttf_trk_folder_wheel = dttf_trk_folder + "/WHEEL_" + whn;
	dbe->setCurrentFolder(dttf_trk_folder_wheel);
	
	
	for(int ibx=0;ibx<3;ibx++){
	  int tbx=ibx-1; 
	  
	  ostringstream bxnum;
	  bxnum << ibx;
	  string bxn = bxnum.str();
	  string dttf_trk_folder_bx = dttf_trk_folder_wheel + "/BX_" + bxn;
	  
	  dbe->setCurrentFolder(dttf_trk_folder_bx);
	  
	  //QUALITY folder
	  string dttf_trk_folder_quality = dttf_trk_folder_bx + "/Quality";
	  dbe->setCurrentFolder(dttf_trk_folder_quality);

	  for(int ise=0;ise<12;ise++){
	    sprintf(hname,"dttf_p_qual_bx%d_wh%d_se%d",tbx,twh,ise);
	    dttf_p_qual[ibx][iwh][ise] = dbe->book1D(hname,hname,8,-0.5,7.5);	  
	  }

	  //PHI folder
	  string dttf_trk_folder_phi = dttf_trk_folder_bx + "/Phi";
	  dbe->setCurrentFolder(dttf_trk_folder_phi);

	  for(int ise=0;ise<12;ise++){
	    sprintf(hname,"dttf_p_phi_bx%d_wh%d_se%d",tbx,twh,ise);
	    dttf_p_phi[ibx][iwh][ise] = dbe->book1D(hname,hname,144,-0.5,144.5);
	    //dttf_p_phi[ibx][iwh][ise] = dbe->book1D(hname,hname, 32,-16.5, 15.5);
	  }

	  //ETA folder
	  string dttf_trk_folder_eta = dttf_trk_folder_bx + "/Eta";
	  dbe->setCurrentFolder(dttf_trk_folder_eta);

	  for(int ise=0;ise<12;ise++){
	    sprintf(hname,"dttf_p_eta_bx%d_wh%d_se%d",tbx,twh,ise);
	    dttf_p_eta[ibx][iwh][ise] = dbe->book1D(hname,hname,64,-32.5,32.5);//fix range and bin size!
	  }
	  
	  //PT folder
	  string dttf_trk_folder_pt = dttf_trk_folder_bx + "/PT";
	  dbe->setCurrentFolder(dttf_trk_folder_pt);

	  for(int ise=0;ise<12;ise++){
	    sprintf(hname,"dttf_p_pt_bx%d_wh%d_se%d",tbx,twh,ise);
	    dttf_p_pt[ibx][iwh][ise]= dbe->book1D(hname,hname,32,-0.5,31.5);
	  }

	  //CHARGE folder
	  string dttf_trk_folder_charge = dttf_trk_folder_bx + "/Charge";
	  dbe->setCurrentFolder(dttf_trk_folder_charge);
	  
	  for(int ise=0;ise<12;ise++){
	    sprintf(hname,"dttf_p_q_bx%d_wh%d_se%d",tbx,twh,ise);
	    dttf_p_q[ibx][iwh][ise] = dbe->book1D(hname,hname,2,-0.5,1.5);
	  }

	}
	
	//track occupancy info - for each wheel
	dbe->setCurrentFolder(dttf_trk_folder_wheel);
	sprintf(hname,"dttf_p_phi_eta_wh%d",twh);
	dttf_p_phi_eta[iwh] = dbe->book2D(hname,hname,144,-0.5,143.5,64,-0.5,64.5);
      }	

      //integrated values
      string dttf_trk_folder_integrated = dttf_trk_folder + "/INTEG";
      dbe->setCurrentFolder(dttf_trk_folder_integrated);

      //packed values
      sprintf(hname,"dttf_p_phi_integ");
      dttf_p_phi_integ = dbe->book1D(hname,hname,144,-0.5,143.5);
	    
      sprintf(hname,"dttf_p_eta_integ");
      dttf_p_eta_integ = dbe->book1D(hname,hname,64,-32.5,32.5);
	    
      sprintf(hname,"dttf_p_pt_integ");
      dttf_p_pt_integ  = dbe->book1D(hname,hname,32,-0.5,31.5);

      sprintf(hname,"dttf_p_qual_integ");
      dttf_p_qual_integ  = dbe->book1D(hname,hname,8,-0.5,7.5);


      //physical values
      sprintf(hname,"dttf_phys_phi_integ");
      dttf_phys_phi_integ = dbe->book1D(hname,hname,144,-0.5,360.5);
	    
      sprintf(hname,"dttf_phys_eta_integ");
      dttf_phys_eta_integ = dbe->book1D(hname,hname,64,-1.2,1.2);//eta range (-1.2,1.2)
	    
      sprintf(hname,"dttf_phys_pt_integ");
      dttf_phys_pt_integ  = dbe->book1D(hname,hname,100,-0.5,99.5);//what is max pt value?

      //track occupancy info - everything
      sprintf(hname,"dttf_p_phi_eta");
      dttf_p_phi_eta_integ = dbe->book2D(hname,hname,144,-0.5,360.5,64,-1.2,1.2);

      sprintf(hname,"Occupancy_Summary");
      //dttf_occupancySummary = dbe->book2D(hname,hname,12,1,13,6,0,6);
      //dttf_occupancySummary->setAxisTitle("sector",1);
      //dttf_occupancySummary->setAxisTitle("wheel",2);
      dttf_occupancySummary = dbe->book2D(hname,hname,6,0,6,12,1,13);
      dttf_occupancySummary->setAxisTitle("wheel",1);
      dttf_occupancySummary->setAxisTitle("sector",2);
      dttf_occupancySummary->setBinLabel(1,"N2",1);
      dttf_occupancySummary->setBinLabel(2,"N1",1);
      dttf_occupancySummary->setBinLabel(3,"N0",1);
      dttf_occupancySummary->setBinLabel(4,"P0",1);
      dttf_occupancySummary->setBinLabel(5,"P1",1);
      dttf_occupancySummary->setBinLabel(6,"P2",1);


      /*

      dbe->setCurrentFolder(l1tsubsystemfolder);

      for (int ibx=0 ; ibx<=2; ibx++) {
	
	ostringstream bxnum;
	bxnum << ibx-1;
	string bxn;
	if (ibx<2)
	  bxn = bxnum.str();
	else
	  bxn = "+" + bxnum.str();
	

    	//phi
        dttpgphmapbx[ibx] = dbe->book2D("DT_TPG_phi_map_bx"+bxn,
					"Map of triggers per station (BX="+bxn+")",20,1,21,12,0,12);
	setMapPhLabel(dttpgphmapbx[ibx]);

	//Theta
	dttpgthbx[ibx] = dbe->book1D("DT_TPG_theta_bx_"+bxn, 
				     "DT TPG theta bx "+bxn, 50, -24.5, 24.5 ) ;  
	
	dttpgthmapbx[ibx] = dbe->book2D("DT_TPG_theta_map_bx_"+bxn,
					"Map of triggers per station (BX="+bxn+")",15,1,16,12,0,12);
	setMapThLabel(dttpgthmapbx[ibx]);

      }
      

//      dttpgphmap = dbe->book2D("DT_TPG_phi_map","Map of triggers per station",20,1,21,12,0,12); // moved to errfolder

      dttpgphmapcorr = dbe->book2D("DT_TPG_phi_map_corr",
				   "Map of correlated triggers per station",20,1,21,12,0,12);
      dttpgphmap2nd = dbe->book2D("DT_TPG_phi_map_2nd",
				  "Map of second tracks per station",20,1,21,12,0,12);
      dttpgphbestmap = dbe->book2D("DT_TPG_phi_best_map",
				   "Map of best triggers per station",20,1,21,12,0,12);
      dttpgphbestmapcorr = dbe->book2D("DT_TPG_phi_best_map_corr",
				       "Map of correlated best triggers per station",20,1,21,12,0,12);
      setMapPhLabel(dttpgphmapcorr);
      setMapPhLabel(dttpgphmap2nd);
      setMapPhLabel(dttpgphbestmap);
      setMapPhLabel(dttpgphbestmapcorr);

      dttpgthmap = dbe->book2D("DT_TPG_theta_map",
			       "Map of triggers per station",15,1,16,12,0,12);
      dttpgthmaph = dbe->book2D("DT_TPG_theta_map_h",
				"Map of H quality triggers per station",15,1,16,12,0,12);
      dttpgthbestmap = dbe->book2D("DT_TPG_theta_best_map",
				   "Map of besttriggers per station",15,1,16,12,0,12);
      dttpgthbestmaph = dbe->book2D("DT_TPG_theta_best_map_h",
				    "Map of H quality best triggers per station",15,1,16,12,0,12);
      setMapThLabel(dttpgthmap);
      setMapThLabel(dttpgthmaph);
      setMapThLabel(dttpgthbestmap);
      setMapThLabel(dttpgthbestmaph);


*/

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


  /*
///
  edm::Handle<L1MuDTChambPhContainer > myL1MuDTChambPhContainer;  
  e.getByLabel(dttpgSource_,myL1MuDTChambPhContainer);
  
  if (!myL1MuDTChambPhContainer.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1MuDTChambPhContainer with label "
			     << dttpgSource_.label() ;
    return;
  }
  
  L1MuDTChambPhContainer::Phi_Container *myPhContainer =  
    myL1MuDTChambPhContainer->getContainer();

  edm::Handle<L1MuDTChambThContainer > myL1MuDTChambThContainer;  
  e.getByLabel(dttpgSource_,myL1MuDTChambThContainer);
  
  if (!myL1MuDTChambThContainer.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1MuDTChambThContainer with label "
			     << dttpgSource_.label() ;
    edm::LogInfo("DataNotFound") << "if this fails try to add DATA to the process name." ;

    return;
  }

  L1MuDTChambThContainer::The_Container* myThContainer =  
    myL1MuDTChambThContainer->getContainer();

  int ndttpgphtrack = 0;
  int ndttpgthtrack = 0; 
  int NumberOfSegmentsPhi[3]={0,0,0};
  
  for( L1MuDTChambPhContainer::Phi_Container::const_iterator 
	 DTPhDigiItr =  myPhContainer->begin() ;
       DTPhDigiItr != myPhContainer->end() ;
       ++DTPhDigiItr ) 
    {        
      int bx = DTPhDigiItr->bxNum() - DTPhDigiItr->Ts2Tag();
      int wh = DTPhDigiItr->whNum();
      int se = DTPhDigiItr->scNum();
      int st = DTPhDigiItr->stNum();

      bxnumber[wh+2][se][st] -> Fill(bx);
      
      if(bx == -1)
       NumberOfSegmentsPhi[0]++;
      if(bx == 0)
       NumberOfSegmentsPhi[1]++;
      if(bx == 1)
       NumberOfSegmentsPhi[2]++;   
    }
   
  //Fill Histos for Segment counter for each bx separately 
  
  for(int k=0;k<3;k++){
    dttpgphbx[k+2]->Fill(NumberOfSegmentsPhi[k]);
  }

  int bxCounterDttfPhi=0; // = no. of BX's with non-zero data
  for (int k=0;k<3;k++){
    if (NumberOfSegmentsPhi[k]>0)
      bxCounterDttfPhi++;
  }

  // the BX "code" 
  
  int bxCodePhi=0;
  if(bxCounterDttfPhi==0){
    bxCodePhi=0;
  }else if(bxCounterDttfPhi==1){
    for(int k=0;k<3;k++){
      if(NumberOfSegmentsPhi[k]>0)
        bxCodePhi=k+2;
    }
  }else if(bxCounterDttfPhi==2){
    for(int k=0;k<3;k++){
      if(NumberOfSegmentsPhi[k]==0)
        bxCodePhi=8-k;
    }
  }else if(bxCounterDttfPhi==3){
    bxCodePhi=10;
  }

  //The bx analyzer histo
  dttpgphbx[0]->Fill(bxCodePhi);


  const L1MuDTChambPhDigi* bestPhQualMap[5][12][4];
  memset(bestPhQualMap,0,240*sizeof(L1MuDTChambPhDigi*));


  for( L1MuDTChambPhContainer::Phi_Container::const_iterator 
	 DTPhDigiItr =  myPhContainer->begin() ;
       DTPhDigiItr != myPhContainer->end() ;
       ++DTPhDigiItr ) 
    {           

      ndttpgphtrack++;
      
      int bxindex = DTPhDigiItr->bxNum() - DTPhDigiItr->Ts2Tag() + 1;
      int wh = DTPhDigiItr->whNum();
      int se = DTPhDigiItr->scNum();
      int st = DTPhDigiItr->stNum();

      dttpgphwheel[bxindex]->Fill(wh);
      if (verbose_)
	{
	  cout << "DTTPG phi wheel number " << DTPhDigiItr->whNum() << endl;
	}
      dttpgphsector[bxindex][wh+2]->Fill(se);
      if (verbose_)
	{   
	  cout << "DTTPG phi sector number " << DTPhDigiItr->scNum() << endl;
	}
      dttpgphstation[bxindex][wh+2][se]->Fill(st);
      if (verbose_)
	{
	  cout << "DTTPG phi sstation number " << DTPhDigiItr->stNum() << endl;
	}
	
      if(DTPhDigiItr->Ts2Tag()==0) {	
       dttpgphsg1phiAngle[bxindex][wh+2][se][st] -> Fill(DTPhDigiItr->phi());
       if (verbose_)
 	{
 	  cout << "DTTPG phi phi " << DTPhDigiItr->phi() << endl;
 	}
      dttpgphsg1phiBandingAngle[bxindex][wh+2][se][st] -> Fill(DTPhDigiItr->phiB());
       if (verbose_)
 	{
 	  cout << "DTTPG phi phiB " << DTPhDigiItr->phiB() << endl;
 	}
      dttpgphsg1quality[bxindex][wh+2][se][st] -> Fill(DTPhDigiItr->code());
       if (verbose_)
 	{
 	  cout << "DTTPG phi quality " << DTPhDigiItr->code() << endl;
 	}
      } else if(DTPhDigiItr->Ts2Tag()==1) {
      dttpgphsg2phiAngle[bxindex][wh+2][se][st] -> Fill(DTPhDigiItr->phi());
       if (verbose_)
 	{
 	  cout << "DTTPG phi phi " << DTPhDigiItr->phi() << endl;
 	}
      dttpgphsg2phiBandingAngle[bxindex][wh+2][se][st] -> Fill(DTPhDigiItr->phiB());
       if (verbose_)
 	{
 	  cout << "DTTPG phi phiB " << DTPhDigiItr->phiB() << endl;
 	}
      dttpgphsg2quality[bxindex][wh+2][se][st] -> Fill(DTPhDigiItr->code());
       if (verbose_)
 	{
 	  cout << "DTTPG phi quality " << DTPhDigiItr->code() << endl;
 	}

      dttpgphts2tag[bxindex][wh+2][se][st] -> Fill(DTPhDigiItr->Ts2Tag());
       if (verbose_)
 	{
 	  cout << "DTTPG phi ts2tag " << DTPhDigiItr->Ts2Tag() << endl;
 	}
     }
      
      int ypos = DTPhDigiItr->scNum();
      int xpos = DTPhDigiItr->stNum()+4*(DTPhDigiItr->whNum()+2);
      dttpgphmap->Fill(xpos,ypos);
      if (DTPhDigiItr->Ts2Tag())
	dttpgphmap2nd->Fill(xpos,ypos);
      dttpgphmapbx[bxindex]->Fill(xpos,ypos);
      if (DTPhDigiItr->code()>3)
	dttpgphmapcorr->Fill(xpos,ypos);
      
      if (bestPhQualMap[DTPhDigiItr->whNum()+2][ DTPhDigiItr->scNum()][DTPhDigiItr->stNum()-1]==0 ||
	  bestPhQualMap[DTPhDigiItr->whNum()+2][ DTPhDigiItr->scNum()][DTPhDigiItr->stNum()-1]->code()<DTPhDigiItr->code())
	{
	  bestPhQualMap[DTPhDigiItr->whNum()+2][ DTPhDigiItr->scNum()][DTPhDigiItr->stNum()-1]=&(*DTPhDigiItr);
	}
      
    }

  for (int iwh=0; iwh<5; iwh++){
    for (int isec=0; isec<12; isec++){
      for (int ist=0; ist<4; ist++){
	if (bestPhQualMap[iwh][isec][ist]){
	  int xpos = iwh*4+ist+1;
	  dttpgphbestmap->Fill(xpos,isec);
	  if(bestPhQualMap[iwh][isec][ist]->code()>3)
	    dttpgphbestmapcorr->Fill(xpos,isec);
	}
      }
    }
  }
  

  int bestThQualMap[5][12][3];
  memset(bestThQualMap,0,180*sizeof(int));
  //for( vector<L1MuDTChambThDigi>::const_iterator 
  for( L1MuDTChambThContainer::The_Container::const_iterator 
	 DTThDigiItr =  myThContainer->begin() ;
       DTThDigiItr != myThContainer->end() ;
       ++DTThDigiItr ) 
    {           		
      ndttpgthtrack++;
      
      int bxindex = DTThDigiItr->bxNum() + 1;
      int wh = DTThDigiItr->whNum();
      int se = DTThDigiItr->scNum();
      int st = DTThDigiItr->stNum();

      dttpgthwheel[bxindex]->Fill(wh);
      if (verbose_)
	{
	  cout << "DTTPG theta wheel number " << DTThDigiItr->whNum() << endl;
	}
      dttpgthsector[bxindex][wh+2]->Fill(se);
      if (verbose_)
	{
	  cout << "DTTPG theta sector number " << DTThDigiItr->scNum() << endl;
	}
      dttpgthstation[bxindex][wh+2][se]->Fill(st);
      if (verbose_)
	{   
	  cout << "DTTPG theta station number " << DTThDigiItr->stNum() << endl;
	}
      dttpgthbx[bxindex]->Fill(DTThDigiItr->bxNum());
      if (verbose_)
	{
	  cout << "DTTPG theta bx number " << DTThDigiItr->bxNum() << endl;
	}
      int thcode[7]= {0,0,0,0,0,0,0};
      for (int j = 0; j < 7; j++)
	{
	  dttpgththeta[bxindex][wh+2][se][st]->Fill(DTThDigiItr->position(j));
	  if (verbose_)
	    {
	      cout << "DTTPG theta position " << DTThDigiItr->position(j) << endl;
	    }
	  thcode[j]=DTThDigiItr->code(j);
	  dttpgthquality[bxindex][wh+2][se][st]->Fill(thcode[j]);
	  if (verbose_)
	    {
	      cout << "DTTPG theta quality " << DTThDigiItr->code(j) << endl;
	    }
	}
      
      int ypos = DTThDigiItr->scNum();
      int xpos = DTThDigiItr->stNum()+4*(DTThDigiItr->whNum()+2);
      int bestqual=0;
      dttpgthmap->Fill(xpos,ypos);
      dttpgthmapbx[bxindex]->Fill(xpos,ypos);
      for (int pos = 0; pos < 7; pos++){
	if (thcode[pos]>bestqual)
	  bestqual=thcode[pos];
	if(thcode[pos]==2)
	  dttpgthmaph->Fill(xpos,ypos);
      }

      if (bestThQualMap[DTThDigiItr->whNum()+2][ DTThDigiItr->scNum()][DTThDigiItr->stNum()-1] < bestqual)
	{
	  bestThQualMap[DTThDigiItr->whNum()+2][ DTThDigiItr->scNum()][DTThDigiItr->stNum()-1]=bestqual;
	}
    }
  
  for (int iwh=0; iwh<5; iwh++){
    for (int isec=0; isec<12; isec++){
      for (int ist=0; ist<3; ist++){
	if (bestThQualMap[iwh][isec][ist]){
	  int xpos = iwh*4+ist+1;
	  dttpgthbestmap->Fill(xpos,isec);
	  if(bestThQualMap[iwh][isec][ist]==2)
	    dttpgthbestmaph->Fill(xpos,isec);
	}
      }
    }
  }


  dttpgphntrack->Fill(ndttpgphtrack);
  if (verbose_)
    {
      cout << "DTTPG phi ntrack " << ndttpgphtrack << endl;
    }
  dttpgthntrack->Fill(ndttpgthtrack);
  if (verbose_) {
    cout << "DTTPG theta ntrack " << ndttpgthtrack << endl;
  }
///
*/


  
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
  //int count = 0;

  /*
///
  int NumberOfSegmentsOut[3]={0,0,0};
  for ( L1MuDTTrackContainer::TrackContainer::const_iterator i 
	  = t->begin(); i != t->end(); ++i ) {
    if(i->bx() ==-1)
      NumberOfSegmentsOut[0]++;
    if(i->bx() ==0)
      NumberOfSegmentsOut[1]++;
    if(i->bx() ==1)
      NumberOfSegmentsOut[2]++;
  }
  

// Fill Histos for Segment counter
  for(int k=0;k<3;k++){
    dttpgphbx[k+5]->Fill(NumberOfSegmentsOut[k]);
  }

// Bunch assigments

  int bxCounterDttfOut=0;
  for (int k=0;k<3;k++){
    if (NumberOfSegmentsOut[k]>0)
      bxCounterDttfOut++;
  }

  int bxCodeOut=0;
  if(bxCounterDttfOut==0){
    bxCodeOut=0;
  }else if(bxCounterDttfOut==1){
    for(int k=0;k<3;k++){
      if(NumberOfSegmentsOut[k]>0)
	bxCodeOut=k+2;
    }
  }else if(bxCounterDttfOut==2){
    for(int k=0;k<3;k++){
      if(NumberOfSegmentsOut[k]==0)
	bxCodeOut=8-k;
    }
  }else if(bxCounterDttfOut==3){
    bxCodeOut=10;
  }
  
  //The bx analyzer histo
  dttpgphbx[1]->Fill(bxCodeOut);

//   End Dttf Output Bunch analysis

   // the 2-DIM histo with phi.input vs. output
//   dttpgphbxcomp->Fill(bxCodePhi,bxCodeOut);
///
*/





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
    //count++;
    int bxindex = i->bx() + 1;
    int wh = i->whNum();//wh has possible values {-3,-2,-1,1,2,3}
    int se = i->scNum();
    
    int wh2;//make wh2 go from 0 to 5
    if(wh<0)wh2=wh+3;
    else wh2=wh+2;
    
    //cout << "bxindex " << bxindex << endl;
    //std::cout << "whNum = " << wh << "  bxindex = " << bxindex << std::endl;
    //std::cout << "wh2 = " << wh2 << std::endl;
    //std::cout << "se = " << se << std::endl;
    //std::cout << "eta_packed = " << i->eta_packed() << std::endl;
    //std::cout << "count = " << count << std::endl;


    int phi_local = i->phi_packed();//range: 0 < phi_local < 31 
    if(phi_local > 15) phi_local -= 32; //range: -16 < phi_local < 15
    
    int phi_global = phi_local + se*12; //range: -16 < phi_global < 147
    if(phi_global < 0) phi_global = 144; //range: 0 < phi_global < 147
    if(phi_global > 143) phi_global -= 144; //range: 0 < phi_global < 143

    float phi_phys = phi_global * 2.5;  

    dttf_p_phi[bxindex][wh2][se]->Fill(phi_global);

    dttf_p_qual[bxindex][wh2][se]->Fill(i->quality_packed());
    dttf_p_pt[bxindex][wh2][se]->Fill(i->pt_packed());
    dttf_p_q[bxindex][wh2][se]->Fill(i->charge_packed());

    int eta_global = i->eta_packed() - 32;
    dttf_p_eta[bxindex][wh2][se]->Fill(eta_global);

    float eta_phys = eta_global / 2.4 ;

    if(bxindex==1){
      //dttf_p_phi_integ->Fill(i->phi_packed());
      dttf_p_phi_integ->Fill(phi_global);
      dttf_p_pt_integ->Fill(i->pt_packed());
      dttf_p_eta_integ->Fill(eta_global);
      dttf_p_qual_integ->Fill(i->quality_packed());


      //dttf_phys_phi_integ->Fill(i->phiValue());
      dttf_phys_phi_integ->Fill(phi_phys);
      dttf_phys_pt_integ->Fill(i->ptValue());
      //dttf_phys_eta_integ->Fill(i->etaValue());
      dttf_phys_eta_integ->Fill(eta_phys);
      
      dttf_p_phi_eta[wh2]->Fill(i->phi_packed(),i->eta_packed());
      dttf_p_phi_eta_integ->Fill(i->phi_packed(),i->eta_packed());
      
      //dttf_occupancySummary->Fill(se+1,wh2);
      dttf_occupancySummary->Fill(wh2,se+1);
    }
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
