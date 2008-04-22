/*
 * \file L1TDTTPG.cc
 *
 * $Date: 2008/01/22 18:56:01 $
 * $Revision: 1.15 $
 * \author J. Berryhill
 *
 * $Log: L1TDTTPG.cc,v $
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
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/L1TMonitor/interface/L1TDTTPG.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace std;
using namespace edm;

L1TDTTPG::L1TDTTPG(const ParameterSet& ps)
  : dttpgSource_( ps.getParameter< InputTag >("dttpgSource") )
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TDTTPG: constructor...." << endl;

  logFile_.open("L1TDTTPG.log");

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
  else{
    outputFile_ = "L1TDQM.root";
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }


  if ( dbe !=NULL ) {
    dbe->setCurrentFolder("L1T/L1TDTTPG");
  }


}

L1TDTTPG::~L1TDTTPG()
{
}

void L1TDTTPG::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1T/L1TDTTPG");
    dbe->rmdir("L1T/L1TDTTPG");
  }


  if ( dbe ) 
    {
      dbe->setCurrentFolder("L1T/L1TDTTPG");


   //hist1[0]
   dttpgphbx[0] = dbe->book1D("BxEncoding_PHI",
                      "Bunch encoding DTTF Phi",11,0,11);
   //hist1[1]
   dttpgphbx[1] = dbe->book1D("BxEncoding_OUT",
                      "Bunch encoding DTTF Output",11,0,11);


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

      dttpgphntrack = dbe->book1D("DT_TPG_phi_ntrack", 
				  "DT TPG phi ntrack", 20, -0.5, 19.5 ) ;  
      dttpgthntrack = dbe->book1D("DT_TPG_theta_ntrack", 
				  "DT TPG theta ntrack", 20, -0.5, 19.5 ) ;  


      dttpgphwheel[0] = dbe->book1D("DT_TPG_phi_wheel_number_-1", 
				 "DT TPG phi wheel number -1", 5, -2.5, 2.5 ) ;  
      dttpgphsector[0] = dbe->book1D("DT_TPG_phi_sector_number_-1", 
				     "DT TPG phi sector number -1", 12, -0.5, 11.5 );  
      dttpgphstation[0] = dbe->book1D("DT_TPG_phi_station_number_-1", 
				   "DT TPG phi station number -1", 5, 0.5, 4.5 ) ;
      dttpgphphi[0] = dbe->book1D("DT_TPG_phi_-1", 
			       "DT TPG phi -1", 100, -2100., 2100. ) ;  
      dttpgphphiB[0] = dbe->book1D("DT_TPG_phiB_-1", 
				"DT TPG phiB -1", 100, -550., 550. ) ;  
      dttpgphquality[0] = dbe->book1D("DT_TPG_phi_quality_-1", 
				   "DT TPG phi quality -1", 8, -0.5, 7.5 ) ;  
      dttpgphts2tag[0] = dbe->book1D("DT_TPG_phi_Ts2Tag_-1", 
				  "DT TPG phi Ts2Tag -1", 2, -0.5, 1.5 ) ;  
      dttpgphbxcnt[0] = dbe->book1D("DT_TPG_phi_BxCnt_-1", 
				 "DT TPG phi BxCnt -1", 10, -0.5, 9.5 ) ;  

      dttpgthbx[0] = dbe->book1D("DT_TPG_theta_bx_-1", 
			      "DT TPG theta bx -1", 50, -24.5, 24.5 ) ;  

      dttpgthwheel[0] = dbe->book1D("DT_TPG_theta_wheel_number_-1", 
				 "DT TPG theta wheel number -1", 5, -2.5, 2.5 ) ;  
      dttpgthsector[0] = dbe->book1D("DT_TPG_theta_sector_number_-1", 
				  "DT TPG theta sector number -1", 12, -0.5, 11.5 ) ;  
      dttpgthstation[0] = dbe->book1D("DT_TPG_theta_station_number_-1", 
				   "DT TPG theta station number -1", 5, -0.5, 4.5 ) ;  
      dttpgththeta[0] = dbe->book1D("DT_TPG_theta_-1", 
				 "DT TPG theta -1", 20, -0.5, 19.5 ) ;  
      dttpgthquality[0] = dbe->book1D("DT_TPG_theta_quality_-1", 
				   "DT TPG theta quality -1", 8, -0.5, 7.5 ) ;  
      // Phi output
      dttf_p_phi[0] = dbe->book1D("dttf_p_phi_-1", "dttf phi output #phi -1", 256, 
			      -0.5, 255.5);
      dttf_p_qual[0] = dbe->book1D("dttf_p_qual_-1", "dttf phi output qual -1", 8, -0.5, 7.5);
      dttf_p_q[0] = dbe->book1D("dttf_p_q_-1", "dttf phi output q -1", 2, -0.5, 1.5);
      dttf_p_pt[0] = dbe->book1D("dttf_p_pt_-1", "dttf phi output p_{t} -1", 32, -0.5, 31.5);




      dttpgphwheel[1] = dbe->book1D("DT_TPG_phi_wheel_number_0", 
				 "DT TPG phi wheel number 0", 5, -2.5, 2.5 ) ;  
      dttpgphsector[1] = dbe->book1D("DT_TPG_phi_sector_number_0", 
				     "DT TPG phi sector number 0", 12, -0.5, 11.5 );  
      dttpgphstation[1] = dbe->book1D("DT_TPG_phi_station_number_0", 
				   "DT TPG phi station number 0", 5, 0.5, 4.5 ) ;
      dttpgphphi[1] = dbe->book1D("DT_TPG_phi_0", 
			       "DT TPG phi 0", 100, -2100., 2100. ) ;  
      dttpgphphiB[1] = dbe->book1D("DT_TPG_phiB_0", 
				"DT TPG phiB 0", 100, -550., 550. ) ;  
      dttpgphquality[1] = dbe->book1D("DT_TPG_phi_quality_0", 
				   "DT TPG phi quality 0", 8, -0.5, 7.5 ) ;  
      dttpgphts2tag[1] = dbe->book1D("DT_TPG_phi_Ts2Tag_0", 
				  "DT TPG phi Ts2Tag 0", 2, -0.5, 1.5 ) ;  
      dttpgphbxcnt[1] = dbe->book1D("DT_TPG_phi_BxCnt_0", 
				 "DT TPG phi BxCnt 0", 10, -0.5, 9.5 ) ;  

      dttpgthbx[1] = dbe->book1D("DT_TPG_theta_bx_0", 
			      "DT TPG theta bx 0", 50, -24.5, 24.5 ) ;  
      dttpgthwheel[1] = dbe->book1D("DT_TPG_theta_wheel_number_0", 
				 "DT TPG theta wheel number 0", 5, -2.5, 2.5 ) ;  
      dttpgthsector[1] = dbe->book1D("DT_TPG_theta_sector_number_0", 
				  "DT TPG theta sector number 0", 12, -0.5, 11.5 ) ;  
      dttpgthstation[1] = dbe->book1D("DT_TPG_theta_station_number_0", 
				   "DT TPG theta station number 0", 5, -0.5, 4.5 ) ;  
      dttpgththeta[1] = dbe->book1D("DT_TPG_theta_0", 
				 "DT TPG theta 0", 20, -0.5, 19.5 ) ;  
      dttpgthquality[1] = dbe->book1D("DT_TPG_theta_quality_0", 
				   "DT TPG theta quality 0", 8, -0.5, 7.5 ) ;  
      // Phi output
      dttf_p_phi[1] = dbe->book1D("dttf_p_phi_0", "dttf phi output #phi 0", 256, 
			      -0.5, 255.5);
      dttf_p_qual[1] = dbe->book1D("dttf_p_qual_0", "dttf phi output qual 0", 8, -0.5, 7.5);
      dttf_p_q[1] = dbe->book1D("dttf_p_q_0", "dttf phi output q 0", 2, -0.5, 1.5);
      dttf_p_pt[1] = dbe->book1D("dttf_p_pt_0", "dttf phi output p_{t} 0", 32, -0.5, 31.5);
    


      dttpgphwheel[2] = dbe->book1D("DT_TPG_phi_wheel_number_+1", 
				 "DT TPG phi wheel number +1", 5, -2.5, 2.5 ) ;  
      dttpgphsector[2] = dbe->book1D("DT_TPG_phi_sector_number_+1", 
				     "DT TPG phi sector number +1", 12, -0.5, 11.5 );  
      dttpgphstation[2] = dbe->book1D("DT_TPG_phi_station_number_+1", 
				   "DT TPG phi station number +1", 5, 0.5, 4.5 ) ;
      dttpgphphi[2] = dbe->book1D("DT_TPG_phi_+1", 
			       "DT TPG phi +1", 100, -2100., 2100. ) ;  
      dttpgphphiB[2] = dbe->book1D("DT_TPG_phiB_+1", 
				"DT TPG phiB +1", 100, -550., 550. ) ;  
      dttpgphquality[2] = dbe->book1D("DT_TPG_phi_quality_+1", 
				   "DT TPG phi quality +1", 8, -0.5, 7.5 ) ;  
      dttpgphts2tag[2] = dbe->book1D("DT_TPG_phi_Ts2Tag_+1", 
				  "DT TPG phi Ts2Tag +1", 2, -0.5, 1.5 ) ;  
      dttpgphbxcnt[2] = dbe->book1D("DT_TPG_phi_BxCnt_+1", 
				 "DT TPG phi BxCnt +1", 10, -0.5, 9.5 ) ;  

      dttpgthbx[2] = dbe->book1D("DT_TPG_theta_bx_+1", 
			      "DT TPG theta bx +1", 50, -24.5, 24.5 ) ;  

      dttpgthwheel[2] = dbe->book1D("DT_TPG_theta_wheel_number_+1", 
				 "DT TPG theta wheel number +1", 5, -2.5, 2.5 ) ;  
      dttpgthsector[2] = dbe->book1D("DT_TPG_theta_sector_number_+1", 
				  "DT TPG theta sector number +1", 12, -0.5, 11.5 ) ;  
      dttpgthstation[2] = dbe->book1D("DT_TPG_theta_station_number_+1", 
				   "DT TPG theta station number +1", 5, -0.5, 4.5 ) ;  
      dttpgththeta[2] = dbe->book1D("DT_TPG_theta_+1", 
				 "DT TPG theta +1", 20, -0.5, 19.5 ) ;  
      dttpgthquality[2] = dbe->book1D("DT_TPG_theta_quality_+1", 
				   "DT TPG theta quality +1", 8, -0.5, 7.5 ) ;  
      // Phi output
      dttf_p_phi[2] = dbe->book1D("dttf_p_phi_+1", "dttf phi output #phi +1", 256, 
			      -0.5, 255.5);
      dttf_p_qual[2] = dbe->book1D("dttf_p_qual_+1", "dttf phi output qual +1", 8, -0.5, 7.5);
      dttf_p_q[2] = dbe->book1D("dttf_p_q_+1", "dttf phi output q +1", 2, -0.5, 1.5);
      dttf_p_pt[2] = dbe->book1D("dttf_p_pt_+1", "dttf phi output p_{t} +1", 32, -0.5, 31.5);




    }  
}


void L1TDTTPG::endJob(void)
{
  if(verbose_) cout << "L1TDTTPG: end job...." << endl;
  LogInfo("L1TDTTPG") << "analyzed " << nev_ << " events"; 

  if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

  return;
}

void L1TDTTPG::analyze(const Event& e, const EventSetup& c)
{

  nev_++; 
  if(verbose_) cout << "L1TDTTPG: analyze...." << endl;

  edm::Handle<L1MuDTChambPhContainer > myL1MuDTChambPhContainer;  
  e.getByLabel(dttpgSource_,myL1MuDTChambPhContainer);
  
  if (!myL1MuDTChambPhContainer.isValid()) {
    edm::LogInfo("L1TDTTPG") << "can't find L1MuDTChambPhContainer with label "
			     << dttpgSource_.label() ;
    return;
  }
  L1MuDTChambPhContainer::Phi_Container *myPhContainer =  
    myL1MuDTChambPhContainer->getContainer();

  edm::Handle<L1MuDTChambThContainer > myL1MuDTChambThContainer;  
  e.getByLabel(dttpgSource_,myL1MuDTChambThContainer);
  
  if (!myL1MuDTChambThContainer.isValid()) {
    edm::LogInfo("L1TDTTPG") << "can't find L1MuDTChambThContainer with label "
			     << dttpgSource_.label() ;
    edm::LogInfo("L1TDTTPG") << "if this fails try to add DATA to the process name." ;

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
      if(bx == -1)
       NumberOfSegmentsPhi[0]++;
      if(bx == 0)
       NumberOfSegmentsPhi[1]++;
      if(bx == 1)
       NumberOfSegmentsPhi[2]++;   
    }
   /*Fill Histos for Segment counter for each bx separately */

  for(int k=0;k<3;k++){
     dttpgphbx[k+2]->Fill(NumberOfSegmentsPhi[k]);
   }
   int bxCounterDttfPhi=0; // = no. of BX's with non-zero data
   for (int k=0;k<3;k++){
     if (NumberOfSegmentsPhi[k]>0)
       bxCounterDttfPhi++;
   }

   /* the BX "code" */

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


 for( L1MuDTChambPhContainer::Phi_Container::const_iterator 
	 DTPhDigiItr =  myPhContainer->begin() ;
       DTPhDigiItr != myPhContainer->end() ;
       ++DTPhDigiItr ) 
    {           

      ndttpgphtrack++;

      int bxindex = DTPhDigiItr->bxNum() - DTPhDigiItr->Ts2Tag() + 1;

      dttpgphwheel[bxindex]->Fill(DTPhDigiItr->whNum());
      if (verbose_)
	{
	  cout << "DTTPG phi wheel number " << DTPhDigiItr->whNum() << endl;
	}
      dttpgphstation[bxindex]->Fill(DTPhDigiItr->stNum());
      if (verbose_)
	{   
	  cout << "DTTPG phi station number " << DTPhDigiItr->stNum() << endl;
	}
      dttpgphsector[bxindex]->Fill(DTPhDigiItr->scNum());
      if (verbose_)
	{
	  cout << "DTTPG phi sector number " << DTPhDigiItr->scNum() << endl;
	}
      dttpgphphi[bxindex]->Fill(DTPhDigiItr->phi());
      if (verbose_)
	{
	  cout << "DTTPG phi phi " << DTPhDigiItr->phi() << endl;
	}
      dttpgphphiB[bxindex]->Fill(DTPhDigiItr->phiB());
      if (verbose_)
	{
	  cout << "DTTPG phi phiB " << DTPhDigiItr->phiB() << endl;
	}
      dttpgphquality[bxindex]->Fill(DTPhDigiItr->code());
      if (verbose_)
	{
	  cout << "DTTPG phi quality " << DTPhDigiItr->code() << endl;
	}
      dttpgphts2tag[bxindex]->Fill(DTPhDigiItr->Ts2Tag());
      if (verbose_)
	{
	  cout << "DTTPG phi ts2tag " << DTPhDigiItr->Ts2Tag() << endl;
	}
      dttpgphbxcnt[bxindex]->Fill(DTPhDigiItr->BxCnt());
      if (verbose_)
	{
	  cout << "DTTPG phi bxcnt " << DTPhDigiItr->BxCnt() << endl;
	}
    }

  //for( vector<L1MuDTChambThDigi>::const_iterator 
  for( L1MuDTChambThContainer::The_Container::const_iterator 
	 DTThDigiItr =  myThContainer->begin() ;
       DTThDigiItr != myThContainer->end() ;
       ++DTThDigiItr ) 
    {           		
      ndttpgthtrack++;

      int bxindex = DTThDigiItr->bxNum() + 1;

      dttpgthwheel[bxindex]->Fill(DTThDigiItr->whNum());
      if (verbose_)
	{
	  cout << "DTTPG theta wheel number " << DTThDigiItr->whNum() << endl;
	}
      dttpgthstation[bxindex]->Fill(DTThDigiItr->stNum());
      if (verbose_)
	{   
	  cout << "DTTPG theta station number " << DTThDigiItr->stNum() << endl;
	}
      dttpgthsector[bxindex]->Fill(DTThDigiItr->scNum());
      if (verbose_)
	{
	  cout << "DTTPG theta sector number " << DTThDigiItr->scNum() << endl;
	}
      dttpgthbx[bxindex]->Fill(DTThDigiItr->bxNum());
      if (verbose_)
	{
	  cout << "DTTPG theta bx number " << DTThDigiItr->bxNum() << endl;
	}
      for (int j = 0; j < 7; j++)
	{
	  dttpgththeta[bxindex]->Fill(DTThDigiItr->position(j));
	  if (verbose_)
	    {
	      cout << "DTTPG theta position " << DTThDigiItr->position(j) << endl;
	    }
	  dttpgthquality[bxindex]->Fill(DTThDigiItr->code(j));
	  if (verbose_)
	    {
	      cout << "DTTPG theta quality " << DTThDigiItr->code(j) << endl;
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

  edm::Handle<L1MuDTTrackContainer > myL1MuDTTrackContainer;

  
    std::string trstring;
    trstring = dttpgSource_.label()+":"+"DATA"+":"+dttpgSource_.process();
    edm::InputTag trInputTag(trstring);
    e.getByLabel(trInputTag,myL1MuDTTrackContainer);

  if (!myL1MuDTTrackContainer.isValid()) {
    edm::LogInfo("L1TDTTPG") << "can't find L1MuDTTrackContainer with label "
                               << dttpgSource_.label() ;
    return;
  }

  L1MuDTTrackContainer::TrackContainer *t =  myL1MuDTTrackContainer->getContainer();



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


   /*Fill Histos for Segment counter*/
   for(int k=0;k<3;k++){
     dttpgphbx[k+5]->Fill(NumberOfSegmentsOut[k]);
   }

   /*Bunch assigments*/

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

   /*End Dttf Output Bunch analysis*/

   // the 2-DIM histo with phi.input vs. output
   dttpgphbxcomp->Fill(bxCodePhi,bxCodeOut);


  for ( L1MuDTTrackContainer::TrackContainer::const_iterator i 
	  = t->begin(); i != t->end(); ++i ) {
    if ( verbose_ ) {
      std::cout << "bx = " << i->bx() 
		<< std::endl;
      std::cout << "quality (packed) = " << i->quality_packed() 
		<< std::endl;
      std::cout << "pt      (packed) = " << i->pt_packed() 
		<< std::endl;
      std::cout << "phi     (packed) = " << i->phi_packed() 
		<< std::endl;
      std::cout << "charge  (packed) = " << i->charge_packed() 
		<< std::endl;
    }


    int bxindex = i->bx() + 1;
    dttf_p_phi[bxindex]->Fill(i->phi_packed());
    dttf_p_qual[bxindex]->Fill(i->quality_packed());
    dttf_p_pt[bxindex]->Fill(i->pt_packed());
    dttf_p_q[bxindex]->Fill(i->charge_packed());
  }
    
}


