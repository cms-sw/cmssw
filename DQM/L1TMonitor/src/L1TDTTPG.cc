/*
 * \file L1TDTTPG.cc
 *
 * \author J. Berryhill
 *
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

#include "DQM/L1TMonitor/interface/L1TDTTPG.h"

using namespace std;
using namespace edm;

L1TDTTPG::L1TDTTPG(const ParameterSet& ps)
  : dttpgSourcePhContainer_token_( consumes<L1MuDTChambPhContainer>(ps.getParameter< InputTag >("dttpgSource") )),
    dttpgSourceThContainer_token_( consumes<L1MuDTChambThContainer>(ps.getParameter< InputTag >("dttpgSource") )),
    dttpgSource_( ps.getParameter< InputTag >("dttpgSource") )
{
  trstring_ = dttpgSource_.label()+":"+"DATA"+":"+dttpgSource_.process();
  trToken_ = consumes<L1MuDTTrackContainer>(trstring_);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TDTTPG: constructor...." << endl;

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }
}

L1TDTTPG::~L1TDTTPG()
{
}


void L1TDTTPG::dqmBeginRun(const edm::Run &r, const edm::EventSetup &c){
  //empty
}

void L1TDTTPG::beginLuminosityBlock(const edm::LuminosityBlock &l, const edm::EventSetup &c){
  //empty
}

void L1TDTTPG::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&) 
{
  nev_ = 0;
  ibooker.setCurrentFolder("L1T/L1TDTTPG");

  //hist1[0]
  dttpgphbx[0] = ibooker.book1D("BxEncoding_PHI",
				 "Bunch encoding DTTF Phi",11,0,11);
  //hist1[1]
  dttpgphbx[1] = ibooker.book1D("BxEncoding_OUT",
				 "Bunch encoding DTTF Output",11,0,11);

      //hist1[2]
  dttpgphbx[2] = ibooker.book1D("NumberOfSegmentsPHI_BunchNeg1",
				 "Number of segments for bunch -1 Dttf Phi",
				 20,0,20);
      //hist1[3]
  dttpgphbx[3] = ibooker.book1D("NumberOfSegmentsPHI_Bunch0",
				 "Number of segments for bunch 0 Dttf Phi",
				 20,0,20);
      //hist1[4]
  dttpgphbx[4] = ibooker.book1D("NumberOfSegmentsPHI_Bunch1",
				 "Number of segments for bunch 1 Dttf Phi",
				 20,0,20);

      //hist1[5]
  dttpgphbx[5] = ibooker.book1D("NumberOfSegmentsOUT_BunchNeg1",
				 "Number of segments for bunch -1 Dttf Output",
				 20,0,20);
      //hist1[6] 
  dttpgphbx[6] = ibooker.book1D("NumberOfSegmentsOUT_Bunch0",
				 "Number of segments for bunch 0 Dttf Output",
				 20,0,20);
      //hist1[7]
  dttpgphbx[7] = ibooker.book1D("NumberOfSegmentsOUT_Bunch1",
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
   
  dttpgphbxcomp = ibooker.book2D("BxEncoding_PHI_OUT",
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

  dttpgphntrack = ibooker.book1D("DT_TPG_phi_ntrack", 
				  "DT TPG phi ntrack", 20, -0.5, 19.5 ) ;  
  dttpgthntrack = ibooker.book1D("DT_TPG_theta_ntrack", 
				  "DT TPG theta ntrack", 20, -0.5, 19.5 ) ;  

  for (int ibx=0 ; ibx<=2; ibx++) {
	
    ostringstream bxnum;
    bxnum << ibx-1;
    string bxn;
    if (ibx<2)
      bxn = bxnum.str();
    else
      bxn = "+" + bxnum.str();
	
    // Phi
    dttpgphwheel[ibx] = ibooker.book1D("DT_TPG_phi_wheel_number_"+bxn, 
					    "DT TPG phi wheel number "+bxn, 5, -2.5, 2.5 ) ;  
    dttpgphsector[ibx] = ibooker.book1D("DT_TPG_phi_sector_number_"+bxn, 
					 "DT TPG phi sector number "+bxn, 12, -0.5, 11.5 );  
    dttpgphstation[ibx] = ibooker.book1D("DT_TPG_phi_station_number_"+bxn, 
					  "DT TPG phi station number "+bxn, 5, 0.5, 4.5 ) ;
// 	dttpgphphi[ibx] = dbe->book1D("DT_TPG_phi_"+bxn, 
// 				      "DT TPG phi "+bxn, 100, -2100., 2100. ) ;  
// 	dttpgphphiB[ibx] = dbe->book1D("DT_TPG_phiB_"+bxn, 
// 				       "DT TPG phiB "+bxn, 100, -550., 550. ) ;  
    dttpgphquality[ibx] = ibooker.book1D("DT_TPG_phi_quality_"+bxn, 
					  "DT TPG phi quality "+bxn, 8, -0.5, 7.5 ) ;  
    dttpgphts2tag[ibx] = ibooker.book1D("DT_TPG_phi_Ts2Tag_"+bxn, 
					 "DT TPG phi Ts2Tag "+bxn, 2, -0.5, 1.5 ) ;  
// 	dttpgphbxcnt[ibx] = dbe->book1D("DT_TPG_phi_BxCnt_"+bxn, 
// 					"DT TPG phi BxCnt "+bxn, 10, -0.5, 9.5 ) ;  
    dttpgphmapbx[ibx] = ibooker.book2D("DT_TPG_phi_map_bx"+bxn,
				      "Map of triggers per station (BX="+bxn+")",20,1,21,12,0,12);
    setMapPhLabel(dttpgphmapbx[ibx]);

    //Theta
    dttpgthbx[ibx] = ibooker.book1D("DT_TPG_theta_bx_"+bxn, 
				     "DT TPG theta bx "+bxn, 50, -24.5, 24.5 ) ;  
    dttpgthwheel[ibx] = ibooker.book1D("DT_TPG_theta_wheel_number_"+bxn, 
					"DT TPG theta wheel number "+bxn, 5, -2.5, 2.5 ) ;  
    dttpgthsector[ibx] = ibooker.book1D("DT_TPG_theta_sector_number_"+bxn, 
					 "DT TPG theta sector number "+bxn, 12, -0.5, 11.5 ) ;  
    dttpgthstation[ibx] = ibooker.book1D("DT_TPG_theta_station_number_"+bxn, 
					  "DT TPG theta station number "+bxn, 5, -0.5, 4.5 ) ;  
    dttpgththeta[ibx] = ibooker.book1D("DT_TPG_theta_"+bxn, 
					"DT TPG theta "+bxn, 20, -0.5, 19.5 ) ;  
    dttpgthquality[ibx] = ibooker.book1D("DT_TPG_theta_quality_"+bxn, 
					  "DT TPG theta quality "+bxn, 8, -0.5, 7.5 ) ;  
    dttpgthmapbx[ibx] = ibooker.book2D("DT_TPG_theta_map_bx_"+bxn,
					"Map of triggers per station (BX="+bxn+")",15,1,16,12,0,12);
    setMapThLabel(dttpgthmapbx[ibx]);

    // Phi output
    dttf_p_phi[ibx] = ibooker.book1D("dttf_p_phi_"+bxn, "dttf phi output #phi "+bxn, 256, 
				      -0.5, 255.5);
    dttf_p_qual[ibx] = ibooker.book1D("dttf_p_qual_"+bxn, "dttf phi output qual "+bxn, 8, -0.5, 7.5);
    dttf_p_q[ibx] = ibooker.book1D("dttf_p_q_"+bxn, "dttf phi output q "+bxn, 2, -0.5, 1.5);
    dttf_p_pt[ibx] = ibooker.book1D("dttf_p_pt_"+bxn, "dttf phi output p_{t} "+bxn, 32, -0.5, 31.5);
      
  }

  dttpgphmap = ibooker.book2D("DT_TPG_phi_map",
			       "Map of triggers per station",20,1,21,12,0,12);
  dttpgphmapcorr = ibooker.book2D("DT_TPG_phi_map_corr",
				   "Map of correlated triggers per station",20,1,21,12,0,12);
  dttpgphmap2nd = ibooker.book2D("DT_TPG_phi_map_2nd",
				  "Map of second tracks per station",20,1,21,12,0,12);
  dttpgphbestmap = ibooker.book2D("DT_TPG_phi_best_map",
				   "Map of best triggers per station",20,1,21,12,0,12);
  dttpgphbestmapcorr = ibooker.book2D("DT_TPG_phi_best_map_corr",
				       "Map of correlated best triggers per station",20,1,21,12,0,12);
  setMapPhLabel(dttpgphmap);
  setMapPhLabel(dttpgphmapcorr);
  setMapPhLabel(dttpgphmap2nd);
  setMapPhLabel(dttpgphbestmap);
  setMapPhLabel(dttpgphbestmapcorr);
      


  dttpgthmap = ibooker.book2D("DT_TPG_theta_map",
			       "Map of triggers per station",15,1,16,12,0,12);
  dttpgthmaph = ibooker.book2D("DT_TPG_theta_map_h",
				"Map of H quality triggers per station",15,1,16,12,0,12);
  dttpgthbestmap = ibooker.book2D("DT_TPG_theta_best_map",
				   "Map of besttriggers per station",15,1,16,12,0,12);
  dttpgthbestmaph = ibooker.book2D("DT_TPG_theta_best_map_h",
				    "Map of H quality best triggers per station",15,1,16,12,0,12);
  setMapThLabel(dttpgthmap);
  setMapThLabel(dttpgthmaph);
  setMapThLabel(dttpgthbestmap);
  setMapThLabel(dttpgthbestmaph);
}

void L1TDTTPG::analyze(const Event& e, const EventSetup& c)
{

  nev_++; 
  if(verbose_) cout << "L1TDTTPG: analyze...." << endl;

  edm::Handle<L1MuDTChambPhContainer > myL1MuDTChambPhContainer;  
  e.getByToken(dttpgSourcePhContainer_token_,myL1MuDTChambPhContainer);
  
  if (!myL1MuDTChambPhContainer.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1MuDTChambPhContainer with label "
			     << dttpgSource_.label() ;
    return;
  }
  L1MuDTChambPhContainer::Phi_Container const *myPhContainer =  
    myL1MuDTChambPhContainer->getContainer();

  edm::Handle<L1MuDTChambThContainer > myL1MuDTChambThContainer;  
  e.getByToken(dttpgSourceThContainer_token_,myL1MuDTChambThContainer);
  
  if (!myL1MuDTChambThContainer.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1MuDTChambThContainer with label "
			     << dttpgSource_.label() ;
    edm::LogInfo("DataNotFound") << "if this fails try to add DATA to the process name." ;

    return;
  }
  L1MuDTChambThContainer::The_Container const* myThContainer =  
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


   const L1MuDTChambPhDigi* bestPhQualMap[5][12][4];
   memset(bestPhQualMap,0,240*sizeof(L1MuDTChambPhDigi*));

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
      int thcode[7]= {0,0,0,0,0,0,0};
      for (int j = 0; j < 7; j++)
	{
	  dttpgththeta[bxindex]->Fill(DTThDigiItr->position(j));
	  if (verbose_)
	    {
	      cout << "DTTPG theta position " << DTThDigiItr->position(j) << endl;
	    }
	  thcode[j]=DTThDigiItr->code(j);
	  dttpgthquality[bxindex]->Fill(thcode[j]);
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

  edm::Handle<L1MuDTTrackContainer > myL1MuDTTrackContainer;
  e.getByToken(trToken_,myL1MuDTTrackContainer);

  if (!myL1MuDTTrackContainer.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1MuDTTrackContainer with label "
                               << dttpgSource_.label() ;
    return;
  }

  L1MuDTTrackContainer::TrackContainer const * t =  myL1MuDTTrackContainer->getContainer();



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

void L1TDTTPG::setMapPhLabel(MonitorElement *me)
{

  me->setAxisTitle("DTTF Sector",2);
      for(int i=0;i<5;i++){
	ostringstream wheel;
	wheel << i-2;
	me->setBinLabel(1+i*4,"Wheel "+ wheel.str(),1);
      }
  
}

void L1TDTTPG::setMapThLabel(MonitorElement *me)
{

  me->setAxisTitle("DTTF Sector",2);
      for(int i=0;i<5;i++){
	ostringstream wheel;
	wheel << i-2;
	me->setBinLabel(1+i*3,"Wheel "+ wheel.str(),1);
      }
  
}
