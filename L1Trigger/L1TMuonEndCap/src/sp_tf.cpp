///////////////////////////////////////////////////////////////
// This C++ source file was automatically generated	     //
// by VPPC from a Verilog HDL project.			     //
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/    //
//							     //
// Author    : madorsky//				     //
// Timestamp : Fri Feb  1 08:50:45 2013			     //
///////////////////////////////////////////////////////////////

#include "L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/sp_tf.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "math.h"
#include "Riostream.h"
#include "TFile.h"
#include "TTree.h"

extern size_t __glob_alwaysn__;

extern bool __glob_change__;
extern size_t perm_i;
extern signal_ stdout_sig;


sptf::sptf(const PSet& ps)
{
  count = 0;
  built = false;
  glbl_gsr = true;

  _tpinputs = ps.getParameter<std::vector<edm::InputTag> >("primitiveSrcs");
  _convTrkInputs = ps.getParameter<std::vector<edm::InputTag> >("converterSrcs");

  produces<L1TMuon::InternalTrackCollection> ( "EmuITC" ).setBranchAlias( "EmuITC" );

}

//sptf::sptf(std::vector<edm::InputTag> the_tpinputs, std::vector<edm::InputTag> the_convTrkInputs)//", 
          // edm::ParameterSet the_LUTparam)
//{
//  _tpinputs = the_tpinputs;
//  _convTrkInputs = the_convTrkInputs;
  //LUTparam = the_LUTparam;

//  count = 0;
//  built = false;  
//  glbl_gsr = true; 
//  defparam();

//} // end of constructor


void sptf::produce(edm::Event& ev, const edm::EventSetup& es)
{


	cout<<"Begin SPTF producer:::::::::::::::::::::::::::::::::::::::\n:::::::::::::::::::::::::::::::::::::::::::::::::::::::\n\n";
	
  auto_ptr<L1TMuon::InternalTrackCollection> trkCollect (new L1TMuon::InternalTrackCollection);

//  trkCollect->push_back(L1TMuon::InternalTrack());

  //////////////////////////////////////////////
  ///////// Get Trigger Primitives /////////////
  //////////////////////////////////////////////

  //TriggerPrimitiveList 
  std::vector<TriggerPrimitive> CSCTP;
  std::vector<TriggerPrimitive> tester;
  count++;
  cout << "\n    EVENT " << count << endl;

  // pulling in the csc info from the vector (should be only element)
  auto tpsrc = _tpinputs.cbegin();

  InternalTrack tempTrack;
  tempTrack.setType(2); // track made of CSC hits.. InternalTrack needs some fixing in case a brand new track is being made..

  cout << "Looking for label " << tpsrc->label()
       << ":" << tpsrc->instance() << ":" << tpsrc->process()
       << " CSCTP size is: " << CSCTP.size() << endl;
  cout << "About to access from the event record.\n";
  edm::Handle<TriggerPrimitiveCollection> tps;
  ev.getByLabel(*tpsrc,tps);
  cout << "Finished accessing from the event record.\n";

  auto tp = tps->cbegin();
  auto tpend = tps->cend();
  int ii=0, gg=0;
  int hitCount = 0;
  std::cout << "*** Loop over trigger primitives: " << std::endl;
  for( ; tp != tpend; ++tp )
  {
    
      cout << "-- ev.bunchCrossing() = " << ev.bunchCrossing() << endl;
      cout << "   ev.orbitNumber() = " << ev.orbitNumber() << endl;
      cout << "   ev.run() = " << ev.run() << endl;
      cout << "   tp-tps->cbegin() = " << tp-tps->cbegin() << endl;
      cout << "   tp->getCSCData().bx = " << tp->getCSCData().bx << endl;
      cout << "   tp->getCSCData().bx0 = " << tp->getCSCData().bx0 << endl;
      cout << "   tp->getCSCData().trknmb = " << tp->getCSCData().trknmb << endl;
      cout << "   tp->getCSCData().valid = " << tp->getCSCData().valid << endl;
      cout << "   tp->getCSCData().quality = " << tp->getCSCData().quality << endl;
      cout << "   tp->getCSCData().keywire = " << tp->getCSCData().keywire << endl;
      cout << "   tp->getCSCData().strip = " << tp->getCSCData().strip << endl;
      cout << "~-~-~-~-~-~-~-~-~-~-~-\n";

      if(tp->subsystem() == 1)
      {
          gg++;
          CSCDetId fu = tp->detId<CSCDetId>();
//        TriggerPrimInput input;
//        input.SetValues(fu.station(),fu.chamber(),fu.ring(),fu.triggerSector(),fu.endcap(),tp->Id(),
//                        tp->getCSCData().quality,tp->getPattern(),tp->getWire(),tp->getStrip());
//
          if(fu.triggerSector() == 1)
          {
              //TriggerPrimitive tpref(tps,tp - tps -> cbegin());
              tempTrack.addStub( *tp );

       //       (trkCollect->at(0)).addStub( tpref );         

       ///       if(fu.station() == 1)
       ///       {         
                  CSCTP.push_back(*tp); 
                  tester.push_back(*tp);
                  the_bxValue.push_back(tp->getCSCData().bx);  
                  the_inputOrder[tp->getCSCData().bx - 4].push_back(hitCount);
       ///       }
          }
      }
      hitCount++;

  }


cout << "Tester size: " << tester.size() << endl;

  if (tempTrack.getStubs().size() > 0 )
  {

      std::cout << "CSCTP size: " << CSCTP.size() << std::endl;
      std::cout << "Internal Track has : " << tempTrack . getStubs().size() << " stubs. Printout: " << std::endl;
      const unsigned id = 4*L1TMuon::InternalTrack::kCSC+2-1; // unique identifier for each csc station
    //  L1MuRegionalCand cand = *(tempTrack.parent());
  //    L1MuRegionalCand cand = tempTrack.parent();
 //     cand.setPhiValue(3.1415);
 //     float phiVal = cand.phiValue();
 //     cout << "phiValue = " << phiVal << endl;

      //float phiVal = L1TMuon::(L1MuRegionalCand::InternalTrack::phiValue());
      //cout << "phiValue = " << phiVal << endl;
      cout << "id=" << id << endl;

      if(!(tempTrack.getStubs()).count(id)) { cout << "CSC Station map empty.\n";  }

  }

//  ConvHits = PrimConv(CSCTP);

/////// END OF PRIMITIVE EXTRACTION ////////////


// Looping over the converted hits and storing the primitives in the 5-bx vectors

  hitCount = 0;
  int bxIndex = 0;
  for(std::vector<TriggerPrimitive>::iterator h = tester.begin(); h != tester.end(); h++)
//  for(std::vector<ConvertedHit>::iterator h = ConvHits.begin();h != ConvHits.end();h++)
  {
      cout << "An iteration through the trig prim h loop.\n";

      TriggerPrimitive C3 = *h;
      CSCDetId Det = C3.detId<CSCDetId>();
      int station = Det.station();

      //int sector = Det.triggerSector();
      int chamber = Det.chamber();
      int sub = 0;
      int strip = (Det.ring() == 4) ? (C3.getCSCData().strip + 128): C3.getCSCData().strip;
      	
      
      std::cout<<"ring = "<<Det.ring()<<" strip = "<<strip<<std::endl;
      
      if(station == 1)
      {
      
      	  if(chamber%6 > 2)
	  	sub = 1;
	  else
	  	sub = 2;
	  
	  /*
          switch(sector)
          {
              case 1:if(chamber == 3 || chamber == 4 || chamber == 5){sub = 1;}else sub = 2;break;
              case 2:if(chamber == 9 || chamber == 10 || chamber == 11){sub = 1;}else sub = 2;break;
              case 3:if(chamber == 15 || chamber == 16 || chamber == 17){sub = 1;}else sub = 2;break;
              case 4:if(chamber == 21 || chamber == 22 || chamber == 23){sub = 1;}else sub = 2;break;
              case 5:if(chamber == 27 || chamber == 28 || chamber == 29){sub = 1;}else sub = 2;break;
              case 6:if(chamber == 33 || chamber == 34 || chamber == 35){sub = 1;}else sub = 2;break;
          }
	  */
      }


//      if((*h).Id() > 9) {(*h).SetId((*h).Id()-9);  (*h).SetStrip((*h).Strip()+128); }
   //   if((*h).id > 9) {(*h).id = (*h).id - 9; (*h).strip = (*h).strip + 128; }

      bxIndex = the_bxValue[hitCount] - 4;
      the_primSelector[bxIndex].push_back( 0 );
      //the_inputOrder[bxIndex].push_back( hitCount          );
      the_bx_jitter[bxIndex].push_back(  count             );
      the_endcap[bxIndex].push_back(     1                 );
      the_sector[bxIndex].push_back(     1                 );
      the_subsector[bxIndex].push_back(  sub               );
//      the_subsector[bxIndex].push_back(  (*h).sub          );
      the_station[bxIndex].push_back(    station           );
//      the_station[bxIndex].push_back(    (*h).station      );
      the_valid[bxIndex].push_back(      1                 );
      the_quality[bxIndex].push_back(    C3.getCSCData().quality    );
//      the_quality[bxIndex].push_back(    (*h).quality      );
      the_pattern[bxIndex].push_back(    C3.getPattern()    );
//      the_pattern[bxIndex].push_back(    (*h).pattern      );
      the_wiregroup[bxIndex].push_back(  C3.getCSCData().keywire       );
//      the_wiregroup[bxIndex].push_back(  (*h).wire         );
      the_cscid[bxIndex].push_back(      C3.Id()         );
//      the_cscid[bxIndex].push_back(      (*h).id           );
      the_bend[bxIndex].push_back(       0                 );
      the_halfstrip[bxIndex].push_back(  strip      );
//      the_halfstrip[bxIndex].push_back(  (*h).strip        );
      hitCount++;
  }

  // Defining the parameters and running the event
//  if(static_cast<int>(the_endcap[2].size()) > 0  || static_cast<int>(the_endcap[4].size()) > 0)
//  {
//      defparam();



//// Beginning of what used to be the run event portion  ///

	
  // initialize event storage
  for (iev = 0; iev < _max_ev; iev=iev+1)
      for (ist = 0; ist < 5; ist=ist+1) 
          for (icid = 0; icid < 9; icid=icid+1)
	          for (ipr = 0; ipr < _seg_ch; ipr=ipr+1)
	          {
	              quality  [iev][ist][icid][ipr] = const_(4, 0x0UL);
	              wiregroup[iev][ist][icid][ipr] = const_(7, 0x0UL);
	              hstrip   [iev][ist][icid][ipr] = const_(8, 0x0UL);
	              clctpat  [iev][ist][icid][ipr] = const_(4, 0x0UL);
	          }
  
  /////////  START OF VECTOR DATA READING //////////////////////////
  
  _event = 2;
  int tempVar = 0; 
  for(int m=0; m<5; m++)
  {
  //    cout << "Bx Index: " << m << endl;
      for(int i=0; i<5; i=i+1)
	      for(int j=0; j<9; j=j+1)
	          pr_cnt[i][j] = 0;  
      j=0;
      _event = _event + 1;

      for(int k=0; k<static_cast<int>((the_endcap[m]).size()); k++)
	  {
	      _bx_jitter   =  the_bx_jitter[m][k];
	      _endcap      =  the_endcap[m][k];
	      _sector      =  the_sector[m][k];
	      _subsector   =  the_subsector[m][k];
	      _station     =  the_station[m][k];
	      _valid       =  the_valid[m][k];
	      _quality     =  the_quality[m][k];
	      _pattern     =  the_pattern[m][k];
	      _wiregroup   =  the_wiregroup[m][k];
	      _cscid       =  the_cscid[m][k] - 1;
	      _bend        =  the_bend[m][k];
	      _halfstrip   =  the_halfstrip[m][k];
	  
	      _sector = 1;
	  
	      if(_station == 1 && _subsector == 1) _station = 0;
	  
	      cout << "About to apply a series of tests to the hit data.\n";
	      if (_station > 4)  cout << "Station is too large.\n";
	      else if (_sector != 1)  cout << "Sector != 1.\n";
	      else if (_cscid > 8)  cout << "Cscid is too large.\n";
	      else if (pr_cnt[_station][_cscid] >= _seg_ch)  cout << "Pr_cnt is too large.\n" ;
	      else if (_event >= _max_ev) continue;
	      else
	      {
	          cout << "Hit data passed all requirements. Printing out the values for this hit.\n";
	          cout << _event  << " " << _endcap << " " << _sector << " " << _subsector << " " << _station << endl;
	          cout <<  _valid << " " << _quality << " " << _pattern << " " << _wiregroup << endl;
	          cout << _cscid+1 << " " << _bend << " " << _halfstrip << endl;
	      
	          _bx_jitter = 0; // remove jitter
	      
	          quality  [_event][_station][_cscid][pr_cnt[_station][_cscid]] = _quality;
	          wiregroup[_event][_station][_cscid][pr_cnt[_station][_cscid]] = _wiregroup;
	          hstrip   [_event][_station][_cscid][pr_cnt[_station][_cscid]] = _halfstrip;
	          clctpat  [_event][_station][_cscid][pr_cnt[_station][_cscid]] = _pattern;
	      
	          // increase primitive counter
	          pr_cnt[_station][_cscid] = pr_cnt[_station][_cscid] + 1;  
	      }
	  }   
  }

  /////////  END OF VECTOR DATA READING ///////////////////////////
  
  seli = 0;
  addri = 0;
  r_ini = 0;
  wei = 0;
  clki = 0;
  int theEvent = 0;  /// my addition  

  elapsed_time = 0;
  
  best_tracks = Sfopen("best_tracks.out", "w");
  
  //for (i = 0; i < 200 + _max_ev-1; i = i+1)
  for (i=0; i <= 200+_max_ev; i = i+1)
  {   
      wei = 0;
      for (k = 0; k < 5; k = k+1)
	  {
	      csi[k] = 0;
	      for (j = 0; j < 9; j = j+1)
	      {
	          for (si = 0; si < _seg_ch; si = si+1)
		      {
		          qi[k][j][si] = 0;
		          wgi[k][j][si] = 0;
		          hstri[k][j][si] = 0;
		          cpati[k][j][si] = 0;
		      }
	      }
	  }
      
      // write ph_init and th_init parameters into ME1/1 only
      if (i < 36)
	  {
	      csi[i/18][(i/6)%3] = 1;//[station][chamber]
	      seli = 0;
	      wei = 1;
	      addri = i%6;
	      if (( (addri) == 0)) { r_ini = ph_init[i/18][(i/6)%3]; } else 
	      if (( (addri) == // ph_init_b
		  1)) { r_ini = ph_disp[(i/18)*12 + (i/6)%3]; } else 
					if (( (addri) == // ph_disp_b
					      2)) { r_ini = ph_init[i/18][(i/6)%3 + 9]; } else 
					if (( (addri) == // ph_init_a
					      3)) { r_ini = ph_disp[(i/18)*12 + (i/6)%3 + 9]; } else 
					if (( (addri) == // ph_disp_a
					      4)) { r_ini = th_init[(i/18)*12 + (i/6)%3 + 9]; } else 
					if (( (addri) == // th_init
					      5)) { r_ini = th_disp[(i/18)*12 + (i/6)%3 + 9]; } 
	  }
      
      // all other ME1 chambers
      if (i >= 36 && i < 36+48)
	  {
	      ii = i - 36;
	      csi[ii/24][(ii/4)%6+3] = 1;//[station][chamber]
	      seli = 0;
	      wei = 1;
	      addri = ii % 4;
	      if (( (addri) == 0)) { r_ini = ph_init[ii/24][(ii/4)%6+3]; } else 
	      if (( (addri) == // ph_init
		  1)) { r_ini = th_init[(ii/24)*12 + (ii/4)%6+3]; } else 
					if (( (addri) == // th_init
					      2)) { r_ini = ph_disp[(ii/24)*12 + (ii/4)%6+3]; } else 
					if (( (addri) == // ph_disp
					      3)) { r_ini = th_disp[(ii/24)*12 + (ii/4)%6+3]; } 
	  }
      
      // ME2,3,4 chambers
      if (i >= 36+48 && i < 36+48+108)
	  {
	      ii = i - (36+48);
	      csi[ii/36+2][(ii/4)%9] = 1; //[station][chamber]
	      seli = 0;
	      wei = 1;
	      addri = ii % 4;
	      if (( (addri) == 0)) { r_ini = ph_init[ii/36+2][(ii/4)%9]; } else 
	      if (( (addri) == // ph_init
		  1)) { r_ini = th_init[(ii/36)*9 + (ii/4)%9 + 24]; } else 
					if (( (addri) == // th_init
					      2)) { r_ini = ph_disp[(ii/36)*9 + (ii/4)%9 + 24]; } else 
					if (( (addri) == // ph_disp
					      3)) { r_ini = th_disp[(ii/36)*9 + (ii/4)%9 + 24]; } 
	  }
      
      // read params registers
      if (i == 193)
	  {
	      ///  Deleted a bunch of code that was already commented out ///
	  }
      
      if (i > 200)
	  {
	      good_ev = 0;
	      st_cnt = 0;
	      for (ist = 0; ist < 5; ist=ist+1) 
	      {
	          for (icid = 0; icid < 9; icid=icid+1)
		      {
		          for (si = 0; si < _seg_ch; si = si+1)
		          {
		      
		              qi   [ist][icid][si] = quality  [theEvent][ist][icid][si];
		              wgi  [ist][icid][si] = wiregroup[theEvent][ist][icid][si];
		              hstri[ist][icid][si] = hstrip   [theEvent][ist][icid][si];
		              cpati[ist][icid][si] = clctpat  [theEvent][ist][icid][si];
		      
		              // check if there is chamber data, update good event station mask
		              if (qi  [ist][icid][si] > 0)  good_ev[ist] = 1;
			      
		          }
		      } // for (icid = 0; icid < 9; icid=icid+1)
	          // count stations in this event
	          if (good_ev[ist]) st_cnt = st_cnt + 1;
	      }
	      theEvent = theEvent + 1;
	      // count event as good if more than 2 stations, other than 3-4
	      if (good_ev != 0 && good_ev != 1 && good_ev != 2 && good_ev != 4 && good_ev != 8 && good_ev != 16 && good_ev != 24) good_ev_cnt = good_ev_cnt+1;
	      if (good_ev == 24) st_cnt = 7; // 3-4 tracks marked as having 7 stations
	      begin_time = Stime;
	  }     
     
   //   cout << "i = " << i << endl;
 
      for (j = 0; j < 2; j = j+1)
	  {
	      clk_drive(clki, j);
	  
	      while(true)
	      {
	          __glob_change__ = false;
	          init();
     //         cout << "Just did an init()\n";
	          if (!__glob_change__) break;
       //       cout << "Did not break.\n";
	          perm_i = 0;
	          uut (qi, wgi, hstri, cpati, csi, pps_csi, seli, addri, r_ini, r_outo, wei, bt_phi, bt_theta,
		           bt_cpattern, bt_delta_ph, bt_delta_th, bt_sign_ph, bt_sign_th, bt_rank, bt_vi,
		           bt_hi, bt_ci, bt_si, clki, clki);
         //     cout << "Finished doing a () operator on uut.\n";
	      }
	  
	  }
      
      /////// ACCESSING THE VARIABLES ///////////
      if (i > 200)
	  {                    
	      iev = i-200;
	  
	      for (ip = 0; ip < 5; ip = ip+1)
	      {
	          for (j = 0; j < 9; j = j+1)//
		      {
		          for (k = 0; k < 2; k = k+1)//
		          {
		              if (uut.ph[ip][j][k] != 0 && iev == 6)
			          {	
			  
			              ////// PRINT PHI VALUES////////////
			              ///////////////////////////////////
			              the_emuPhi.push_back(uut.ph[ip][j][k]);
			              
				      Swrite("%d\n",& uut.ph[ip][j][k]);
				      
				      std::cout<<"sta = "<<ip<<" and chamber = "<<j<<std::endl;
				      
			              if (ip <= 1 && j < 3) // ME11
			              {
			      
			                  /////////PRINT THETA VALUES////////////
			                  //////////////////////////////////////			                  
				              if (uut.th11[ip][j][k*2] != 0){
					      	the_emuTheta.push_back(uut.th11[ip][j][k*2]);
						Swrite ("%d\n", & uut.th11[ip][j][k*2]);
					      }
				              else{
					      	the_emuTheta.push_back(uut.th11[ip][j][k*2+1]);
						Swrite ("%d\n", & uut.th11[ip][j][k*2+1]);
					      }  
				             
			                  //////////PRINT PH_HIT VALUES///////////
			                  ///////////////////////////////////////
			                  for(int o = 0;o < 24;o++)
				              {
				                  if(uut.pcs.genblk.station11[ip].csc11[j].pc11.ph_hit[o] != 0)
				                  {
				                      the_emuPhhit.push_back(o);
						      std::cout<<"1-"<<o<<" ";
				                  }
				              }  
			              }	
			              else if(ip <= 1 && j > 2)
			              {
			      
			                  /////////PRINT THETA VALUES////////////
			                  //////////////////////////////////////
				              if (uut.th[ip][j][k] != 0){
					      	the_emuTheta.push_back(uut.th[ip][j][k]);
						Swrite ("%d\n", & uut.th[ip][j][k]);
					      }
				              
			                  //////////PRINT PH_HIT VALUES///////////
			                  ///////////////////////////////////////
			                  for(int o = 0;o < 24;o++)
				              {
				                  if(uut.pcs.genblk.station12[ip].csc12[j].pc12.ph_hit[o] != 0)
				                  {
				                      the_emuPhhit.push_back(o);
						      std::cout<<"2-"<<o<<" ";
				                  }
				              }
			              }
			              else
			              {
			                  /////////PRINT THETA VALUES////////////
			                  //////////////////////////////////////			                  				              
				              the_emuTheta.push_back(uut.th[ip][j][k]);
					      Swrite("%d\n", & uut.th[ip][j][k]);
				              													    
			                  //////////PRINT PH_HIT VALUES///////////
			                  ///////////////////////////////////////
			                  for(int o = 0;o < 44;o++)
				              {
				                  if(uut.pcs.genblk.station[ip].csc[j].pc.ph_hit[o] != 0)
				                  { 
				                      the_emuPhhit.push_back(o);
						      std::cout<<"3-"<<o<<" ";
				                  }
				              }
			              }
			  
			  
			              //////////PRINT PHZVL VALUES///////////
			              ///////////////////////////////////////
			              the_emuPhzvl.push_back(uut.phzvl[ip][j]); 
				      Swrite("%d\n\n", & uut.phzvl[ip][j]);
			          }
		          }
		      }
	      }
	  
	      if(iev == 11)
	      {
	          //////////PRINT QUALITY CODES///////////
	          ////////////////////////////////////////
	          ///////////////////////////////////////
	          for (iz = 0; iz < 4; iz = iz+1)
		      {
		          //for (ist = ph_raw_w; ist > 0; ist = ist - 1)
		          for (ist = 1; ist < _ph_raw_w; ist = ist + 1) // reverse printing order
		          {
		              // cout << "uut.ph_rank[iz][ist-1] = " << uut.ph_rank[iz][ist-1] << endl;
		              if (uut.ph_rank[iz][ist-1] > 0)
			          { 
				  
				      Swrite("key_strip:%d q: %d ly: %b%b%b  st: %b%b%b\n", 
									   &ist, & uut.ph_rank[iz][ist-1],
									   &uut.ph_rank[iz][ist-1][4], &uut.ph_rank[iz][ist-1][2], &uut.ph_rank[iz][ist-1][0],
									   &uut.ph_rank[iz][ist-1][5], &uut.ph_rank[iz][ist-1][3], &uut.ph_rank[iz][ist-1][1]
									   );
				      	
			              the_emuStrip.push_back(ist);
			              the_emuQuality.push_back(uut.ph_rank[iz][ist-1]);
			  
			              tempVar = 0;
			              if(uut.ph_rank[iz][ist-1][4] == 1) tempVar = tempVar + 4;
		    	          if(uut.ph_rank[iz][ist-1][2] == 1) tempVar = tempVar + 2;
			              if(uut.ph_rank[iz][ist-1][0] == 1) tempVar = tempVar + 1;
			              the_emuLayer.push_back(tempVar);
			  
			              tempVar = 0;
			              if(uut.ph_rank[iz][ist-1][5] == 1) tempVar = tempVar + 4;
			              if(uut.ph_rank[iz][ist-1][3] == 1) tempVar = tempVar + 2;
			              if(uut.ph_rank[iz][ist-1][1] == 1) tempVar = tempVar + 1;
			              the_emuStraight.push_back(tempVar); 
    			      }
	    	      }
	    	  } // for (iz = 0; iz < 4; iz = iz+1)
	          ////			std::cout<<"\n\n";
	  

          } // end of if iev==11

           //   cout << "Outside quality code saving, before possible track info output.\n";

          if(iev==18 && bt_rank[0] > 0)
          {
              // Printing out found track information.
              for (ip = 0; ip < 3; ip = ip+1)
              {
                  if (bt_rank[ip] != 0)
                  {

                      trkCollect->push_back(L1TMuon::InternalTrack());
                    
                      cout << "ME ";
                      if (bt_rank[ip][5]) cout << "-1";
                      if (bt_rank[ip][3]) cout << "-2";
                      if (bt_rank[ip][1]) cout << "-3";
                      if (bt_rank[ip][0]) cout << "-4";
                   
                      cout << " track: " << ip << ", rank: " << bt_rank[ip] << ", ph_deltas: " << bt_delta_ph[ip][0] << " " << bt_delta_ph[ip][1];
                      cout << ", th_deltas: " << bt_delta_th[ip][0] << " " << bt_delta_th[ip][1] << ", phi: " << bt_phi[ip] << ", theta: " << bt_theta[ip];
                      cout << ", cpat: " << bt_cpattern[ip] << endl;

                      (trkCollect->at(ip)).phi   = bt_phi[ip];
                      (trkCollect->at(ip)).theta = bt_theta[ip];
		      (trkCollect->at(ip)).rank  = bt_rank[ip];
		      std::vector<int> delt (2,-999);
		      std::vector<std::vector<int>> delta (2,delt);
		      delta[0][0] = bt_delta_ph[ip][0];
		      delta[0][1] = bt_delta_ph[ip][1];
		      delta[1][0] = bt_delta_th[ip][0];
		      delta[1][1] = bt_delta_th[ip][1];
		      (trkCollect->at(ip)).deltas = delta;
		      

                      int foundHitCntr = 0;
                      int numHitsInTrack = 0;
                      // Looping over the stations
                      for (j = 0; j < 5; j = j+1)
                      {
                          cout << bt_vi[ip][j] << ":" << bt_hi[ip][j] << ":" << bt_ci[ip][j] << ":" << bt_si[ip][j] << "   ";

                          if(static_cast<int>(bt_vi[ip][j]) == 1) numHitsInTrack++;
                          cout << "Searching for this hit.\n";

                          bool foundHit = false;
                          for(int m=0; m<5 && !foundHit; m++)
                          {
                              for(int i=0; i<static_cast<int>(the_station[m].size()) && !foundHit; i++)
                              {
                              
                                  int hitStation;
                                  int hitCscid = the_cscid[m][i] - 1;
                                  if(the_station[m][i] == 1 && the_subsector[m][i] == 1) hitStation = 0;
                                  else hitStation = the_station[m][i];

                                  // If the valid for this track stub does not match a given hit
                                  if(the_valid[m][i]   !=  static_cast<int>(bt_vi[ip][j]) )
                                  {
                                      cout << "Valid mismatch. the_valid[" << m << "][" << i << "]=" << the_valid[m][i]
                                           << " != bt_vi[" << ip << "][" << j << "]=" << bt_vi[ip][j] << endl; 
                                      continue;
                                  }
                              
                                  // If the station for this track stub does not match a given hit
                                  if(hitStation !=  static_cast<int>(j)            ) 
                                  { 
                                      cout << "Station mismatch. hitStation=" << the_station[m][i]
                                           << " != j=" << j << endl; 
                                      continue; 
                                  }

                                  // If the cscid for this track stub does not match a given hit
                                  if((the_cscid[m][i]-1)   !=  static_cast<int>(bt_ci[ip][j]) ) 
                                  { 
                                      cout << "Cscid mismatch. hitCscid=" << hitCscid
                                           << " != bt_ci[" << ip << "][" << j << "]=" << bt_ci[ip][j] << endl;
                                      continue; 
                                  }

                                  // If the pr_cnt does not match num segments for a given hit
                                  if(static_cast<int>(pr_cnt[hitStation][hitCscid]) != static_cast<int>(bt_si[ip][j]) ) 
                                  {
                                      cout << "Pr_cnt mismatch. pr_cnt[" << hitStation << "][" << hitCscid << "]=" 
                                           << pr_cnt[hitStation][hitCscid] << " != bt_si[" << ip << "][" << j << "]="
                                           << bt_si[ip][j] << endl;
                                      continue;
                                  }

                                  // If the bx position does not match that for a given hit
                                  if((the_bxValue[i]-4) != static_cast<int>(bt_hi[ip][j]) )
                                  {
                                      cout << "Bx Value mismatch. (the_bxValue[hit#=" << i << "]-4)=" << the_bxValue[i]-4
                                           << " != bt_hi[" << ip << "][" << j << "]=" << bt_ci[ip][j] << endl;
                                      continue;
                                  }

                                  cout << "Found the hit that was part of the track.\n";
                                  foundHit = true;
                                  if(foundHit) the_primSelector[m][i] = ip+1;
                                  foundHitCntr++;
                              }
                          }    
                      }
                      cout << "Identified " << foundHitCntr << " of " << numHitsInTrack << " hits making up this track.";

                      if(foundHitCntr == numHitsInTrack) cout << " SUCCESS\n";
                      else cout << " FAILURE\n";
                  }
              }
          } // end of if iev == 18     
      } // end of if i>200 (iev > 0)
  }
  
  ////	Swrite ("elapsed_time: %t\n", & elapsed_time);
  //////////	Sfclose (best_tracks);
  
  ///// this is the weird bracket		}
///  cout << "Finished the operator function.\n";


/////// end of what used to be runEvent, beginning of the last part of produce  //////


//      runEvent();
//  }

  cout << "Finished runEvent().\n";


  cout << "Now we figure out which hits to select from the trigger prim input.\n";

  // Looping over the tracks
  for(int ip=0; ip<3; ip++)
  {
   //   trkCollect->push_back(L1TMuon::InternalTrack());
     // cout << "Looping over tracks\n";
   
      // Looping over the bx values
      for(int m=0; m<5; m++)
      {
          //cout << "Looping over the bx values\n";
          // Looping over the hits in a given bx
          for(int i=0; i<static_cast<int>(the_station[m].size()); i++)
          {
              cout << "Trk: " << ip+1 << ",  BX: " << m << ",  Hit: " << i << ",    primSelect: " << the_primSelector[m][i]
                   << ",    inputOrder[m=" << m << "][i=" << i << "] = " << the_inputOrder[m][i] << endl;              


          }
      }
  }

  tp = tps->cbegin();
  tpend = tps->cend();
  ii=0;
  gg=0;
  hitCount = 0;
  std::cout << "*** Loop over trigger primitives: " << std::endl;
  // Looping over all trigger primitives from the event record
  for( ; tp != tpend; ++tp )
  {
      // Loop over the_inputOrder to see if this primitive is even used
      for(int m=0; m<5; m++)
      {
          for(int i=0; i<static_cast<int>(the_station[m].size()); i++)
          {      
              // If this primitive matches one used in one of the tracks
              if(the_inputOrder[m][i] == hitCount)
              {
                  // Verifying that this hit actually is used by a track
                  if(the_primSelector[m][i] > 0) 
                  {
                      // Creating a stub reference and adding it to the collection
                      //TriggerPrimitiveRef tpref(tps,tp-tps->cbegin());
                      (trkCollect->at(the_primSelector[m][i] - 1)) . addStub( *tp );
                      cout << "Just added a stub to the emu ITC.\n";
                  }
              }
          }
      }
    
      hitCount++;
  }

           


cout << "Finished the track stuff.\n";


  for(int i=0; i<static_cast<int>(the_emuPhi.size()); i++)
  {
//      emuPhi       -> push_back ( the_emuPhi.at(i)      );
//      emuTheta     -> push_back ( the_emuTheta.at(i)    );
//      emuPhhit     -> push_back ( the_emuPhhit.at(i)    );
//      emuPhzvl     -> push_back ( the_emuPhzvl.at(i)    );

      cout << "Hit " << i+1 << ":  Phi=" << the_emuPhi.at(i) << ", Theta=" << the_emuTheta.at(i)
           << " Phhit=" << the_emuPhhit.at(i) << ", Phzvl=" << the_emuPhzvl.at(i) << endl;

///////      trkCollect->push_back(L1TMuon::InternalTrack());
//      (trkCollect->at(i)).varStorage.phi = the_emuPhi.at(i);
//      (trkCollect->at(i)).varStorage.theta = the_emuTheta.at(i);

///////    (trkCollect->at(i)).phi = the_emuPhi.at(i);
///////    (trkCollect->at(i)).theta = the_emuTheta.at(i);


  }

  for(int i=0; i<static_cast<int>(the_emuStrip.size()); i++)
  {
//      emuStrip     -> push_back ( the_emuStrip.at(i)    );
//      emuStraight  -> push_back ( the_emuStraight.at(i) );
//      emuLayer     -> push_back ( the_emuLayer.at(i)    );
//      emuQuality   -> push_back ( the_emuQuality.at(i)  );

      cout << "Evt " << count << "- Intermed. Vars " << i+1 << ":  Strip=" << the_emuStrip.at(i) << ", Straight=" << the_emuStraight.at(i)
           << ", Layer=" << the_emuLayer.at(i) << ", Quality=" << the_emuQuality.at(i) << endl;
  }

  cout << "----------------\n\n";

  // Clearing the primitives
  for(int m=0; m<5; m++)
  {
      the_primSelector[m].clear();
      the_inputOrder[m].clear();
      the_bx_jitter[m].clear();
      the_endcap[m].clear();
      the_sector[m].clear();
      the_subsector[m].clear();
      the_station[m].clear();
      the_valid[m].clear();
      the_quality[m].clear();
      the_pattern[m].clear();
      the_wiregroup[m].clear();
      the_cscid[m].clear();
      the_bend[m].clear();
      the_halfstrip[m].clear();
  }

cout << "Cleared the primitives.\n";

  // Clearing the variables
  the_emuPhi.clear();
  the_emuTheta.clear();
  the_emuPhhit.clear();
  the_emuPhzvl.clear();
  the_emuStrip.clear();
  the_emuStraight.clear();
  the_emuLayer.clear();
  the_emuQuality.clear();

cout << "cleared everything.\n";

ev.put( trkCollect , "EmuITC" );
  
  cout<<"End SPTF producer:::::::::::::::::::::::::::::::::::::::\n:::::::::::::::::::::::::::::::::::::::::::::::::::::::\n\n";
  
}   /// end of produce




// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
//void sptf::defparam()
//{
  //	station = 1;
  //	cscid = 1;
//}

// vppc: this function allocates memory for internal signals
void sptf::build()
{
	built = true;
	qi__storage.add_dim(4, 0);
	qi__storage.add_dim(8, 0);
	qi__storage.add_dim(_seg_ch-1, 0);
	qi__storage.bw(3, 0);
	qi__storage.build();
	qi.add_dim(4, 0);
	qi.add_dim(8, 0);
	qi.add_dim(_seg_ch-1, 0);
	qi.bw(3, 0);
	qi.build();
	qi.set_storage (&qi__storage);
	wgi__storage.add_dim(4, 0);
	wgi__storage.add_dim(8, 0);
	wgi__storage.add_dim(_seg_ch-1, 0);
	wgi__storage.bw(_bw_wg-1, 0);
	wgi__storage.build();
	wgi.add_dim(4, 0);
	wgi.add_dim(8, 0);
	wgi.add_dim(_seg_ch-1, 0);
	wgi.bw(_bw_wg-1, 0);
	wgi.build();
	wgi.set_storage (&wgi__storage);
	hstri__storage.add_dim(4, 0);
	hstri__storage.add_dim(8, 0);
	hstri__storage.add_dim(_seg_ch-1, 0);
	hstri__storage.bw(_bw_hs-1, 0);
	hstri__storage.build();
	hstri.add_dim(4, 0);
	hstri.add_dim(8, 0);
	hstri.add_dim(_seg_ch-1, 0);
	hstri.bw(_bw_hs-1, 0);
	hstri.build();
	hstri.set_storage (&hstri__storage);
	cpati__storage.add_dim(4, 0);
	cpati__storage.add_dim(8, 0);
	cpati__storage.add_dim(_seg_ch-1, 0);
	cpati__storage.bw(3, 0);
	cpati__storage.build();
	cpati.add_dim(4, 0);
	cpati.add_dim(8, 0);
	cpati.add_dim(_seg_ch-1, 0);
	cpati.bw(3, 0);
	cpati.build();
	cpati.set_storage (&cpati__storage);
	csi__storage.add_dim(4, 0);
	csi__storage.bw(8, 0);
	csi__storage.build();
	csi.add_dim(4, 0);
	csi.bw(8, 0);
	csi.build();
	csi.set_storage (&csi__storage);
	pps_csi__storage.add_dim(2, 0);
	pps_csi__storage.bw(4, 0);
	pps_csi__storage.build();
	pps_csi.add_dim(2, 0);
	pps_csi.bw(4, 0);
	pps_csi.build();
	pps_csi.set_storage (&pps_csi__storage);
	seli__storage.bw(1, 0);
	seli.bw(1, 0);
	seli.set_storage (&seli__storage);
	addri__storage.bw(_bw_addr-1, 0);
	addri.bw(_bw_addr-1, 0);
	addri.set_storage (&addri__storage);
	r_ini__storage.bw(11, 0);
	r_ini.bw(11, 0);
	r_ini.set_storage (&r_ini__storage);
	wei__storage.bw(0, 0);
	wei.bw(0, 0);
	wei.set_storage (&wei__storage);
	clki__storage.bw(0, 0);
	clki.bw(0, 0);
	clki.set_storage (&clki__storage);
	ph_init__storage.add_dim(4, 0);
	ph_init__storage.add_dim(11, 0);
	ph_init__storage.bw(_bw_fph, 0);
	ph_init__storage.build();
	ph_init.add_dim(4, 0);
	ph_init.add_dim(11, 0);
	ph_init.bw(_bw_fph, 0);
	ph_init.build();
	ph_init.set_storage (&ph_init__storage);
	th_init__storage.add_dim(50, 0);
	th_init__storage.bw(_bw_th-1, 0);
	th_init__storage.build();
	th_init.add_dim(50, 0);
	th_init.bw(_bw_th-1, 0);
	th_init.build();
	th_init.set_storage (&th_init__storage);
	ph_disp__storage.add_dim(50, 0);
	ph_disp__storage.bw(_bw_ph, 0);
	ph_disp__storage.build();
	ph_disp.add_dim(50, 0);
	ph_disp.bw(_bw_ph, 0);
	ph_disp.build();
	ph_disp.set_storage (&ph_disp__storage);
	th_disp__storage.add_dim(50, 0);
	th_disp__storage.bw(_bw_th-1, 0);
	th_disp__storage.build();
	th_disp.add_dim(50, 0);
	th_disp.bw(_bw_th-1, 0);
	th_disp.build();
	th_disp.set_storage (&th_disp__storage);
	quality__storage.add_dim(_max_ev-1, 0);
	quality__storage.add_dim(4, 0);
	quality__storage.add_dim(8, 0);
	quality__storage.add_dim(_seg_ch-1, 0);
	quality__storage.bw(3, 0);
	quality__storage.build();
	quality.add_dim(_max_ev-1, 0);
	quality.add_dim(4, 0);
	quality.add_dim(8, 0);
	quality.add_dim(_seg_ch-1, 0);
	quality.bw(3, 0);
	quality.build();
	quality.set_storage (&quality__storage);
	wiregroup__storage.add_dim(_max_ev-1, 0);
	wiregroup__storage.add_dim(4, 0);
	wiregroup__storage.add_dim(8, 0);
	wiregroup__storage.add_dim(_seg_ch-1, 0);
	wiregroup__storage.bw(6, 0);
	wiregroup__storage.build();
	wiregroup.add_dim(_max_ev-1, 0);
	wiregroup.add_dim(4, 0);
	wiregroup.add_dim(8, 0);
	wiregroup.add_dim(_seg_ch-1, 0);
	wiregroup.bw(6, 0);
	wiregroup.build();
	wiregroup.set_storage (&wiregroup__storage);
	hstrip__storage.add_dim(_max_ev-1, 0);
	hstrip__storage.add_dim(4, 0);
	hstrip__storage.add_dim(8, 0);
	hstrip__storage.add_dim(_seg_ch-1, 0);
	hstrip__storage.bw(_bw_hs-1, 0);
	hstrip__storage.build();
	hstrip.add_dim(_max_ev-1, 0);
	hstrip.add_dim(4, 0);
	hstrip.add_dim(8, 0);
	hstrip.add_dim(_seg_ch-1, 0);
	hstrip.bw(_bw_hs-1, 0);
	hstrip.build();
	hstrip.set_storage (&hstrip__storage);
	clctpat__storage.add_dim(_max_ev-1, 0);
	clctpat__storage.add_dim(4, 0);
	clctpat__storage.add_dim(8, 0);
	clctpat__storage.add_dim(_seg_ch-1, 0);
	clctpat__storage.bw(3, 0);
	clctpat__storage.build();
	clctpat.add_dim(_max_ev-1, 0);
	clctpat.add_dim(4, 0);
	clctpat.add_dim(8, 0);
	clctpat.add_dim(_seg_ch-1, 0);
	clctpat.bw(3, 0);
	clctpat.build();
	clctpat.set_storage (&clctpat__storage);
	v0__storage.bw(15, 0);
	v0.bw(15, 0);
	v0.set_storage (&v0__storage);
	v1__storage.bw(15, 0);
	v1.bw(15, 0);
	v1.set_storage (&v1__storage);
	v2__storage.bw(15, 0);
	v2.bw(15, 0);
	v2.set_storage (&v2__storage);
	v3__storage.bw(15, 0);
	v3.bw(15, 0);
	v3.set_storage (&v3__storage);
	v4__storage.bw(15, 0);
	v4.bw(15, 0);
	v4.set_storage (&v4__storage);
	v5__storage.bw(15, 0);
	v5.bw(15, 0);
	v5.set_storage (&v5__storage);
	pr_cnt__storage.add_dim(5, 0);
	pr_cnt__storage.add_dim(8, 0);
	pr_cnt__storage.bw(2, 0);
	pr_cnt__storage.build();
	pr_cnt.add_dim(5, 0);
	pr_cnt.add_dim(8, 0);
	pr_cnt.bw(2, 0);
	pr_cnt.build();
	pr_cnt.set_storage (&pr_cnt__storage);
	_event__storage.bw(9, 0);
	_event.bw(9, 0);
	_event.set_storage (&_event__storage);
	_bx_jitter__storage.bw(9, 0);
	_bx_jitter.bw(9, 0);
	_bx_jitter.set_storage (&_bx_jitter__storage);
	_endcap__storage.bw(1, 0);
	_endcap.bw(1, 0);
	_endcap.set_storage (&_endcap__storage);
	_sector__storage.bw(2, 0);
	_sector.bw(2, 0);
	_sector.set_storage (&_sector__storage);
	_subsector__storage.bw(1, 0);
	_subsector.bw(1, 0);
	_subsector.set_storage (&_subsector__storage);
	_station__storage.bw(2, 0);
	_station.bw(2, 0);
	_station.set_storage (&_station__storage);
	_cscid__storage.bw(3, 0);
	_cscid.bw(3, 0);
	_cscid.set_storage (&_cscid__storage);
	_bend__storage.bw(3, 0);
	_bend.bw(3, 0);
	_bend.set_storage (&_bend__storage);
	_halfstrip__storage.bw(7, 0);
	_halfstrip.bw(7, 0);
	_halfstrip.set_storage (&_halfstrip__storage);
	_valid__storage.bw(0, 0);
	_valid.bw(0, 0);
	_valid.set_storage (&_valid__storage);
	_quality__storage.bw(3, 0);
	_quality.bw(3, 0);
	_quality.set_storage (&_quality__storage);
	_pattern__storage.bw(3, 0);
	_pattern.bw(3, 0);
	_pattern.set_storage (&_pattern__storage);
	_wiregroup__storage.bw(6, 0);
	_wiregroup.bw(6, 0);
	_wiregroup.set_storage (&_wiregroup__storage);
	line__storage.bw(800, 1);
	line.bw(800, 1);
	line.set_storage (&line__storage);
	ev__storage.bw(9, 0);
	ev.bw(9, 0);
	ev.set_storage (&ev__storage);
	good_ev__storage.bw(4, 0);
	good_ev.bw(4, 0);
	good_ev.set_storage (&good_ev__storage);
	tphi__storage.bw(11, 0);
	tphi.bw(11, 0);
	tphi.set_storage (&tphi__storage);
	a__storage.bw(11, 0);
	a.bw(11, 0);
	a.set_storage (&a__storage);
	b__storage.bw(11, 0);
	b.bw(11, 0);
	b.set_storage (&b__storage);
	d__storage.bw(11, 0);
	d.bw(11, 0);
	d.set_storage (&d__storage);
	pts__storage.bw(0, 0);
	pts.bw(0, 0);
	pts.set_storage (&pts__storage);
	r_outo__storage.bw(11, 0);
	r_outo.bw(11, 0);
	r_outo.set_storage (&r_outo__storage);
	ph_ranko__storage.add_dim(3, 0);
	ph_ranko__storage.add_dim(_ph_raw_w-1, 0);
	ph_ranko__storage.bw(5, 0);
	ph_ranko__storage.build();
	ph_ranko.add_dim(3, 0);
	ph_ranko.add_dim(_ph_raw_w-1, 0);
	ph_ranko.bw(5, 0);
	ph_ranko.build();
	ph_ranko.set_storage (&ph_ranko__storage);
	ph__storage.add_dim(4, 0);
	ph__storage.add_dim(8, 0);
	ph__storage.add_dim(_seg_ch-1, 0);
	ph__storage.bw(_bw_fph-1, 0);
	ph__storage.build();
	ph.add_dim(4, 0);
	ph.add_dim(8, 0);
	ph.add_dim(_seg_ch-1, 0);
	ph.bw(_bw_fph-1, 0);
	ph.build();
	ph.set_storage (&ph__storage);
	th11__storage.add_dim(1, 0);
	th11__storage.add_dim(2, 0);
	th11__storage.add_dim(_th_ch11-1, 0);
	th11__storage.bw(_bw_th-1, 0);
	th11__storage.build();
	th11.add_dim(1, 0);
	th11.add_dim(2, 0);
	th11.add_dim(_th_ch11-1, 0);
	th11.bw(_bw_th-1, 0);
	th11.build();
	th11.set_storage (&th11__storage);
	th__storage.add_dim(4, 0);
	th__storage.add_dim(8, 0);
	th__storage.add_dim(_seg_ch-1, 0);
	th__storage.bw(_bw_th-1, 0);
	th__storage.build();
	th.add_dim(4, 0);
	th.add_dim(8, 0);
	th.add_dim(_seg_ch-1, 0);
	th.bw(_bw_th-1, 0);
	th.build();
	th.set_storage (&th__storage);
	vl__storage.add_dim(4, 0);
	vl__storage.add_dim(8, 0);
	vl__storage.bw(_seg_ch-1, 0);
	vl__storage.build();
	vl.add_dim(4, 0);
	vl.add_dim(8, 0);
	vl.bw(_seg_ch-1, 0);
	vl.build();
	vl.set_storage (&vl__storage);
	phzvl__storage.add_dim(4, 0);
	phzvl__storage.add_dim(8, 0);
	phzvl__storage.bw(2, 0);
	phzvl__storage.build();
	phzvl.add_dim(4, 0);
	phzvl.add_dim(8, 0);
	phzvl.bw(2, 0);
	phzvl.build();
	phzvl.set_storage (&phzvl__storage);
	me11a__storage.add_dim(1, 0);
	me11a__storage.add_dim(2, 0);
	me11a__storage.bw(_seg_ch-1, 0);
	me11a__storage.build();
	me11a.add_dim(1, 0);
	me11a.add_dim(2, 0);
	me11a.bw(_seg_ch-1, 0);
	me11a.build();
	me11a.set_storage (&me11a__storage);
	ph_zone__storage.add_dim(3, 0);
	ph_zone__storage.add_dim(4, 1);
	ph_zone__storage.bw(_ph_raw_w-1, 0);
	ph_zone__storage.build();
	ph_zone.add_dim(3, 0);
	ph_zone.add_dim(4, 1);
	ph_zone.bw(_ph_raw_w-1, 0);
	ph_zone.build();
	ph_zone.set_storage (&ph_zone__storage);
	patt_vi__storage.add_dim(3, 0);
	patt_vi__storage.add_dim(2, 0);
	patt_vi__storage.add_dim(3, 0);
	patt_vi__storage.bw(_seg_ch-1, 0);
	patt_vi__storage.build();
	patt_vi.add_dim(3, 0);
	patt_vi.add_dim(2, 0);
	patt_vi.add_dim(3, 0);
	patt_vi.bw(_seg_ch-1, 0);
	patt_vi.build();
	patt_vi.set_storage (&patt_vi__storage);
	patt_hi__storage.add_dim(3, 0);
	patt_hi__storage.add_dim(2, 0);
	patt_hi__storage.add_dim(3, 0);
	patt_hi__storage.bw(1, 0);
	patt_hi__storage.build();
	patt_hi.add_dim(3, 0);
	patt_hi.add_dim(2, 0);
	patt_hi.add_dim(3, 0);
	patt_hi.bw(1, 0);
	patt_hi.build();
	patt_hi.set_storage (&patt_hi__storage);
	patt_ci__storage.add_dim(3, 0);
	patt_ci__storage.add_dim(2, 0);
	patt_ci__storage.add_dim(3, 0);
	patt_ci__storage.bw(2, 0);
	patt_ci__storage.build();
	patt_ci.add_dim(3, 0);
	patt_ci.add_dim(2, 0);
	patt_ci.add_dim(3, 0);
	patt_ci.bw(2, 0);
	patt_ci.build();
	patt_ci.set_storage (&patt_ci__storage);
	patt_si__storage.add_dim(3, 0);
	patt_si__storage.add_dim(2, 0);
	patt_si__storage.bw(3, 0);
	patt_si__storage.build();
	patt_si.add_dim(3, 0);
	patt_si.add_dim(2, 0);
	patt_si.bw(3, 0);
	patt_si.build();
	patt_si.set_storage (&patt_si__storage);
	ph_num__storage.add_dim(3, 0);
	ph_num__storage.add_dim(2, 0);
	ph_num__storage.bw(_bpow, 0);
	ph_num__storage.build();
	ph_num.add_dim(3, 0);
	ph_num.add_dim(2, 0);
	ph_num.bw(_bpow, 0);
	ph_num.build();
	ph_num.set_storage (&ph_num__storage);
	ph_q__storage.add_dim(3, 0);
	ph_q__storage.add_dim(2, 0);
	ph_q__storage.bw(_bwr-1, 0);
	ph_q__storage.build();
	ph_q.add_dim(3, 0);
	ph_q.add_dim(2, 0);
	ph_q.bw(_bwr-1, 0);
	ph_q.build();
	ph_q.set_storage (&ph_q__storage);
	ph_match__storage.add_dim(3, 0);
	ph_match__storage.add_dim(2, 0);
	ph_match__storage.add_dim(3, 0);
	ph_match__storage.bw(_bw_fph-1, 0);
	ph_match__storage.build();
	ph_match.add_dim(3, 0);
	ph_match.add_dim(2, 0);
	ph_match.add_dim(3, 0);
	ph_match.bw(_bw_fph-1, 0);
	ph_match.build();
	ph_match.set_storage (&ph_match__storage);
	th_match__storage.add_dim(3, 0);
	th_match__storage.add_dim(2, 0);
	th_match__storage.add_dim(3, 0);
	th_match__storage.add_dim(_seg_ch-1, 0);
	th_match__storage.bw(_bw_th-1, 0);
	th_match__storage.build();
	th_match.add_dim(3, 0);
	th_match.add_dim(2, 0);
	th_match.add_dim(3, 0);
	th_match.add_dim(_seg_ch-1, 0);
	th_match.bw(_bw_th-1, 0);
	th_match.build();
	th_match.set_storage (&th_match__storage);
	th_match11__storage.add_dim(1, 0);
	th_match11__storage.add_dim(2, 0);
	th_match11__storage.add_dim(_th_ch11-1, 0);
	th_match11__storage.bw(_bw_th-1, 0);
	th_match11__storage.build();
	th_match11.add_dim(1, 0);
	th_match11.add_dim(2, 0);
	th_match11.add_dim(_th_ch11-1, 0);
	th_match11.bw(_bw_th-1, 0);
	th_match11.build();
	th_match11.set_storage (&th_match11__storage);
	bt_phi__storage.add_dim(2, 0);
	bt_phi__storage.bw(_bw_fph-1, 0);
	bt_phi__storage.build();
	bt_phi.add_dim(2, 0);
	bt_phi.bw(_bw_fph-1, 0);
	bt_phi.build();
	bt_phi.set_storage (&bt_phi__storage);
	bt_theta__storage.add_dim(2, 0);
	bt_theta__storage.bw(_bw_th-1, 0);
	bt_theta__storage.build();
	bt_theta.add_dim(2, 0);
	bt_theta.bw(_bw_th-1, 0);
	bt_theta.build();
	bt_theta.set_storage (&bt_theta__storage);
	bt_cpattern__storage.add_dim(2, 0);
	bt_cpattern__storage.bw(3, 0);
	bt_cpattern__storage.build();
	bt_cpattern.add_dim(2, 0);
	bt_cpattern.bw(3, 0);
	bt_cpattern.build();
	bt_cpattern.set_storage (&bt_cpattern__storage);
	bt_delta_ph__storage.add_dim(2, 0);
	bt_delta_ph__storage.add_dim(1, 0);
	bt_delta_ph__storage.bw(_bw_fph-1, 0);
	bt_delta_ph__storage.build();
	bt_delta_ph.add_dim(2, 0);
	bt_delta_ph.add_dim(1, 0);
	bt_delta_ph.bw(_bw_fph-1, 0);
	bt_delta_ph.build();
	bt_delta_ph.set_storage (&bt_delta_ph__storage);
	bt_delta_th__storage.add_dim(2, 0);
	bt_delta_th__storage.add_dim(1, 0);
	bt_delta_th__storage.bw(_bw_th-1, 0);
	bt_delta_th__storage.build();
	bt_delta_th.add_dim(2, 0);
	bt_delta_th.add_dim(1, 0);
	bt_delta_th.bw(_bw_th-1, 0);
	bt_delta_th.build();
	bt_delta_th.set_storage (&bt_delta_th__storage);
	bt_sign_ph__storage.add_dim(2, 0);
	bt_sign_ph__storage.bw(1, 0);
	bt_sign_ph__storage.build();
	bt_sign_ph.add_dim(2, 0);
	bt_sign_ph.bw(1, 0);
	bt_sign_ph.build();
	bt_sign_ph.set_storage (&bt_sign_ph__storage);
	bt_sign_th__storage.add_dim(2, 0);
	bt_sign_th__storage.bw(1, 0);
	bt_sign_th__storage.build();
	bt_sign_th.add_dim(2, 0);
	bt_sign_th.bw(1, 0);
	bt_sign_th.build();
	bt_sign_th.set_storage (&bt_sign_th__storage);
	bt_rank__storage.add_dim(2, 0);
	bt_rank__storage.bw(_bwr, 0);
	bt_rank__storage.build();
	bt_rank.add_dim(2, 0);
	bt_rank.bw(_bwr, 0);
	bt_rank.build();
	bt_rank.set_storage (&bt_rank__storage);
	bt_vi__storage.add_dim(2, 0);
	bt_vi__storage.add_dim(4, 0);
	bt_vi__storage.bw(_seg_ch-1, 0);
	bt_vi__storage.build();
	bt_vi.add_dim(2, 0);
	bt_vi.add_dim(4, 0);
	bt_vi.bw(_seg_ch-1, 0);
	bt_vi.build();
	bt_vi.set_storage (&bt_vi__storage);
	bt_hi__storage.add_dim(2, 0);
	bt_hi__storage.add_dim(4, 0);
	bt_hi__storage.bw(1, 0);
	bt_hi__storage.build();
	bt_hi.add_dim(2, 0);
	bt_hi.add_dim(4, 0);
	bt_hi.bw(1, 0);
	bt_hi.build();
	bt_hi.set_storage (&bt_hi__storage);
	bt_ci__storage.add_dim(2, 0);
	bt_ci__storage.add_dim(4, 0);
	bt_ci__storage.bw(3, 0);
	bt_ci__storage.build();
	bt_ci.add_dim(2, 0);
	bt_ci.add_dim(4, 0);
	bt_ci.bw(3, 0);
	bt_ci.build();
	bt_ci.set_storage (&bt_ci__storage);
	bt_si__storage.add_dim(2, 0);
	bt_si__storage.bw(4, 0);
	bt_si__storage.build();
	bt_si.add_dim(2, 0);
	bt_si.bw(4, 0);
	bt_si.build();
	bt_si.set_storage (&bt_si__storage);
	iadr__storage.bw(31, 0);
	iadr.bw(31, 0);
	iadr.set_storage (&iadr__storage);
	s__storage.bw(31, 0);
	s.bw(31, 0);
	s.set_storage (&s__storage);
	i__storage.bw(31, 0);
	i.bw(31, 0);
	i.set_storage (&i__storage);
	pi__storage.bw(31, 0);
	pi.bw(31, 0);
	pi.set_storage (&pi__storage);
	j__storage.bw(31, 0);
	j.bw(31, 0);
	j.set_storage (&j__storage);
	sn__storage.bw(31, 0);
	sn.bw(31, 0);
	sn.set_storage (&sn__storage);
	ist__storage.bw(31, 0);
	ist.bw(31, 0);
	ist.set_storage (&ist__storage);
	icid__storage.bw(31, 0);
	icid.bw(31, 0);
	icid.set_storage (&icid__storage);
	ipr__storage.bw(31, 0);
	ipr.bw(31, 0);
	ipr.set_storage (&ipr__storage);
	code__storage.bw(31, 0);
	code.bw(31, 0);
	code.set_storage (&code__storage);
	iev__storage.bw(31, 0);
	iev.bw(31, 0);
	iev.set_storage (&iev__storage);
	im__storage.bw(31, 0);
	im.bw(31, 0);
	im.set_storage (&im__storage);
	iz__storage.bw(31, 0);
	iz.bw(31, 0);
	iz.set_storage (&iz__storage);
	ir__storage.bw(31, 0);
	ir.bw(31, 0);
	ir.set_storage (&ir__storage);
	in__storage.bw(31, 0);
	in.bw(31, 0);
	in.set_storage (&in__storage);
	best_tracks__storage.bw(31, 0);
	best_tracks.bw(31, 0);
	best_tracks.set_storage (&best_tracks__storage);
	stat__storage.bw(31, 0);
	stat.bw(31, 0);
	stat.set_storage (&stat__storage);
	good_ev_cnt__storage.bw(31, 0);
	good_ev_cnt.bw(31, 0);
	good_ev_cnt.set_storage (&good_ev_cnt__storage);
	found_tr__storage.bw(31, 0);
	found_tr.bw(31, 0);
	found_tr.set_storage (&found_tr__storage);
	found_cand__storage.bw(31, 0);
	found_cand.bw(31, 0);
	found_cand.set_storage (&found_cand__storage);
	st__storage.bw(31, 0);
	st.bw(31, 0);
	st.set_storage (&st__storage);
	st_cnt__storage.bw(31, 0);
	st_cnt.bw(31, 0);
	st_cnt.set_storage (&st_cnt__storage);
	iseg__storage.bw(31, 0);
	iseg.bw(31, 0);
	iseg.set_storage (&iseg__storage);
	zi__storage.bw(31, 0);
	zi.bw(31, 0);
	zi.set_storage (&zi__storage);
	si__storage.bw(31, 0);
	si.bw(31, 0);
	si.set_storage (&si__storage);
	ip__storage.bw(31, 0);
	ip.bw(31, 0);
	ip.set_storage (&ip__storage);
	ibx__storage.bw(31, 0);
	ibx.bw(31, 0);
	ibx.set_storage (&ibx__storage);
	ich__storage.bw(31, 0);
	ich.bw(31, 0);
	ich.set_storage (&ich__storage);
	isg__storage.bw(31, 0);
	isg.bw(31, 0);
	isg.set_storage (&isg__storage);
	ii__storage.bw(31, 0);
	ii.bw(31, 0);
	ii.set_storage (&ii__storage);
	kp__storage.bw(31, 0);
	kp.bw(31, 0);
	kp.set_storage (&kp__storage);
	begin_time__storage.bw(31, 0);
	begin_time.bw(31, 0);
	begin_time.set_storage (&begin_time__storage);
	end_time__storage.bw(31, 0);
	end_time.bw(31, 0);
	end_time.set_storage (&end_time__storage);
	elapsed_time__storage.bw(31, 0);
	elapsed_time.bw(31, 0);
	elapsed_time.set_storage (&elapsed_time__storage);
	ev = 0;
	good_ev = 0;

}

// vppc: this function checks for changes in any signal on each simulation iteration
void sptf::init ()
{
	if (!built)
	{
				}
	else
	{
		qi__storage.init();
    wgi__storage.init();
    hstri__storage.init();
		cpati__storage.init();
		csi__storage.init();
		pps_csi__storage.init();
		seli__storage.init();
		addri__storage.init();
		r_ini__storage.init();
		wei__storage.init();
		clki__storage.init();
		ph_init__storage.init();
		th_init__storage.init();
		ph_disp__storage.init();
		th_disp__storage.init();
		quality__storage.init();
		wiregroup__storage.init();
		hstrip__storage.init();
		clctpat__storage.init();
		v0__storage.init();
		v1__storage.init();
		v2__storage.init();
		v3__storage.init();
		v4__storage.init();
		v5__storage.init();
		pr_cnt__storage.init();
		_event__storage.init();
		_bx_jitter__storage.init();
		_endcap__storage.init();
		_sector__storage.init();
		_subsector__storage.init();
		_station__storage.init();
		_cscid__storage.init();
		_bend__storage.init();
		_halfstrip__storage.init();
		_valid__storage.init();
		_quality__storage.init();
		_pattern__storage.init();
		_wiregroup__storage.init();
		line__storage.init();
		ev__storage.init();
		good_ev__storage.init();
		tphi__storage.init();
		a__storage.init();
		b__storage.init();
		d__storage.init();
		pts__storage.init();
		r_outo__storage.init();
		ph_ranko__storage.init();
		ph__storage.init();
		th11__storage.init();
		th__storage.init();
		vl__storage.init();
		phzvl__storage.init();
		me11a__storage.init();
		ph_zone__storage.init();
		patt_vi__storage.init();
		patt_hi__storage.init();
		patt_ci__storage.init();
		patt_si__storage.init();
		ph_num__storage.init();
		ph_q__storage.init();
		ph_match__storage.init();
		th_match__storage.init();
		th_match11__storage.init();
		bt_phi__storage.init();
		bt_theta__storage.init();
		bt_cpattern__storage.init();
		bt_delta_ph__storage.init();
		bt_delta_th__storage.init();
		bt_sign_ph__storage.init();
		bt_sign_th__storage.init();
		bt_rank__storage.init();
		bt_vi__storage.init();
		bt_hi__storage.init();
		bt_ci__storage.init();
		bt_si__storage.init();
		uut.init();

	}
}



//void sptf::beginRun(const edm::Run& ir, const edm::EventSetup& es)

void sptf::beginJob()
{

///    cout << "In beginRun, doing setup.\n";

	

    // Allocating space for 5 events in each double vector variable
    for(int i=0; i<5; i++)
    {
        the_primSelector.push_back(vector<int>());
        the_inputOrder.push_back(vector<int>());
        the_bx_jitter.push_back(vector<int>());
        the_endcap.push_back(vector<int>());
        the_sector.push_back(vector<int>());
        the_subsector.push_back(vector<int>());
        the_station.push_back(vector<int>());
        the_valid.push_back(vector<int>());
        the_quality.push_back(vector<int>());
        the_pattern.push_back(vector<int>());
        the_wiregroup.push_back(vector<int>());
        the_cscid.push_back(vector<int>());
        the_bend.push_back(vector<int>());
        the_halfstrip.push_back(vector<int>());
    }

    sim_lib_init();

//// Beginning of code that was originally inside if(!built) ////////////

    build();

//////////  End of code that was original part of if(!built)  ///////////
   
    iadr = 0;
    s = 0;
    i = 0;
    j = 0;
    good_ev_cnt = 0;
    found_tr = 0;
    found_cand = 0;

    perm_i = 0;
    uut(qi, wgi, hstri, cpati, csi, pps_csi, seli, addri, r_ini, r_outo, wei, bt_phi, bt_theta, bt_cpattern,
        bt_delta_ph, bt_delta_th, bt_sign_ph, bt_sign_th, bt_rank, bt_vi, bt_hi, bt_ci, bt_si, clki, clki);
        {
            // fill th LUTs
			
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_corr_lut_endcap_1_sec_1_sub_1_st_1_ch_1.lut").fullPath().c_str(), uut.pcs.genblk.station11[0].csc11[0].pc11.th_corr_mem);
			Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_corr_lut_endcap_1_sec_1_sub_1_st_1_ch_2.lut").fullPath().c_str(), uut.pcs.genblk.station11[0].csc11[1].pc11.th_corr_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_corr_lut_endcap_1_sec_1_sub_1_st_1_ch_3.lut").fullPath().c_str(), uut.pcs.genblk.station11[0].csc11[2].pc11.th_corr_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_corr_lut_endcap_1_sec_1_sub_2_st_1_ch_1.lut").fullPath().c_str(), uut.pcs.genblk.station11[1].csc11[0].pc11.th_corr_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_corr_lut_endcap_1_sec_1_sub_2_st_1_ch_2.lut").fullPath().c_str(), uut.pcs.genblk.station11[1].csc11[1].pc11.th_corr_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_corr_lut_endcap_1_sec_1_sub_2_st_1_ch_3.lut").fullPath().c_str(), uut.pcs.genblk.station11[1].csc11[2].pc11.th_corr_mem);

            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_10.lut").fullPath().c_str(), uut.pcs.genblk.station11[0].csc11[0].pc11.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_11.lut").fullPath().c_str(), uut.pcs.genblk.station11[0].csc11[1].pc11.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_12.lut").fullPath().c_str(), uut.pcs.genblk.station11[0].csc11[2].pc11.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_4.lut").fullPath().c_str(), uut.pcs.genblk.station12[0].csc12[3].pc12.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_5.lut").fullPath().c_str(), uut.pcs.genblk.station12[0].csc12[4].pc12.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_6.lut").fullPath().c_str(), uut.pcs.genblk.station12[0].csc12[5].pc12.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_7.lut").fullPath().c_str(), uut.pcs.genblk.station12[0].csc12[6].pc12.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_8.lut").fullPath().c_str(), uut.pcs.genblk.station12[0].csc12[7].pc12.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_1_st_1_ch_9.lut").fullPath().c_str(), uut.pcs.genblk.station12[0].csc12[8].pc12.th_mem);

            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_10.lut").fullPath().c_str(), uut.pcs.genblk.station11[1].csc11[0].pc11.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_11.lut").fullPath().c_str(), uut.pcs.genblk.station11[1].csc11[1].pc11.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_12.lut").fullPath().c_str(), uut.pcs.genblk.station11[1].csc11[2].pc11.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_4.lut").fullPath().c_str(), uut.pcs.genblk.station12[1].csc12[3].pc12.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_5.lut").fullPath().c_str(), uut.pcs.genblk.station12[1].csc12[4].pc12.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_6.lut").fullPath().c_str(), uut.pcs.genblk.station12[1].csc12[5].pc12.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_7.lut").fullPath().c_str(), uut.pcs.genblk.station12[1].csc12[6].pc12.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_8.lut").fullPath().c_str(), uut.pcs.genblk.station12[1].csc12[7].pc12.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_sub_2_st_1_ch_9.lut").fullPath().c_str(), uut.pcs.genblk.station12[1].csc12[8].pc12.th_mem);

            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_1.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[0].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_2.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[1].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_3.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[2].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_4.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[3].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_5.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[4].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_6.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[5].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_7.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[6].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_8.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[7].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_2_ch_9.lut").fullPath().c_str(), uut.pcs.genblk.station[2].csc[8].pc.th_mem);

            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_1.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[0].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_2.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[1].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_3.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[2].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_4.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[3].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_5.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[4].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_6.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[5].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_7.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[6].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_8.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[7].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_3_ch_9.lut").fullPath().c_str(), uut.pcs.genblk.station[3].csc[8].pc.th_mem);

            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_1.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[0].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_2.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[1].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_3.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[2].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_4.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[3].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_5.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[4].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_6.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[5].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_7.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[6].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_8.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[7].pc.th_mem);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/vl_th_lut_endcap_1_sec_1_st_4_ch_9.lut").fullPath().c_str(), uut.pcs.genblk.station[4].csc[8].pc.th_mem);

            //Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/ph_init_endcap_1_sect_1.lut").fullPath().c_str(), ph_init);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/th_init_endcap_1_sect_1.lut").fullPath().c_str(), th_init);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/ph_disp_endcap_1_sect_1.lut").fullPath().c_str(), ph_disp);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/th_disp_endcap_1_sect_1.lut").fullPath().c_str(), th_disp);

            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/ph_init_full_endcap_1_sect_1_st_0.lut").fullPath().c_str(), ph_init[0]);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/ph_init_full_endcap_1_sect_1_st_1.lut").fullPath().c_str(), ph_init[1]);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/ph_init_full_endcap_1_sect_1_st_2.lut").fullPath().c_str(), ph_init[2]);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/ph_init_full_endcap_1_sect_1_st_3.lut").fullPath().c_str(), ph_init[3]);
            Sreadmemh(edm::FileInPath("L1Trigger/L1TMuonEndCap/src/core_gen_summer_2013/vl_lut/ph_init_full_endcap_1_sect_1_st_4.lut").fullPath().c_str(), ph_init[4]);

            // initialize event storage
            for (iev = 0; iev < _max_ev; iev=iev+1)
                for (ist = 0; ist < 5; ist=ist+1)
                    for (icid = 0; icid < 9; icid=icid+1)
                        for (ipr = 0; ipr < _seg_ch; ipr=ipr+1)
                        {
                            quality  [iev][ist][icid][ipr] = const_(4, 0x0UL);
                            wiregroup[iev][ist][icid][ipr] = const_(7, 0x0UL);
                            hstrip   [iev][ist][icid][ipr] = const_(8, 0x0UL);
                            clctpat  [iev][ist][icid][ipr] = const_(4, 0x0UL);
                        }

        }

}


//void sptf::endRun(const edm::Run& ir, const edm::EventSetup& es)
//{
//}

//void sptf::beginJob()
//{
//}

void sptf::endJob()
{

	

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(sptf);

