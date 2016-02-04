void test_HVP5Histo(){

/*connect with oracle server */
TSQLServer *db=TSQLServer::Connect("oracle://pccmsecdb:1521/omds","cms_ecal_hv","zonep5mtcc");


 Int_t sm=13; // from 0 to 36
 Int_t chan_num=1; // from 1 to 34
 char table_name[20]="PVSS_HV_IMON_DAT";
 char field_name[20]="IMON";
 
 Float_t curmax=5; // max of the histogram

 Int_t sm_num=1041000000+10000*sm+sm;

 gStyle->SetOptStat(0);
 gStyle->SetOptFit();
 gStyle->SetPalette(1,0);
 TDatime da(2005,01,01,00,00,00);
 gStyle->SetTimeOffset(da.Convert());

 char titolo[100];
 sprintf(titolo,"%s.%s for chan %d of SM %d",table_name,field_name,chan_num, sm );
 
 
 c1 = new TCanvas("c1","History plot from DB",200,10,600,400);
 c1->SetGrid();
 //  c1->Divide(2,3);

/* c2 = new TCanvas("c2","The HV evolution in time",200,10,600,400);
c2->SetGrid();
c3 = new TCanvas("c3","The HV versus channel",200,10,600,400);
c3->SetGrid();
c4 = new TCanvas("c4","The Current",200,10,600,400);
c4->SetGrid();
c5 = new TCanvas("c5","The HV Current evolution in time",200,10,600,400);
c5->SetGrid();
c6 = new TCanvas("c6","The HV Current versus channel",200,10,600,400);
c6->SetGrid();
*/

float tStart=520;
float tEnd=620;
 float day_to_sec=24.*60.*60.;
 tStart=(tStart-2)*day_to_sec;
 tEnd=(tEnd+2)*day_to_sec;
 int nbins= (tEnd-tStart)/day_to_sec;
 hv_time_vs_vmon  = new TH2F("temp_vs_time",titolo,nbins,tStart ,tEnd , 100, 0.,curmax );

 hv_time_vs_vmon->GetXaxis()->SetTitle("Time");
 hv_time_vs_vmon->GetYaxis()->SetTitle(field_name);
 hv_time_vs_vmon->GetXaxis()->SetTimeDisplay(1);  // The X axis is a time axis 
 hv_time_vs_vmon->GetXaxis()->SetTimeFormat("%d-%m-%y");
 hv_time_vs_vmon->GetXaxis()->SetLabelSize(0.02);
 hv_time_vs_vmon->SetMarkerStyle(20);
 hv_time_vs_vmon->SetMarkerSize(0.7);
 // 

// hv_imon  = new TH1F("hv_imon","HV IMON",100,0.,10.);
// hv_time_vs_imon  = new TH2F("hv_time_vs_imon","HV TIME VS IMON",100,1.E+12,10.E+12,100,0.,10.);
hv_vmon  = new TH1F("hv_vmon","HV VMON",100,0.,500.);
// hv_time_vs_vmon  = new TH2F("hv_time_vs_vmon","HV TIME VS VMON",100,1.E+12,10.E+12,100,0.,500.);
// hv_chan_vs_imon  = new TH2F("hv_chan_vs_imon","HV CHAN VS IMON", 17, 0.5, 17.5,2,0.5, 2.5 );
// hv_chan_vs_vmon  = new TH2F("hv_chan_vs_vmon","HV CHAN VS VMON", 17, 0.5, 17.5, 2,0.5, 2.5 );
// hv_chan_vs_v0  = new TH2F("hv_chan_vs_v0","HV CHAN VS V0", 17, 0.5, 17.5, 2,0.5, 2.5 );
//  hv_chan_vs_diff  = new TH2F("hv_chan_vs_v0","HV CHAN VS V0", 17, 0.5, 17.5, 2,0.5, 2.5 );

/* query  */
 char sql[750];
 char query_skeleton4[400];
 char query_skeleton4a[50]="SELECT CAST( ";
 char query_skeleton4b[200]=" AS NUMBER), (SINCE-to_date('01-JAN-2005 00:00:00','DD-MON-YYYY HH24:MI:SS')), (till-to_date('31-DEC-9999 23:59:59','DD-MON-YYYY HH24:MI:SS'))  FROM ";
 char query_skeleton1[25]=" , channelview where ";
 char query_skeleton2[100]=" .logic_id=channelview.logic_id and channelview.name=channelview.maps_to and channelview.id1= ";
 char query_skeleton5[40]=" and channelview.id2= ";
 char query_skeleton3[40]=" order by since ";

 sprintf(query_skeleton4,"%s%s.%s%s%s",query_skeleton4a,table_name,field_name,query_skeleton4b,table_name);

  sprintf(sql,"%s%s%s%s%d%s%d%s",query_skeleton4, query_skeleton1,table_name,
	  query_skeleton2,sm,query_skeleton5,chan_num,query_skeleton3);

  cout << sql << endl ;

  // char * sql="select logic_id,  Since-to_date('01-JAN-2005 00:00:00','DD-MON-YYYY HH24:MI:SS') , till, vmon from pvss_hv_imon_dat where logic_id= order by logic_id, since";


TSQLResult *res=db->Query(sql);   

 float vmon;
 float tmon;
 float ttill;
 
 do {int j; 
 TSQLRow *row1=res->Next();
 TOracleRow * row2=(TOracleRow *)row1;  

 for (int i=0; i<res->GetFieldCount();i++) {
   printf(" %*.*s ",row2->GetFieldLength(i),row2->GetFieldLength(i),row2->GetField(i)); 

   if(i==0) vmon=atof( row2->GetField(i));
   if(i==1) tmon=atof( row2->GetField(i))*day_to_sec;
   if(i==2) ttill=atof( row2->GetField(i))*day_to_sec;
 }
 cout <<  endl;
 Float_t tpippo= (Float_t)vmon;
 Float_t tt= (Float_t)tmon;
 hv_vmon->Fill(tpippo);
 hv_time_vs_vmon->Fill(tt, tpippo,1.);
 cout << tpippo<< tmon<< ttill << endl; 
 delete row2;
 
 } while (ttill<0);
 
 //  c1->cd(1);
 //hv_vmon->Draw();
 //c1->Update();
 
 //c1->cd(3);
 hv_time_vs_vmon->Draw();
 c1->Update();

 /* 
 // now actual Idark plot
 
char * sql_act_id="select logic_id-1051110000, till,  value00 from COND_HV_IMON where logic_id>1051110000 and till>9E+15 order by logic_id";
TSQLResult *res_act_id=db->Query(sql_act_id);   

 float idark=0.;
 int chan=0;

 do{ int j;
 TSQLRow *row1=res_act_id->Next();
 TOracleRow * row2=(TOracleRow *)row1;  
 
 for (int i=0; i<res_act_id->GetFieldCount();i++) {
   printf(" %*.*s ",row2->GetFieldLength(i),row2->GetFieldLength(i),row2->GetField(i)); 
   if(i==0) chan=atoi( row2->GetField(i));
   if(i==2) idark=atof( row2->GetField(i));
 }
 cout <<  endl;


 Float_t tpippo= (Float_t) idark;
  Float_t tchanx= (Float_t) (2-chan%2) ;
 Float_t tchany= (Float_t) ((chan-1)/2+1) ;
 hv_chan_vs_imon->Fill(tchany, tchanx, tpippo);
 cout << chan << " " << tchanx << " " << tchany << endl; 
 // delete row2;
 } while (chan!=34);
 
 c1->cd(5);
 hv_chan_vs_imon->SetMaximum(10.0);
 hv_chan_vs_imon->Draw("colz");
 c1->Update();


 // now actual HV plot 
char * sql_act_hv="select logic_id-1051110000,  till, value00 from COND_HV_VMON where logic_id>1051110000 and till>9E+15 order by logic_id";
TSQLResult *res_act_hv=db->Query(sql_act_hv);   
 Float_t data[35];

 do{ int j;
 TSQLRow *row1=res_act_hv->Next();
 TOracleRow * row2=(TOracleRow *)row1;  
 
 for (int i=0; i<res_act_hv->GetFieldCount();i++) {
   printf(" %*.*s ",row2->GetFieldLength(i),row2->GetFieldLength(i),row2->GetField(i)); 
   if(i==0) chan=atoi( row2->GetField(i));
   if(i==2) idark=atof( row2->GetField(i));
 }
 cout <<  endl;

 Float_t tpippo= (Float_t) idark;
 data[chan]=tpippo;
 Float_t tchanx= (Float_t) (2-chan%2) ;
 Float_t tchany= (Float_t) ((chan-1)/2+1) ;
 hv_chan_vs_vmon->Fill(tchany, tchanx, tpippo);


 cout << chan << " " << tchanx << " " << tchany << endl; 
 delete row2;

 } while (chan!=34);
 
 // now actual HV V0 plot 
char * sql_act_v0="select logic_id-1051110000,  till, value00 from COND_HV_V0 where logic_id>1051110000 and till>9E+15 order by logic_id";
TSQLResult *res_act_v0=db->Query(sql_act_v0);   
 float v0=0.;

do{ int j;
 
 TSQLRow *rows=res_act_v0->Next();
 TOracleRow * row3=(TOracleRow *)rows;  
 
 for (int i=0; i<res_act_v0->GetFieldCount();i++) {
   printf(" %*.*s ",row3->GetFieldLength(i),row3->GetFieldLength(i),row3->GetField(i)); 
   if(i==0) chan=atoi( row3->GetField(i));
   if(i==2) v0=atof( row3->GetField(i));   
 }
 cout <<  endl;

 
 Float_t tv0   = (Float_t) v0;
 Float_t tchanx= (Float_t) (2-chan%2) ;
 Float_t tchany= (Float_t) ((chan-1)/2+1) ;
 Float_t diff=data[chan]-tv0;
 
 hv_chan_vs_diff->Fill(tchany, tchanx, diff);
 hv_chan_vs_v0->Fill(tchany, tchanx, tv0);
 
 cout << chan << " " << tchanx << " " << tchany << endl; 
 
 delete row3;
 } while (chan!=34);
 


 c1->cd(4);
 hv_chan_vs_vmon->SetMaximum(450.0);
 hv_chan_vs_vmon->SetMinimum(350.0);
hv_chan_vs_vmon->Draw("colz");

 c1->cd(6);
 hv_chan_vs_diff->SetMaximum(30.0);
 hv_chan_vs_diff->SetMinimum(-30.0);
 hv_chan_vs_diff->Draw("colz");
 c1->cd(2);
 hv_chan_vs_v0->SetMaximum(450.0);
 hv_chan_vs_v0->SetMinimum(350.0);
 hv_chan_vs_v0->Draw("colz");
 c1->Update();

 */ 
 printf("end \n");

 delete res;
 delete db;
 
}
