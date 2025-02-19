void test_HVHisto(){

/*connect with oracle server */
TSQLServer *db=TSQLServer::Connect("oracle://pccmsecdb:1521/ecalh4db","test01","oratest01");

 gStyle->SetOptStat(0);
   gStyle->SetOptFit();
 gStyle->SetPalette(1,0);
c1 = new TCanvas("c1","The HV",200,10,600,400);
c1->SetGrid();
c1->Divide(2,3);

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

hv_imon  = new TH1F("hv_imon","HV IMON",100,0.,10.);
hv_time_vs_imon  = new TH2F("hv_time_vs_imon","HV TIME VS IMON",100,1.E+12,10.E+12,100,0.,10.);
hv_vmon  = new TH1F("hv_vmon","HV VMON",100,0.,500.);
hv_time_vs_vmon  = new TH2F("hv_time_vs_vmon","HV TIME VS VMON",100,1.E+12,10.E+12,100,0.,500.);
hv_chan_vs_imon  = new TH2F("hv_chan_vs_imon","HV CHAN VS IMON", 17, 0.5, 17.5,2,0.5, 2.5 );
hv_chan_vs_vmon  = new TH2F("hv_chan_vs_vmon","HV CHAN VS VMON", 17, 0.5, 17.5, 2,0.5, 2.5 );
hv_chan_vs_v0  = new TH2F("hv_chan_vs_v0","HV CHAN VS V0", 17, 0.5, 17.5, 2,0.5, 2.5 );
hv_chan_vs_diff  = new TH2F("hv_chan_vs_v0","HV CHAN VS V0", 17, 0.5, 17.5, 2,0.5, 2.5 );

/*print rows one by one */


char * sql="select logic_id, since-1.1E+15, till, value00 from COND_HV_VMON where logic_id=1051110034 order by logic_id, since";
TSQLResult *res=db->Query(sql);   

 float vmon;
 float tmon;
 float ttill;
 
 do {int j; 
 TSQLRow *row1=res->Next();
 TOracleRow * row2=(TOracleRow *)row1;  

 for (int i=0; i<res->GetFieldCount();i++) {
   printf(" %*.*s ",row2->GetFieldLength(i),row2->GetFieldLength(i),row2->GetField(i)); 
   if(i==3) vmon=atof( row2->GetField(i));
   if(i==1) tmon=atof( row2->GetField(i));
   if(i==2) ttill=atof( row2->GetField(i));
 }
 cout <<  endl;
 Float_t tpippo= (Float_t)vmon;
 Float_t tt= (Float_t)tmon;
 hv_vmon->Fill(tpippo);
 hv_time_vs_vmon->Fill(tt, tpippo,1.);
 cout << tpippo<< tmon<< ttill << endl; 
 delete row2;
 
 } while (ttill<5E+15);
 
 c1->cd(1);
 hv_vmon->Draw();
 c1->Update();

 c1->cd(3);
 hv_time_vs_vmon->Draw();
 c1->Update();

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

 
 printf("end \n");

 delete res;
 delete db;
 
}
