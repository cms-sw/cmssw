void test_AnyHistory(){

  // select SM number and channel number
  Int_t sm=9; // from 0 to 36
  Int_t chan_num=1036; // from 1 to 1700

  // select table and field

 // this is for BLUE or RED APD/PN or APD Mean 
   //  char table_name[20]="MON_LASER_RED_DAT";
   //   char table_name[20]="MON_LASER_BLUE_DAT";

  // char field_name[20]="APD_OVER_PN_MEAN";
   //    char field_name[20]="APD_MEAN";

 // this is for pedestals RMS
    char table_name[20]="MON_PEDESTALS_DAT";
     char field_name[20]="PED_RMS_G12";
  //  char field_name[20]="PED_RMS_G1";
  // char field_name[20]="PED_RMS_G6";


 Int_t sm_num=1041000000+10000*sm+sm;

/*connect with oracle server */
TSQLServer *db=TSQLServer::Connect("oracle://pccmsecdb:1521/ecalh4db","read01","XXXpasswordXXX");

 gStyle->SetOptStat(0);
 gStyle->SetOptFit();
 gStyle->SetPalette(1,0);

 char titolo[100];
  sprintf(titolo,"%s.%s for chan %d of SM %d",table_name,field_name,chan_num, sm_num );


 c1 = new TCanvas("c1","History plot from DB",200,10,600,400);
 c1->SetGrid();
 // c1->Divide(2,3);

TDatime da(2005,01,01,00,00,00);
gStyle->SetTimeOffset(da.Convert());


/* first see how many rows we have */
 
 char sql_test[620];
 char query_skeleton1[33]="SELECT count(mon_run_iov.iov_id) "; 
 char query_skeleton2[420];

char query_1[20]=" from channelview, ";
char query_2[110]=" , mon_run_iov , run_iov, run_tag, run_dat WHERE run_iov.iov_id=mon_run_iov.RUN_IOV_ID and channelview.id2= ";
char query_2b[27]=" and mon_run_iov.iov_id= ";
char query_3[36]=".iov_id and channelview.logic_id= ";
char query_4[169]=".logic_id and channelview.name=channelview.maps_to and  run_iov.tag_id=run_tag.tag_id and run_tag.LOCATION_ID=1 and run_dat.iov_id=run_iov.iov_id and run_dat.logic_id=";

  sprintf(query_skeleton2,"%s%s%s%d%s%s%s%s%s",query_1,table_name, query_2,chan_num, query_2b, 
	  table_name,  query_3,table_name,  query_4);

  sprintf(sql_test,"%s%s%d",query_skeleton1, query_skeleton2,sm_num);


 TSQLResult *res=db->Query(sql_test);   
 int j=0;
 int nRows=0;

 do {
   j++;
   TSQLRow *row1=res->Next();
   TOracleRow * row2=(TOracleRow *)row1;  
   nRows=atoi( row2->GetField(0));
   delete row2;
   
 } while (j<1);
 cout <<"number of entries in the DB = "<< nRows<< endl; 

 /* now see the starting and end time of the plot */

 char query_skeleton0[90]="SELECT min(SUBRUN_START-to_date('01-JAN-2005 00:00:00','DD-MON-YYYY HH24:MI:SS'))"; 

  sprintf(sql_test,"%s%s%d",query_skeleton0, query_skeleton2,sm_num);

 TSQLResult *res=db->Query(sql_test);   
 int j=0;
 float tStart=0;

 do {
   j++;
   TSQLRow *row1=res->Next();
   TOracleRow * row2=(TOracleRow *)row1;  
   tStart=atof( row2->GetField(0));
   delete row2;
   
 } while (j<1);
 cout <<"tStart= "<< tStart << endl; 

 char query_skeleton3[90]="SELECT max(SUBRUN_START-to_date('01-JAN-2005 00:00:00','DD-MON-YYYY HH24:MI:SS'))"; 

  sprintf(sql_test,"%s%s%d",query_skeleton3, query_skeleton2,sm_num);

 TSQLResult *res=db->Query(sql_test);   
 int j=0;
 float tEnd=0;

 do {
   j++;
   TSQLRow *row1=res->Next();
   TOracleRow * row2=(TOracleRow *)row1;  
   tEnd=atof( row2->GetField(0));
   delete row2;
   
 } while (j<1);
 cout <<"tEnd= "<< tEnd << endl; 

 float day_to_sec=24.*60.*60.;
 tStart=(tStart-2)*day_to_sec;
 tEnd=(tEnd+2)*day_to_sec;
temp_vs_time  = new TH2F("temp_vs_time",titolo, 100,tStart ,tEnd , 100, 0.,100. );
 temp_vs_time->GetXaxis()->SetTitle("Time");
 temp_vs_time->GetYaxis()->SetTitle(field_name);
temp_vs_time->GetXaxis()->SetTimeDisplay(1);  // The X axis is a time axis 
temp_vs_time->GetXaxis()->SetTimeFormat("%d-%m-%y");
 temp_vs_time->GetXaxis()->SetLabelSize(0.02);
 // 

 char query_skeleton4[200];
 char query_skeleton4a[49]="SELECT channelview.id2, CAST( ";
 char query_skeleton4b[110]=" AS NUMBER), (SUBRUN_START-to_date('01-JAN-2005 00:00:00','DD-MON-YYYY HH24:MI:SS')) , channelview.logic_id "; 

  sprintf(query_skeleton4,"%s%s.%s%s",query_skeleton4a,table_name,field_name,query_skeleton4b);

  sprintf(sql_test,"%s%s%d",query_skeleton4, query_skeleton2,sm_num);



 TSQLResult *res=db->Query(sql_test);   
 
 float temp=0;
 float time_meas=0;
 int chan=0;
 int j=0; 
 do {
   j++;
   TSQLRow *row1=res->Next();
   TOracleRow * row2=(TOracleRow *)row1;  
   
   for (int i=0; i<res->GetFieldCount();i++) {
         printf(" %*.*s ",row2->GetFieldLength(i),row2->GetFieldLength(i),row2->GetField(i)); 
     if(i==0) chan=atoi( row2->GetField(i));
     if(i==1) temp=atof( row2->GetField(i));
     if(i==2) time_meas=atof( row2->GetField(i))*day_to_sec;
   }
     cout <<  endl;
   
   Float_t tpippo= (Float_t)temp;
   Float_t tchanx= (Float_t) ((chan-1)/10) ;
   Float_t tchany= (Float_t) ((chan-1)%10) ;
   temp_vs_time->Fill(time_meas, tpippo,1.);
   
   
   //   cout << chan << " " << tchanx << " " << tchany << " " << tpippo << endl; 
   
   
   
   delete row2;
   
 } while (j<nRows);
 cout <<"loop done "<< endl; 

 
 // c1->cd(1);
 //hv_vmon->Draw();
 //c1->Update();

 //  c1->cd(1);


temp_vs_time->SetMarkerStyle(20);
temp_vs_time->SetMarkerSize(0.7);
 temp_vs_time->Draw();
 c1->Update(); 
 

 
 printf("end \n");

 delete res;
 delete db;
 
}
