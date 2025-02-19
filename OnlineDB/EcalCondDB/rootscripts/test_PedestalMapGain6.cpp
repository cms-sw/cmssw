void test_PedestalMapGain6(){

/*connect with oracle server */
TSQLServer *db=TSQLServer::Connect("oracle://pccmsecdb:1521/ecalh4db","read01","XXXpasswordXXX");

 gStyle->SetOptStat(0);
 gStyle->SetOptFit();
 gStyle->SetPalette(1,0);
 c1 = new TCanvas("c1","The Map",200,10,600,400);
 c1->SetGrid();
 // c1->Divide(2,3);

 
temp_chan  = new TH2F("temp_chan","SM map", 85, -0.5, 84.5,20,-0.5, 19.5 );
 temp_chan->SetMaximum(1.);
 temp_chan->SetMinimum(0.);


char * sql="SELECT count( channelview.logic_id) from channelview,MON_PEDESTALS_DAT , mon_run_iov, run_iov WHERE  mon_run_iov.iov_id=(select mon_run_iov.iov_id from mon_run_iov where subrun_start=(select max(SUBRUN_START) from MON_RUN_IOV, RUN_IOV, RUN_TAG, run_type_def where RUN_iov.tag_id=RUN_tag.tag_id and RUN_tag.LOCATION_ID=1 and run_IOV.iov_id=mon_run_iov.run_iov_id and run_type_def.config_ver=1 and run_type_def.def_id=run_tag.RUN_TYPE_ID and run_type_def.run_type='PEDESTAL')) and run_iov.iov_id=mon_run_iov.run_iov_id and mon_run_iov.iov_id=MON_PEDESTALS_DAT.iov_id and channelview.logic_id=MON_PEDESTALS_DAT.logic_id and channelview.name=channelview.maps_to order by id1, id2 "  ;

 TSQLResult *res=db->Query(sql);   
 

 int nlines=0;
 int j=0; 
 do {
   j++;
   TSQLRow *row1=res->Next();
   TOracleRow * row2=(TOracleRow *)row1;  
   
   for (int i=0; i<res->GetFieldCount();i++) {
     //  printf(" %*.*s ",row2->GetFieldLength(i),row2->GetFieldLength(i),row2->GetField(i)); 
     if(i==0) nlines=atoi( row2->GetField(i));
   }
   // cout <<  endl;
   delete row2;
   
 } while (j<1);
  cout <<  "Number of records in the DB: " << nlines << endl; 
  cout <<"first loop done "<< endl; 


/*print rows one by one */

char * sql="SELECT channelview.id2, CAST( MON_PEDESTALS_DAT.PED_RMS_G6 AS NUMBER), run_iov.run_num  from channelview,MON_PEDESTALS_DAT , mon_run_iov, run_iov WHERE  mon_run_iov.iov_id=(select mon_run_iov.iov_id from mon_run_iov where subrun_start=(select max(SUBRUN_START) from MON_RUN_IOV, RUN_IOV, RUN_TAG, run_type_def where RUN_iov.tag_id=RUN_tag.tag_id and RUN_tag.LOCATION_ID=1 and run_IOV.iov_id=mon_run_iov.run_iov_id and run_type_def.config_ver=1 and run_type_def.def_id=run_tag.RUN_TYPE_ID and run_type_def.run_type='PEDESTAL')) and run_iov.iov_id=mon_run_iov.run_iov_id and mon_run_iov.iov_id=MON_PEDESTALS_DAT.iov_id and channelview.logic_id=MON_PEDESTALS_DAT.logic_id and channelview.name=channelview.maps_to order by id1, id2 "  ;

 TSQLResult *res=db->Query(sql);   
 
 float temp=0;
 int chan=0;
 int j=0; 
 int run_num=0;
 do {
   j++;
   TSQLRow *row1=res->Next();
   TOracleRow * row2=(TOracleRow *)row1;  
   
   for (int i=0; i<res->GetFieldCount();i++) {
     //   printf(" %*.*s ",row2->GetFieldLength(i),row2->GetFieldLength(i),row2->GetField(i)); 
     if(i==0) chan=atoi( row2->GetField(i));
     if(i==1) temp=atof( row2->GetField(i));
     if(i==2) run_num=atoi( row2->GetField(i));
   }
   //   cout <<  endl;
   
   Float_t tpippo= (Float_t)temp;
   Float_t tchanx= (Float_t) ((chan-1)/20) ;
   Float_t tchany= (Float_t) ((chan-1)%20) ;
   temp_chan->Fill(tchanx, tchany, tpippo);
   
     
   delete row2;
   
 } while (j<nlines);
  cout <<  "run number: " << run_num << endl; 
  cout <<"loop done "<< endl; 

 
 // c1->cd(1);
 //hv_vmon->Draw();
 //c1->Update();

 //  c1->cd(1);
 temp_chan->Draw("colz");
 c1->Update(); 
 

 
 printf("end \n");
 delete res;
 delete db;
 
}
