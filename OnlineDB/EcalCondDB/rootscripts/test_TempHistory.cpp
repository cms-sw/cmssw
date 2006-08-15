void test_TempHistory(){

/*connect with oracle server */
TSQLServer *db=TSQLServer::Connect("oracle://pccmsecdb:1521/ecalh4db","read01","XXXpasswordXXX");

 gStyle->SetOptStat(0);
 gStyle->SetOptFit();
 gStyle->SetPalette(1,0);
 c1 = new TCanvas("c1","The Temperatures",200,10,600,400);
 c1->SetGrid();
 // c1->Divide(2,3);

temp_vs_time  = new TH2F("temp_vs_time","TEMP CHAN VS TIME", 100, 450, 550, 100, 15.,25. );
 temp_vs_time->GetXaxis()->SetTitle("Time(days since 1/1/2005)");
 temp_vs_time->GetYaxis()->SetTitle("Temperature (C)");

 /* first see how many rows we have */
char * sql="SELECT count(dcu_iov.iov_id)  from channelview, DCU_CAPSULE_TEMP_DAT, dcu_iov WHERE channelview.id1=4 and channelview.id2=10 and dcu_iov.iov_id=DCU_CAPSULE_TEMP_DAT.iov_id and channelview.logic_id=DCU_CAPSULE_TEMP_DAT.logic_id order by id1, id2, since "  ;

 TSQLResult *res=db->Query(sql);   
 int j=0;
 int nRows=0;

 do {
   j++;
   TSQLRow *row1=res->Next();
   TOracleRow * row2=(TOracleRow *)row1;  
   nRows=atoi( row2->GetField(0));
   delete row2;
   
 } while (j<1);
 cout <<"loop done "<< endl; 



char * sql="SELECT channelview.id1, channelview.id2, CAST(DCU_CAPSULE_TEMP_DAT.capsule_temp AS NUMBER), (since-to_date('01-JAN-2005 00:00:00','DD-MON-YYYY HH24:MI:SS')) , channelview.logic_id  from channelview, DCU_CAPSULE_TEMP_DAT, dcu_iov WHERE channelview.id1=4 and channelview.id2=10 and dcu_iov.iov_id=DCU_CAPSULE_TEMP_DAT.iov_id and channelview.logic_id=DCU_CAPSULE_TEMP_DAT.logic_id order by id1, id2, since "  ;


 TSQLResult *res=db->Query(sql);   
 
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
     if(i==1) chan=atoi( row2->GetField(i));
     if(i==2) temp=atof( row2->GetField(i));
     if(i==3) time_meas=atof( row2->GetField(i));
   }
   cout <<  endl;
   
   Float_t tpippo= (Float_t)temp;
   Float_t tchanx= (Float_t) ((chan-1)/10) ;
   Float_t tchany= (Float_t) ((chan-1)%10) ;
   temp_vs_time->Fill(time_meas, tpippo,1.);
   
   
   cout << chan << " " << tchanx << " " << tchany << " " << tpippo << endl; 
   
   
   
   delete row2;
   
 } while (j<nRows);
 cout <<"loop done "<< endl; 

 
 // c1->cd(1);
 //hv_vmon->Draw();
 //c1->Update();

 //  c1->cd(1);
 temp_vs_time->Draw();
 c1->Update(); 
 

 
 printf("end \n");

 delete res;
 delete db;
 
}
