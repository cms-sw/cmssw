void test_TempMap(){

/*connect with oracle server */
TSQLServer *db=TSQLServer::Connect("oracle://pccmsecdb:1521/ecalh4db","read01","XXXpasswordXXX");

 gStyle->SetOptStat(0);
 gStyle->SetOptFit();
 gStyle->SetPalette(1,0);
 c1 = new TCanvas("c1","The Temperatures",200,10,600,400);
 c1->SetGrid();
 // c1->Divide(2,3);


temp_chan  = new TH2F("temp_chan","temp chan", 17, -0.5, 16.5,10,-0.5, 9.5 );
 temp_chan->SetMaximum(25.);
 temp_chan->SetMinimum(15.);


/*print rows one by one */

char * sql="SELECT channelview.id1, channelview.id2, CAST(DCU_CAPSULE_TEMP_DAT.capsule_temp AS NUMBER), (since-to_date('01-JAN-2005 14:55:33','DD-MON-YYYY HH24:MI:SS'))*24 , channelview.logic_id  from channelview, DCU_CAPSULE_TEMP_DAT, dcu_iov WHERE DCU_CAPSULE_TEMP_DAT.iov_id = (select max(iov_id) from DCU_IOV, DCU_TAG where dcu_iov.tag_id=dcu_tag.tag_id and dcu_tag.LOCATION_ID=1) and dcu_iov.iov_id=DCU_CAPSULE_TEMP_DAT.iov_id and channelview.logic_id=DCU_CAPSULE_TEMP_DAT.logic_id order by id1, id2 "  ;

 TSQLResult *res=db->Query(sql);   
 
 float temp=0;
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
   }
   cout <<  endl;
   
   Float_t tpippo= (Float_t)temp;
   Float_t tchanx= (Float_t) ((chan-1)/10) ;
   Float_t tchany= (Float_t) ((chan-1)%10) ;
   temp_chan->Fill(tchanx, tchany, tpippo);
   
   cout << chan << " " << tchanx << " " << tchany << " " << tpippo << endl; 
   
   
   
   delete row2;
   
 } while (j<170);
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
