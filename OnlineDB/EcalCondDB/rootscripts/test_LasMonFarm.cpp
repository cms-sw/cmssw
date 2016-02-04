void test_LasMonFarm(){

/*connect with oracle server */
TSQLServer *db=TSQLServer::Connect("oracle://pccmsecdb:1521/ecalh4db","test05","oratest05");

 gStyle->SetOptStat(0);
   gStyle->SetOptFit();
 gStyle->SetPalette(1,0);
c1 = new TCanvas("c1","The Temperatures",200,10,600,400);
c1->SetGrid();
// c1->Divide(2,3);


laser_chan  = new TH2F("laser_chan","laser_chan", 85, -0.5, 84.5,20,-0.5, 19.5 );

/*print rows one by one */

char * sql="SELECT channelview.id1, channelview.id2, CAST(MF_LASER_BLUE_NORM_DAT.APD_OVER_PNB_MEAN AS NUMBER), channelview.name, channelview.maps_to from channelview, MF_LASER_BLUE_NORM_DAT WHERE  MF_LASER_BLUE_NORM_DAT.iov_id = (select max(iov_id) from MF_LASER_BLUE_NORM_DAT) and channelview.logic_id=MF_LASER_BLUE_NORM_DAT.logic_id order by name, id1, id2 "  ;

 TSQLResult *res=db->Query(sql);   
 
 float temp=0;
 int chan=0;
 int j=0; 
 do {
   j++;
   TSQLRow *row1=res->Next();
   TOracleRow * row2=(TOracleRow *)row1;  
   
   for (int i=0; i<res->GetFieldCount();i++) {
     //     printf(" %*.*s ",row2->GetFieldLength(i),row2->GetFieldLength(i),row2->GetField(i)); 
     if(i==1) chan=atoi( row2->GetField(i));
     if(i==2) temp=atof( row2->GetField(i));
   }
   //  cout <<  endl;
   
   Float_t tpippo= (Float_t)temp;
   Float_t tchanx= (Float_t) ((chan-1)/20) ;
   Float_t tchany= (Float_t) ((chan-1)%20) ;
   laser_chan->Fill(tchanx, tchany, tpippo);
   
   //   cout << chan << " " << tchanx << " " << tchany << " " << tpippo << " " << j<< endl; 
   if(temp !=0) cout  << chan << " "<< temp<< endl;
   
   
   delete row2;
   
 } while (j<1700);
 cout <<"loop done "<< endl; 

 
 // c1->cd(1);
 //hv_vmon->Draw();
 //c1->Update();

 //  c1->cd(1);
 laser_chan->Draw("colz");
 c1->Update(); 
 

 
 printf("end \n");

 delete res;
 delete db;
 
}
