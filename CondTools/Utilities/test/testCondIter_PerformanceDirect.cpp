//---- Test performance IOV ----
//test of speed access to database and saving time after the use of method rewind
//-----------------------------------------------
//Test performance with several dimensions (numbers of IOVs) with direct access to database. You can change the dimension of the database changing the variable "int nDiv" inside the code.
//-----------------------------------------------
//name of the database:
//oracle://cms_orcoff_int2r/CMS_COND_ECAL
//local catalog:
//devCondDBCatalog.xml
//tag:
//EcalPedestals
//-----------------------------------------------


#include "./PlotIOV.cpp"
#include <time.h>
#include <memory>
// #include <boost/shared_ptr.hpp>



void testCondIter_PerformanceDirect(){

std::cout <<std::endl<< "---- Test Performance Program ----"<<std::endl;

  //--------------------------------------
  int nDiv = 20;
  //--------------------------------------

  std::cout << std::endl << "IOV created" << std::endl;


  std::string NameDB;
  std::cout << "Name of DB = ";
  NameDB = "oracle://cms_orcoff_int2r/CMS_COND_ECAL";

  std::string FileXml;
  std::cout << "File .xml = ";
  FileXml = "devCondDBCatalog.xml";

  std::string TagData;
  std::cout << "TagData = ";
  TagData = "EcalPedestals_test";

  std::string User;
  std::cout << "User = ";
  User = "CMS_COND_ECAL";
  std::cout << std::endl;

  std::string Password;
  std::cout << "Passoword = ";
  std::cin >> Password;
  std::cout << std::endl;

  
  
  
  double start, end;
  
  int minimum = 1;
  int maximum;
  
  
  Int_t NDetID = 838926865;
  std::string Value = "mean_x12";
  std::string Data = "m_pedestals";
  std::string General = "EcalPedestals";



 
  Int_t* Creation = new Int_t[nDiv];
  Int_t* Rewind = new Int_t[nDiv];
  Int_t* RewindPlot = new Int_t[nDiv];
  Int_t* Clear = new Int_t[nDiv];
  Int_t* ClearPlot = new Int_t[nDiv];
  
  
  Int_t* X = new Int_t [nDiv];
   
  
  for (int n=0; n<nDiv; n++) {
      
      X[n] = n+1;
      std::cout << "--------------------------------------------------";
      std::cout << std::endl<<"Numbers of Runs = "<< X[n] <<std::endl;

      minimum = 0;
      maximum = X[n];
      
      start = clock();   

      CondCachedIter<EcalPedestals> *Iterator = new CondCachedIter<EcalPedestals>;
      
      Iterator->create(NameDB,FileXml,TagData,User,Password);
      end = clock();

      std::vector<double> Run;
      std::vector<double> RunStart;
      std::vector<double> RunStop;
      std::vector<double> ValueY;

      
      std::cout <<"Time Creation link with Database = " <<  ((double) (end - start)) << " (a.u.)" <<std::endl;
  
      start = clock();   
//       Iterator->setRange((unsigned int) minimum,(unsigned int) maximum);
      Iterator->setMax(maximum);
      Iterator->setMin(minimum);
      
      end = clock();
  
      std::cout <<"Time SetRange = " <<  ((double) (end - start)) << " (a.u.)" <<std::endl;

      
      start = clock();  
      
      int counter = 0;
      NoPlotIOVCache<EcalPedestals>(Iterator,NDetID,Data,Value,General);
      end = clock();
      
       
      Creation[n] = (Int_t) (end - start);
      std::cout << std::endl<<"Time Creation = " <<  ((double) (end - start)) << " (a.u.)" <<std::endl;

      std::cout << std::endl<<"Rewind" <<std::endl;
      start = clock();   
      Iterator->rewind();
      end = clock();    
      Rewind[n] = (Int_t) (end - start);
      std::cout << std::endl<<"Time Rewind = " <<  ((double) (end - start)) << " (a.u.)" <<std::endl;
      
      std::cout << std::endl<<"Rewind + Plot" <<std::endl;
      start = clock();   
      NoPlotIOVCache<EcalPedestals>(Iterator,NDetID,Data,Value,General);
      end = clock();      
      RewindPlot[n] = Rewind[n] + (Int_t) (end - start);
      std::cout << std::endl<<"Time Rewind + Plot = " <<  Rewind[n] + ((double) (end - start)) << " (a.u.)" <<std::endl;


      std::cout << std::endl<<"Clear" <<std::endl;
      start = clock();   
      Iterator->clear();
      end = clock();    
      Clear[n] = (Int_t) (end - start);
      std::cout << std::endl<<"Time Clear = " <<  ((double) (end - start)) << " (a.u.)" <<std::endl;
      
      std::cout << std::endl<<"Clear + Plot" <<std::endl;
      start = clock();      
      NoPlotIOVCache<EcalPedestals>(Iterator,NDetID,Data,Value,General);
      end = clock();      
      ClearPlot[n] = Clear[n] + (Int_t) (end - start);
      std::cout << std::endl<<"Time Clear + Plot = " <<  Clear[n] + ((double) (end - start)) << " (a.u.)" <<std::endl;
      
      delete Iterator;
  }
  
  //---- Graphics ----
  
  TCanvas* ccPerformance = new TCanvas ("Time","Time",10,10,500,400);


  ccPerformance->cd();
  TMultiGraph *mg = new TMultiGraph("Time performance","Time performance"); ;

  TLegend* lg = new TLegend(0.70,0.1,0.85,0.2);
  
  
  
  TGraph* GraphCreation = new TGraph(nDiv,X,Creation);
  GraphCreation->SetTitle("Creation Time vs number of Runs");
  GraphCreation->SetMarkerColor(4);
  GraphCreation->SetMarkerSize(.3);
  GraphCreation->SetMarkerStyle(20);
  GraphCreation->SetLineColor(3);
  GraphCreation->SetFillColor(3);
  mg->Add(GraphCreation);
  lg->AddEntry(GraphCreation,"Creation Time vs number of Runs","L");
  mg->Draw("APL");
  mg->GetXaxis()->SetTitle("# Runs");
  mg->GetYaxis()->SetTitleOffset(1.2);
  mg->GetYaxis()->SetTitle("Time");
  mg->GetXaxis()->SetTitleSize(0.04);
  mg->GetYaxis()->SetTitleSize(0.04);

  TF1 *linearCreation =  new TF1("linearCreation","pol1",0,X[nDiv-1]);
  linearCreation->SetLineWidth(2); 
  linearCreation->SetLineColor(3); 
  GraphCreation->Fit("linearCreation","RME");
  
  double Cre = (linearCreation->GetParameter(1));
  double errCre = (linearCreation->GetParError(1));
  std::cout << std::endl << "Creation = " << Cre << " +/-" << errCre << std::endl;
  
  
  TGraph* GraphRewind = new TGraph(nDiv,X,RewindPlot);
  GraphRewind->SetTitle("Rewind Time vs number of Runs");
  GraphRewind->SetMarkerColor(4);
  GraphRewind->SetMarkerSize(.3);
  GraphRewind->SetMarkerStyle(20);
  GraphRewind->SetLineColor(5);
  GraphRewind->SetFillColor(3);
  mg->Add(GraphRewind);
  lg->AddEntry(GraphRewind,"Rewind Time vs number of Runs","L");
  mg->Draw("APL");



  TF1 *linearRewind =  new TF1("linearRewind","pol1",0,X[nDiv-1]);
  linearRewind->SetLineWidth(2);
  linearRewind->SetLineColor(5);
  GraphRewind->Fit("linearRewind","RME");

  double Rew = (linearRewind->GetParameter(1));
  double errRew = (linearRewind->GetParError(1));
  std::cout << std::endl << "Rewind = " << Rew << " +/-" << errRew << std::endl;
  
  
  TGraph* GraphClear = new TGraph(nDiv,X,ClearPlot);
  GraphClear->SetTitle("Clear Time vs number of Runs");
  GraphClear->SetMarkerColor(4);
  GraphClear->SetMarkerSize(.3);
  GraphClear->SetMarkerStyle(20);
  GraphClear->SetLineColor(6);
  GraphClear->SetFillColor(3);
  mg->Add(GraphClear);
  lg->AddEntry(GraphClear,"Clear Time vs number of Runs","L");
 
  TF1 *linearClear =  new TF1("linearClear","pol1",0,X[nDiv-1]);
  linearClear->SetLineWidth(2); 
  linearClear->SetLineColor(6); 
  GraphClear->Fit("linearClear","RME");  

  double Cle = (linearClear->GetParameter(1));
  double errCle = (linearClear->GetParError(1));
  std::cout << std::endl << "Clear = " << Cle << " +/-" << errCle << std::endl;

  
  ccPerformance->Update();
  
  
  
  
  std::cout << std::endl << "------------------------------------------------------------" << std::endl;
  std::cout << std::endl << "------------------------------------------------------------" << std::endl;
  std::cout << std::endl << "------------------------------------------------------------" << std::endl;
    
  std::cout << std::endl << "Creation = " << Cre << " +/-" << errCre << " (a.u.) / #Runs" << std::endl;
 
  std::cout << std::endl << "Rewind = " << Rew << " +/-" << errRew << " (a.u.) / #Runs" << std::endl;
 
  std::cout << std::endl << "Clear = " << Cle << " +/-" << errCle << " (a.u.) / #Runs" << std::endl;

  std::cout << std::endl << "Ratio Creation/Rewind = " << Cre/Rew << " +/- " << sqrt((errCre/Cre)*(errCre/Cre) + (errRew/Rew)*(errRew/Rew)) * Cre/Rew << " (a.u.) / #Runs" << std::endl;
 
  
  std::cout << std::endl << "------------------------------------------------------------" << std::endl;


  
  
  
  
  
  std::cout << std::endl << "---- class Iter ----" << std::endl;
  
  
  Int_t* CreationIter = new Int_t[nDiv];
    
  for (int n=0; n<nDiv; n++) {
      
      std::cout << "--------------------------------------------------";
      std::cout << std::endl<<"Numbers of Runs = "<< n*1 <<std::endl;

      minimum = 0;
      maximum = X[n];
      
      start = clock();   
      CondIter <EcalPedestals> *IteratorIter = new CondIter<EcalPedestals>;
      IteratorIter->create(NameDB,FileXml,TagData,User,Password);
      end = clock();
      
      std::cout <<"Time Creation link with Database Iter = " <<  ((double) (end - start)) << " (a.u.)" <<std::endl;
        
      start = clock();   
      IteratorIter->setRange(minimum,maximum);
      end = clock();
        
      std::cout <<"Time SetRange Iter = " <<  ((double) (end - start)) << " (a.u.)" <<std::endl;
      
            
      start = clock();   
      NoPlotIOV<EcalPedestals>(IteratorIter,NDetID,Data,Value,General);
   
      end = clock();
            
      CreationIter[n] = (Int_t) (end - start);
      std::cout << std::endl<<"Time Creation Iter = " <<  ((double) (end - start)) << " (a.u.)" <<std::endl;
           
      delete IteratorIter;
      
  }
  
  
  
  TGraph* GraphCreationIter = new TGraph(nDiv,X,CreationIter);
  GraphCreationIter->SetTitle("Creation Time Iter vs number of Runs");
  GraphCreationIter->SetMarkerColor(4);
  GraphCreationIter->SetMarkerSize(.3);
  GraphCreationIter->SetMarkerStyle(20);
  GraphCreationIter->SetLineColor(7);
  GraphCreationIter->SetFillColor(3);
  mg->Add(GraphCreationIter);
  lg->AddEntry(GraphCreationIter,"Creation Time Iter vs number of Runs","L");
  ccPerformance->Update();

  TF1 *linearCreationIter =  new TF1("linearCreationIter","pol1",0,X[nDiv-1]);
  linearCreationIter->SetLineWidth(2);
  linearCreationIter->SetLineColor(7);
  GraphCreationIter->Fit("linearCreationIter","RME");   
  double CreIter = (linearCreationIter->GetParameter(1));
  double errCreIter = (linearCreationIter->GetParError(1));
  std::cout << std::endl << "CreationIter = " << CreIter << " +/-" << errCreIter << std::endl;
  
  lg->Draw();
  
  ccPerformance->Update();

  
  //---- Data analized
  
  minimum = 0;
  maximum = X[nDiv-1];
  
  CondIter<EcalPedestals> *IteratorPlot = new CondIter<EcalPedestals>;
  IteratorPlot->create(NameDB,FileXml,TagData,User,Password);
  IteratorPlot->setRange(minimum,maximum);
  
  std::cout << "--------------------------" << std::endl << std::endl;
  std::cout << "--------------------------" << std::endl << std::endl;
  std::cout << "--------------------------" << std::endl << std::endl;
  std::cout << "--------------------------" << std::endl << std::endl;
  std::cout << "--------------------------" << std::endl << std::endl;
  std::cout << "--------------------------" << std::endl << std::endl;
  
  PlotIOV<EcalPedestals>(IteratorPlot,NDetID,Data,Value,General);

  delete IteratorPlot;
  
  
  std::cout << std::endl << "------------------------------------------------------------" << std::endl;
  std::cout << std::endl << "------------------------------------------------------------" << std::endl;
  std::cout << std::endl << "------------------------------------------------------------" << std::endl;
  
  std::cout << std::endl << "Creation Iter = " << CreIter << " +/-" << errCreIter << " (a.u.) / #Runs" << std::endl;
   
  std::cout << std::endl << "Creation = " << Cre << " +/-" << errCre << " (a.u.) / #Runs" << std::endl;
 
  std::cout << std::endl << "Rewind = " << Rew << " +/-" << errRew << " (a.u.) / #Runs" << std::endl;
 
  std::cout << std::endl << "Clear = " << Cle << " +/-" << errCle << " (a.u.) / #Runs" << std::endl;

  std::cout << std::endl << "Ratio Creation/Rewind = " << Cre/Rew << " +/-" << sqrt((errCre/Cre)*(errCre/Cre) + (errRew/Rew)*(errRew/Rew)) * Cre/Rew << " (a.u.) / #Runs" << std::endl;
 
  
  std::cout << std::endl << "------------------------------------------------------------" << std::endl;

    
  
  ccPerformance->Print("TrialDirect.eps");
  
  
  
  
  
}

