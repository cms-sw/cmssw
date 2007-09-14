//---- Test Program for class CondCachedIter() ---- 
//---- direct access to database ----
//name of the database:
//oracle://cms_orcoff_int2r/CMS_COND_ECAL
//local catalog:
//devCondDBCatalog.xml
//tag:
//EcalPedestals_test
//-----------------------------------------------
//plot of several graphs as example
//-----------------------------------------------




#include <iostream>
#include <string>


int testCondIter_DirectDB(){

std::string NameDB;
std::cout << "Name of DB = ";
// std::cin >> NameDB;
NameDB = "oracle://cms_orcoff_int2r/CMS_COND_ECAL";

std::string FileXml;
std::cout << "File .xml = ";
// std::cin >> FileXml;
FileXml = "devCondDBCatalog.xml";

std::string FileData;
std::cout << "FileData = ";
// std::cin >> FileData;
FileData = "EcalPedestals_test";
std::cout << std::endl;

std::string User;
std::cout << "User = ";
User = "CMS_COND_ECAL";
std::cout << std::endl;

std::string Password;
std::cout << "Passoword = ";
std::cin >> Password;
// Password = "password";
std::cout << std::endl;

CondCachedIter<EcalPedestals> Iterator;
Iterator.create(NameDB,FileXml,FileData,User,Password);

Iterator.setMax(170);
Iterator.setMin(10);

std::cout << "Iterator has been created ..."<<std::endl;


std::cout << "------------ Test ------------" << std::endl;

//---- For Root visualization ----

std::vector<double> Run;
std::vector<double> RunStart;
std::vector<double> RunStop;
std::vector<double> ValueY;
  
const EcalPedestals* reference;

   int counter = 0;

   while(reference = Iterator.next()) { 

        
       unsigned int SizeOfVector = 0;

       SizeOfVector = (reference->m_pedestals).size();
               
       std::cout << "Executed " << SizeOfVector << " times" << std::endl;

       if(SizeOfVector > 0) {
           Run.push_back(Iterator.getTime());
           RunStart.push_back(Iterator.getStartTime());
           RunStop.push_back(Iterator.getStopTime());

           ValueY.push_back(((reference->m_pedestals)[838926865]).mean_x12);
           
           counter++;
       }

   }

   double *Time = new double[2*counter];
   double *Value = new double[2*counter];

   for(unsigned int i=0; i<counter; i++) {
       Time[2*i] = (double) RunStart.at(i);
       Time[2*i+1] = (double) RunStop.at(i) +1;// "+1" since each run lasts 1 unit
       Value[2*i] = ValueY.at(i);
       Value[2*i+1] = ValueY.at(i);
   }


   TCanvas *ccBar;
   TGraph *GraphBar;
   ccBar = new TCanvas ("Bar","Bar",10,10,700,400);

   GraphBar = new TGraph(2*counter,Time,Value);
   GraphBar->SetTitle("Graph X vs Run Bar");
   GraphBar->SetMarkerColor(4);
   GraphBar->SetMarkerSize(.3);
   GraphBar->SetMarkerStyle(20);
   GraphBar->SetLineColor(3);
   GraphBar->SetFillColor(3);
   GraphBar->Draw("APL");
   GraphBar->GetXaxis()->SetTitle("# Run");
   GraphBar->GetYaxis()->SetTitleOffset(1.2);
   GraphBar->GetYaxis()->SetTitle("X");
   GraphBar->GetXaxis()->SetTitleSize(0.04);
   GraphBar->GetYaxis()->SetTitleSize(0.04);
   ccBar->Update();

   Run.clear();
   RunStart.clear();
   RunStop.clear();
   ValueY.clear();
    
   delete Time;
   delete Value;





   return 0;
}



