//---- Test Program for class CondCachedIter() ---- 
//---- direct access to database ----
//-----------------------------------------------
//plot of several graphs as examples

//-----------------------------------------------
//name of the database:
//oracle://cms_orcoff_int2r/CMS_COND_ECAL
//local catalog:
//devCondDBCatalog.xml
//tag:
//EcalPedestals_test
//-----------------------------------------------





#include <iostream>
#include <string>



int testCondIter_Plots(){

std::string NameDB;
std::cout << "Name of DB = ";
// std::cin >> NameDB;
NameDB = "oracle://cms_orcoff_int2r/CMS_COND_ECAL";

std::string FileXml;
std::cout << "File .xml = ";
// std::cin >> FileXml;
FileXml = "devCondDBCatalog.xml";

std::string TagData;
std::cout << "TagData = ";
// std::cin >> TagData;
TagData = "EcalPedestals_test";
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
Iterator.create(NameDB,FileXml,TagData,User,Password);



std::cout << "Iterator has been created ..."<<std::endl;


std::cout << "------------ Test ------------" << std::endl;

//---- For Root visualization ----
double Run[10000];
double RunStart[10000];
double RunStop[10000];
double X[10000];
double Y[10000];

double X2[10000];
double X3[10000];
double X4[10000];
double X5[10000];
double X6[10000];
double X7[10000];
double X8[10000];
double X9[10000];



for (int k=0; k<10000; k++){
    X[k] = 0.;
    X2[k] = 0.;
    X3[k] = 0.;
    X4[k] = 0.;
    X5[k] = 0.;
    X6[k] = 0.;
    X7[k] = 0.;
    X8[k] = 0.;
    X9[k] = 0.;
    Y[k] = 0.;
    Run[k] = 0;
    RunStart[k] = 0;
    RunStop[k] = 0;
}


//-------------------------------------------------------

char hname[100];
sprintf(hname,"Scatter Plot EcalPedestals");
int min = 200;
int max = 215;
TH2F *SP = new TH2F(hname,hname,(max-min)*2,min,max,(max-min)*2,min,max);



//-------------------------------------------------------
TGraph2D *Gr2D = new TGraph2D();

TGraph2D *GrDet = new TGraph2D();
//-------------------------------------------------------



int minimum = 3;
int maximum = 20;
Iterator.setRange(minimum,maximum);

// Iterator.setMax(maximum);
// Iterator.setMin(minimum);

const EcalPedestals* reference;
reference = 0;
int counter = 0;
int cont = 0;

while(reference = Iterator.next()) { 
	
    unsigned int SizeOfVector2 = (reference->m_pedestals).size();
    std::cout << "Executed " << SizeOfVector2 << " times" << std::endl;
	
    if(SizeOfVector2 > 0) {
        Run[counter] = Iterator.getTime();
        RunStart[counter] = Iterator.getStartTime();
        RunStop[counter] = Iterator.getStopTime();
		
        std::cout << "\n\t\tTIME = " << Run[counter] << std::endl;

        std::cout << (((reference->m_pedestals)[838926865]).mean_x12);
        X[counter] = (((reference->m_pedestals)[838959111]).mean_x12);	 
        X2[counter] = (((reference->m_pedestals)[838959112]).mean_x12);	 
        X3[counter] = (((reference->m_pedestals)[838959113]).mean_x12);	 
        X4[counter] = (((reference->m_pedestals)[838959114]).mean_x12);	 
        X5[counter] = (((reference->m_pedestals)[838959115]).mean_x12);	 
        X6[counter] = (((reference->m_pedestals)[838959115]).mean_x12);
        X7[counter] = (((reference->m_pedestals)[838959115]).mean_x12);
        X8[counter] = (((reference->m_pedestals)[838959115]).mean_x12);
        X9[counter] = (((reference->m_pedestals)[838959115]).mean_x12);
	 
        Y[counter] = (((reference->m_pedestals)[838959111]).mean_x1);
	 
        SP->Fill(X[counter],Y[counter],Run[counter]);
        Gr2D->SetPoint(counter,X[counter],Y[counter],Run[counter]);
        GrDet->SetPoint(cont,838959111,Run[counter],X[counter]);
        cont++;
        GrDet->SetPoint(cont,838959112,Run[counter],X2[counter]);
        cont++;
        GrDet->SetPoint(cont,838959113,Run[counter],X3[counter]);
        cont++;
        GrDet->SetPoint(cont,838959114,Run[counter],X4[counter]);
        cont++;
        GrDet->SetPoint(cont,838959115,Run[counter],X5[counter]);
        cont++;
        GrDet->SetPoint(cont,838959116,Run[counter],X6[counter]);
        cont++;
        GrDet->SetPoint(cont,838959117,Run[counter],X7[counter]);
        cont++;
        GrDet->SetPoint(cont,838959118,Run[counter],X8[counter]);
        cont++;
        GrDet->SetPoint(cont,838959119,Run[counter],X9[counter]);
        cont++;

        counter++;	
        std::cout << "\n-------------------------------------------------------------\n";

    }	
}


for(int j=0; j<counter; j++) std::cout << "\n\tRunStart = " << RunStart[j] << "\tValue X = "<<X[j]<<std::endl;



//--------------------------------------------------------------

double tot = Run[counter-1];

double *TOT = new double [tot];
double *TOTX = new double [tot];
double *TOTY = new double [tot];

double ALLX = 0;
double ALLY = 0;

unsigned long int Max = RunStop[counter-1];
std::cout << "             MAX = " <<Max;
int conta = 0;
for(unsigned long int ko=0; ko<Max; ko++) {
    TOT[ko] = ko+1;
    if ((ko+1)>=RunStart[conta]){
        ALLX = X[conta];
        ALLY = Y[conta];
        if ((ko+1)>=RunStop[conta]) conta++;
    }
    TOTX[ko] = ALLX;
    TOTY[ko] = ALLY;
}


TCanvas *cc2 = new TCanvas ("Histo","Histo",10,10,700,400);;
TGraph *GraphBAR = new TGraph(Max,TOT,TOTX);
GraphBAR->SetMarkerColor(1);
GraphBAR->SetMarkerSize(.9);
GraphBAR->SetMarkerStyle(20);
GraphBAR->SetLineColor(1);
GraphBAR->SetFillColor(40);
GraphBAR->Draw("AB");
GraphBAR->SetTitle("Graph EcalPedestals");
GraphBAR->GetXaxis()->SetTitle("# Run");
GraphBAR->GetYaxis()->SetTitleOffset(1.2);
GraphBAR->GetYaxis()->SetTitle("X");
GraphBAR->GetXaxis()->SetTitleSize(0.04);
GraphBAR->GetYaxis()->SetTitleSize(0.04);
cc2->Update();


//--------------------------------------------------------------

TCanvas *cc;
TGraph *Graph;
char name[100];
sprintf(name,"X vs Run");
cc = new TCanvas (name,name,10,10,700,400);

Graph = new TGraph(counter,Run,X);
Graph->SetTitle("Graph EcalPedestals Bar");
Graph->SetMarkerColor(3);
Graph->SetMarkerSize(.9);
Graph->SetMarkerStyle(20);
Graph->SetLineColor(3);
Graph->SetFillColor(3);
Graph->Draw("APL");
Graph->GetXaxis()->SetTitle("# Run");
Graph->GetYaxis()->SetTitleOffset(1.2);
Graph->GetYaxis()->SetTitle("X_12");
Graph->GetXaxis()->SetTitleSize(0.04);
Graph->GetYaxis()->SetTitleSize(0.04);
cc->Update();





//--------------------------------------------------------------

TCanvas *cc2D;
TGraph *Graph2D;
char name2D[100];
sprintf(name2D,"X_1 vs X_12");
cc2D = new TCanvas (name2D,name2D,10,10,700,400);

Graph2D = new TGraph(counter,X,Y);
Graph2D->SetTitle("Graph EcalPedestals X_12 vs X_1");
Graph2D->SetMarkerColor(4);
Graph2D->SetMarkerSize(.3);
Graph2D->SetMarkerStyle(20);
Graph2D->SetLineColor(3);
Graph2D->SetFillColor(3);
Graph2D->Draw("AP");
Graph2D->GetXaxis()->SetTitle("X_1");
Graph2D->GetYaxis()->SetTitleOffset(1.2);
Graph2D->GetYaxis()->SetTitle("X_12");
Graph2D->GetXaxis()->SetTitleSize(0.04);
Graph2D->GetYaxis()->SetTitleSize(0.04);
cc2D->Update();




//--------------------------------------------------------------


gStyle->SetPalette(1);

TCanvas *ccSP4 = new TCanvas ("Different Detectors","Different Detectors",10,10,700,400);
GrDet->Draw("psurf");
GrDet->SetMarkerColor(4);
GrDet->SetMarkerSize(.5);
GrDet->SetMarkerStyle(20);
GrDet->SetLineColor(3);
GrDet->SetMarkerStyle(20);
GrDet->GetXaxis()->SetTitle("Detector ID");
// GrDet->GetXaxis()->SetRange(100,300);
GrDet->GetYaxis()->SetTitle("# Run");
GrDet->GetZaxis()->SetTitle("X_12");
GrDet->GetYaxis()->SetLabelSize(0.02);
GrDet->GetXaxis()->SetLabelSize(0.02);
GrDet->GetZaxis()->SetLabelSize(0.02);
GrDet->GetYaxis()->SetTitleOffset(1.2);
GrDet->GetYaxis()->SetTitleOffset(1.2);
GrDet->GetYaxis()->SetTitleOffset(1.2);
ccSP4->Update();








//--------------------------------------------------------------

double *Time = new double[2*counter];
double *Value = new double[2*counter];


for(unsigned int i=0; i<counter; i++) {
    Time[2*i] = (double) RunStart[i];
    Time[2*i+1] = (double) RunStop[i] + 1;
    Value[2*i] = X[i];
    Value[2*i+1] = X[i];
}



std::cout << std::endl;
for(unsigned int i=0; i<(2*counter); i++) {
    std::cout<<i<<") Time = "<< Time[i]<< "\t X = "<<Value[i]<<std::endl;
}

TCanvas *ccBars;
TGraph *GraphBars;
ccBars = new TCanvas ("Bar","Bar",10,10,700,400);

GraphBars = new TGraph(2*counter,Time,Value);
GraphBars->SetTitle("Graph EcalPedestals X vs Run Bar");
GraphBars->SetMarkerColor(4);
GraphBars->SetMarkerSize(.3);
GraphBars->SetMarkerStyle(20);
GraphBars->SetLineColor(3);
GraphBars->SetFillColor(3);
GraphBars->Draw("APL");
GraphBars->GetXaxis()->SetTitle("# Run");
GraphBars->GetYaxis()->SetTitleOffset(1.2);
GraphBars->GetYaxis()->SetTitle("X");
GraphBars->GetXaxis()->SetTitleSize(0.04);
GraphBars->GetYaxis()->SetTitleSize(0.04);
ccBars->Update();







return 0;
}



