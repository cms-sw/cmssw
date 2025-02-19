#include "phiedges_export.h"
#include <vector>
#include "TF1.h"
#include "TMath.h"

class  SawTeethFunctionObject {
public:
  
  SawTeethFunctionObject(){};
  
  SawTeethFunctionObject(int idx): index(idx)
  {
    // copy corresponding wheel's phiedges
    for (int i=0;i<37;i++) ed[i] = phiedges[index][i];
    // determine n sectors
    n=0;
    while(ed[n]<999) {n++;}
    // set last sector's sorted phi properly:
    ed[n] = TMath::Pi() + fabs(ed[0] - TMath::Pi());
  }
   
  double operator() (double *xx, double *p) {
    double x = xx[0];
    if (x < ed[0]) x += TMath::TwoPi();
    for (int i=0; i<n; i++){
      if (x <= ed[i]) continue;
      if (x >  ed[i+1]) continue;
      // linear fit in the middle of sector
      return p[i*2] + p[i*2+1]*(x - 0.5*(ed[i]+ed[i+1]));
    }
    return 0;    
  }
  
  int getN() {return n;}
  
private:
  int index;
  double ed[37];
  int n;
};

SawTeethFunctionObject *fobj0 = new SawTeethFunctionObject(0);
SawTeethFunctionObject *fobj1 = new SawTeethFunctionObject(1);
SawTeethFunctionObject *fobj2 = new SawTeethFunctionObject(2);
SawTeethFunctionObject *fobj3 = new SawTeethFunctionObject(3);
SawTeethFunctionObject *fobj4 = new SawTeethFunctionObject(4);
SawTeethFunctionObject *fobj5 = new SawTeethFunctionObject(5);
SawTeethFunctionObject *fobj6 = new SawTeethFunctionObject(6);
SawTeethFunctionObject *fobj7 = new SawTeethFunctionObject(7);
SawTeethFunctionObject *fobj8 = new SawTeethFunctionObject(8);
SawTeethFunctionObject *fobj9 = new SawTeethFunctionObject(9);
SawTeethFunctionObject *fobj10 = new SawTeethFunctionObject(10);
SawTeethFunctionObject *fobj11 = new SawTeethFunctionObject(11);
SawTeethFunctionObject *fobj12 = new SawTeethFunctionObject(12);
SawTeethFunctionObject *fobj13 = new SawTeethFunctionObject(13);

TF1 *fitf0 = new TF1("fitf0", fobj0, -TMath::Pi(), TMath::Pi(), 2*fobj0->getN(), "SawTeethFunctionObject");
TF1 *fitf1 = new TF1("fitf1", fobj1, -TMath::Pi(), TMath::Pi(), 2*fobj1->getN(), "SawTeethFunctionObject");
TF1 *fitf2 = new TF1("fitf2", fobj2, -TMath::Pi(), TMath::Pi(), 2*fobj2->getN(), "SawTeethFunctionObject");
TF1 *fitf3 = new TF1("fitf3", fobj3, -TMath::Pi(), TMath::Pi(), 2*fobj3->getN(), "SawTeethFunctionObject");
TF1 *fitf4 = new TF1("fitf4", fobj4, -TMath::Pi(), TMath::Pi(), 2*fobj4->getN(), "SawTeethFunctionObject");
TF1 *fitf5 = new TF1("fitf5", fobj5, -TMath::Pi(), TMath::Pi(), 2*fobj5->getN(), "SawTeethFunctionObject");
TF1 *fitf6 = new TF1("fitf6", fobj6, -TMath::Pi(), TMath::Pi(), 2*fobj6->getN(), "SawTeethFunctionObject");
TF1 *fitf7 = new TF1("fitf7", fobj7, -TMath::Pi(), TMath::Pi(), 2*fobj7->getN(), "SawTeethFunctionObject");
TF1 *fitf8 = new TF1("fitf8", fobj8, -TMath::Pi(), TMath::Pi(), 2*fobj8->getN(), "SawTeethFunctionObject");
TF1 *fitf9 = new TF1("fitf9", fobj9, -TMath::Pi(), TMath::Pi(), 2*fobj9->getN(), "SawTeethFunctionObject");
TF1 *fitf10 = new TF1("fitf10", fobj10, -TMath::Pi(), TMath::Pi(), 2*fobj10->getN(), "SawTeethFunctionObject");
TF1 *fitf11 = new TF1("fitf11", fobj11, -TMath::Pi(), TMath::Pi(), 2*fobj11->getN(), "SawTeethFunctionObject");
TF1 *fitf12 = new TF1("fitf12", fobj12, -TMath::Pi(), TMath::Pi(), 2*fobj12->getN(), "SawTeethFunctionObject");
TF1 *fitf13 = new TF1("fitf13", fobj13, -TMath::Pi(), TMath::Pi(), 2*fobj13->getN(), "SawTeethFunctionObject");

void cleanUpHeap(){
delete fobj0 ;
delete fobj1 ;
delete fobj2 ;
delete fobj3 ;
delete fobj4 ;
delete fobj5 ;
delete fobj6 ;
delete fobj7 ;
delete fobj8 ;
delete fobj9 ;
delete fobj10;
delete fobj11;
delete fobj12;
delete fobj13;
delete fitf0 ;
delete fitf1 ;
delete fitf2 ;
delete fitf3 ;
delete fitf4 ;
delete fitf5 ;
delete fitf6 ;
delete fitf7 ;
delete fitf8 ;
delete fitf9 ;
delete fitf10;
delete fitf11;
delete fitf12;
delete fitf13;
}

/*
std::vector<SawTeethFunctionObject *> fobj;
std::vector<TF1 *> fitf;
TF1 *fitf0, *fitf1, *fitf2, *fitf3, *fitf4, *fitf5, *fitf6, *fitf7, *fitf8, *fitf9, *fitf10, *fitf11, *fitf12, *fitf13;
 
void fitFunctionsInit()
{
  for (int i=0; i<14; i++) {
    SawTeethFunctionObject *o = new SawTeethFunctionObject(i);
    fobj.push_back(o);
    char nm[20];
    sprintf(nm,"fitf%d",i);
    TF1 *f = new TF1(nm, fobj[i], -TMath::Pi(), TMath::Pi(), fobj[i]->getN(),"SawTeethFunctionObject");
    fitf.push_back(f);
  }
  fitf0 = fitf[0];
  fitf1 = fitf[1];
  fitf2 = fitf[2];
  fitf3 = fitf[3];
  fitf4 = fitf[4];
  fitf5 = fitf[5];
  fitf6 = fitf[6];
  fitf7 = fitf[7];
  fitf8 = fitf[8];
  fitf9 = fitf[9];
  fitf10 =fitf[10];
  fitf11 =fitf[11];
  fitf12 =fitf[12];
  fitf13 =fitf[13];
}

*/
