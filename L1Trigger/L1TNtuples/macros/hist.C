#include <TH1.h>
#include <TH2.h>
#include <TProfile2D.h>
#include <TROOT.h>
#include <TGraph.h>
#include <TClass.h>
#include <TFile.h>
#include <TKey.h>

typedef struct {
 TH1F* all;
 TH1F* good;
 TH1F* eff;
} TH1eff;

typedef struct {
 TH2F* all;
 TH2F* good;
 TH2F* eff;
} TH2eff;

TH1F* h1d(char* name, int nbx, double xmin, double xmax) {
 if(TObject* o = gDirectory->Get(name)) delete o;
 TH1F* h = new TH1F(name,"",nbx,xmin,xmax);
 h->Sumw2();
 h->Reset();
 return h;
}

TH1F* h1dv(char* name, int nbx, double* bx) {
 if(TObject* o = gDirectory->Get(name)) delete o;
 TH1F* h = new TH1F(name,"",nbx,bx);
 h->Sumw2();
 h->Reset();
 return h;
}

TH2F* h2d(char* name, int nbx, double xmin, double xmax,
                     int nby, double ymin, double ymax) {
 if(TObject* o = gDirectory->Get(name)) delete o;
 TH2F* h = new TH2F(name,"",nbx,xmin,xmax,nby,ymin,ymax);
 h->Reset();
 return h;
}

TH2F* h2dvy(char* name, int nbx, double xmin, double xmax, int nby, double* by) {
 if(TObject* o = gDirectory->Get(name)) delete o;
 TH2F* h = new TH2F(name,"",nbx,xmin,xmax,nby,by);
 h->Sumw2();
 h->Reset();
 return h;
}

TProfile2D* h2dprof(char* name, int nbx, double xmin, double xmax,
                               int nby, double ymin, double ymax,
                                        double zmin, double zmax) {
 if(TObject* o = gDirectory->Get(name)) delete o;
 TProfile2D* h = new TProfile2D(name,"",nbx,xmin,xmax,nby,ymin,ymax,zmin,zmax,"");
 h->Reset();
 return h;
}

TH1eff& h1effv(char* name, int nbx, double* bx) {
 char tit[20];
 TH1eff* result = new TH1eff();
 sprintf(tit,"%s_all",name);
 result->all = h1dv(tit,nbx,bx);
 sprintf(tit,"%s_good",name);
 result->good = h1dv(tit,nbx,bx);
 sprintf(tit,"%s_eff",name);
 result->eff = h1dv(tit,nbx,bx);
 return *result;
}

TH2eff& h2eff(char* name, int nbx, double xmin, double xmax,
                         int nby, double ymin, double ymax) {
 char tit[20];
 TH2eff* result = new TH2eff();
 sprintf(tit,"%s_all",name);
 result->all = h2d(tit,nbx,xmin,xmax,nby,ymin,ymax);
 sprintf(tit,"%s_good",name);
 result->good = h2d(tit,nbx,xmin,xmax,nby,ymin,ymax);
 sprintf(tit,"%s_eff",name);
 result->eff = h2d(tit,nbx,xmin,xmax,nby,ymin,ymax);
 return *result;
}

TH2eff& h2effvy(char* name, int nbx, double xmin, double xmax,
                           int nby, double* by) {
 char tit[20];
 TH2eff* result = new TH2eff();
 sprintf(tit,"%s_all",name);
 result->all = h2dvy(tit,nbx,xmin,xmax,nby,by);
 sprintf(tit,"%s_good",name);
 result->good = h2dvy(tit,nbx,xmin,xmax,nby,by);
 sprintf(tit,"%s_eff",name);
 result->eff = h2dvy(tit,nbx,xmin,xmax,nby,by);
 return *result;
}

void h2effarray(char* name, int nbx, double xmin, double xmax,
                           int nby, double ymin, double ymax,
                           int ni, TH2eff* he) {
 char tit[20];
 for(int i=0; i<ni; i++) {
   sprintf(tit,"%s_%2.2d",name,i);
   he[i] = h2eff(tit,nbx,xmin,xmax,nby,ymin,ymax);
 }
}

TGraph* mygraph(char* name) {
 if(TObject* o = gDirectory->Get(name)) delete o;
 TGraph* h = new TGraph(0);
 return h;
}

void hreset() {
 TIter next(gDirectory->GetList());
 TObject* obj;
 while((obj=(TObject*)next())) {
   if(strstr(obj->IsA()->GetName(),"TH1")) {
//      cout << "resetting " << obj->GetName() << endl;
     ((TH1*)obj)->Reset();
   }
   if(strstr(obj->IsA()->GetName(),"TH2")) {
//      cout << "resetting " << obj->GetName() << endl;
     ((TH2*)obj)->Reset();
   }
 }
}

void hwrite(char* fn) {
 TIter next(gDirectory->GetList());
 TObject* obj;
 TFile* f = new TFile(fn,"recreate");
 f->cd();
 while((obj=(TObject*)next())) {
   if(strstr(obj->IsA()->GetName(),"TH1")) {
     ((TH1*)obj)->Write();
   }
   if(strstr(obj->IsA()->GetName(),"TH2")) {
     ((TH2*)obj)->Write();
   }
 }
 f->Close();
 delete f;
}
