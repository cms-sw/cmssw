#include "CommonTools/TrackerMap/interface/TmTestWidget.h"
TmTestWidget::TmTestWidget(QWidget *parent, const char *wname)
        : QWidget(parent,wname)
 {
    VisTrackerMapBox* tkMapBox = new VisTrackerMapBox(0,"trackermap");
   QGridLayout *grid = new QGridLayout( this, 1, 1, 10 );
    TrackerMap *tm = tkMapBox->getTrackerMap();
    tkMapBox->show();
    f = new TFile("Tracker.root");   
   char  name[10];
    TString title;
     string s;
    int key;
   Int_t i;
     i=0;
     map<const int,TmModule*>::iterator imod;
    for (imod=IdModuleMap::imoduleMap.begin();imod != IdModuleMap::imoduleMap.end(); imod++){
     i++;if(i%100==0)cout << i << endl;
     sprintf(name,"%d",imod->second->idex);
     TH1F *h = (TH1F*)f->Get(name);
     if(h>0){
     title= h->GetTitle();
    tm->fill(imod->second->idex,(float) h->GetMean());
    ostringstream outs;
    outs<< "Mean Adc "<<h->GetMean()<<"  RMS ADC "<<h->GetRMS()<<"  ;"<<h->GetEntries()<<" digi in this module";
    s = outs.str();
     //cout <<key << " "<< s  << endl;
    tm->setText(imod->second->idex,s);
    }
    }


      TCanvas* c=new TCanvas( "c", "Adc map", 400, 500);
   //setup paint options defines the same variables that where used before
  //calling the print method of trackermap
    tkMapBox->setPaintOptions(true,0,500);
   //grid->addWidget( tkMapBox, 0, 0 );
   QObject::connect(tkMapBox, SIGNAL(moduleSelected(int)),this,SLOT(write(int)));
                                                                                
  //to be called when you want to repaint the trackemap
    tkMapBox->update();

 }
  void TmTestWidget::paintEvent(QPaintEvent *){
             QPainter *p = new QPainter( this );
             QPixmap pxm;
            pxm.load("1000001.ppm");
            p->drawPixmap(0,0,pxm);
  }
  void TmTestWidget::write(int moduleGeomdetId){cout <<  "module" << moduleGeomdetId <<"  clicked!" << endl;
      char  name[10];
     sprintf(name,"%d",moduleGeomdetId);
     TH1F *h = (TH1F*)f->Get(name);
     if(h>0){
     TPostScript myps("1000.ps",113);
     h->Draw();
     myps.Close();
     system("pstopnm 1000.ps");
     system("rm -f 1000.ps");
     repaint();
      }
} 
TmTestWidget::~TmTestWidget(){
    f->Close();
    }
