#include "HcalVisualSelector.h"
#include "PlotAllDisplay.h"

class PlotAllAdapter : public HcalVisualSelector::PlotCallback {
public:
  PlotAllAdapter(PlotAllDisplay* disp, int evtType, int flavType) :
    m_disp(disp), m_evtType(evtType), m_flavType(flavType){
  }
  virtual void plot(const MyHcalDetId& id) {
    m_disp->displayOne(id.ieta,id.iphi,id.depth,m_evtType,m_flavType);
  }
private:
  PlotAllDisplay* m_disp;
  int m_evtType, m_flavType;
};

std::vector<MyHcalDetId>
PlotAllDisplay::spatialFilter(int ieta,
			      int iphi,
			      const std::vector<MyHcalDetId>& inputs)
{
  std::vector<MyHcalDetId> retval;
  std::vector<MyHcalDetId>::const_iterator ii;
  for (ii=inputs.begin(); ii!=inputs.end(); ii++) {
    if (iphi!=0 && ii->iphi!=iphi) continue;
    if (ieta!=0 && ii->ieta!=ieta) continue;
    retval.push_back(*ii);
  }  
  return retval;
}


TH1* PlotAllDisplay::bookMasterHistogram(DisplaySetupStruct& ss,
					 const std::string& basename, int lo,
					 int hi)
{
  char name[120];
  TH1* retval=0;
  if (ss.iphi!=0) {
    sprintf(name,"%s:%s-%s IPHI=%d",
	    ss.eventTypeStr.c_str(),ss.flavTypeStr.c_str(),basename.c_str(),ss.iphi);
    retval=new TH1F(name,name, hi-lo+1, lo-0.5, hi+0.5);
    retval->GetXaxis()->SetTitle("IETA");
  } else {
    sprintf(name,"%s:%s-%s IETA=%d",
	    ss.eventTypeStr.c_str(),ss.flavTypeStr.c_str(),basename.c_str(),ss.ieta);
    retval=new TH1F(name,name, hi-lo+1, lo-0.5, hi+0.5);
    retval->GetXaxis()->SetTitle("IPHI");
  }
  retval->SetDirectory(0);
  retval->SetStats(0);
  return retval;
}

MyHcalSubdetector PlotAllDisplay::getSubDetector(int ieta,int depth) {
  MyHcalSubdetector retval=HcalEmpty;
  int aieta = abs(ieta);

  if(aieta<=16 && depth<=2) retval=HcalBarrel;
  if(aieta<=15 && depth==4) retval=HcalOuter;
  if( (aieta==16&&depth==3)||(aieta>16&&aieta<30) ) retval=HcalEndcap;
  if(aieta>29) retval=HcalForward;

  if(retval==HcalEmpty) printf("Bad detector coordinates!\n");

  return retval;
}


void PlotAllDisplay::displaySummary(int ieta, int iphi, int evtType, int flavType) {
  HistoManager::EventType et=(HistoManager::EventType)evtType;
  HistoManager::HistType ht=(HistoManager::HistType)flavType;
  DisplaySetupStruct setup;
  setup.ieta=ieta;
  setup.iphi=iphi;
  setup.eventTypeStr=HistoManager::nameForEvent(et);
  setup.flavTypeStr=HistoManager::nameForFlavor(ht);

  std::vector<MyHcalDetId> KeyIds = histKeys.getDetIdsForType(ht,et);
  std::vector<MyHcalDetId> ids = spatialFilter(ieta,iphi,KeyIds);

  if (ids.size()==0) {
    printf("The iphi or ieta value entered was not found!\n");
    printf("Make sure the correct event type is selected and the correct ieta/iphi values are entered.\n");
    return;
  }

  std::vector<MyHcalDetId>::const_iterator ii;

  // Sum of all channels
  TH1* sum=0;
  for (ii=ids.begin(); ii!=ids.end(); ii++) {
    TH1* h=histKeys.GetAHistogram(*ii,ht,et);
    if (h==0) continue;
    if (sum==0) {
      sum=(TH1*)h->Clone("All");
      sum->SetDirectory(0);
      char name[120];
      sprintf(name,"All %s:%s",setup.eventTypeStr.c_str(),setup.flavTypeStr.c_str());
      sum->SetTitle(name);
    } else sum->Add(h);
  }

  TCanvas* c=new TCanvas("All","All",60,60,800,600);
  c->cd();
  sum->Draw();
  sum->Draw("SAMEHIST");

  if (ht==HistoManager::PULSE) return;

  // profile of an ieta or iphi

  if (iphi!=0 || ieta!=0) {
    TH1* meanSummary[5];
    TH1* RMSSummary[5];

    int range_lo=100000, range_hi=-100000;
    for(ii=ids.begin(); ii!=ids.end(); ii++) {
      int ibin=(iphi!=0)?(ii->ieta):(ii->iphi);
      if (ibin>range_hi) range_hi=ibin;
      if (ibin<range_lo) range_lo=ibin;
    }

    meanSummary[0]=bookMasterHistogram(setup,"MEAN",range_lo,range_hi);
    RMSSummary[0]=bookMasterHistogram(setup,"RMS",range_lo,range_hi);
    for (int j=1; j<5; j++) {
      int marker=j+23;
      if (j==4) marker=30;

      char aname[120];
      sprintf(aname,"Mean_Depth%d",j);
      meanSummary[j]=new TH1F(aname,aname,(range_hi-range_lo)+1,range_lo-0.5,range_hi+0.5);
      meanSummary[j]->SetDirectory(0);
      meanSummary[j]->SetMarkerStyle(marker);
      meanSummary[j]->SetMarkerColor(j);

      sprintf(aname,"RMS_Depth%d",j);
      RMSSummary[j]=new TH1F(aname,aname,(range_hi-range_lo)+1,range_lo-0.5,range_hi+0.5);
      RMSSummary[j]->SetDirectory(0);
      RMSSummary[j]->SetMarkerStyle(marker);
      RMSSummary[j]->SetMarkerColor(j);
    }

    for(ii=ids.begin(); ii!=ids.end(); ii++) {
      TH1* h=histKeys.GetAHistogram(*ii,ht,et);
      if (h==0) continue;
      double bin=(iphi!=0)?(ii->ieta*1.0):(ii->iphi*1.0);
      meanSummary[ii->depth]->Fill(bin, h->GetMean());
      RMSSummary[ii->depth]->Fill(bin, h->GetRMS());
    }

    double ml=1e16,mh=-1e16;
    for (int j=1; j<5; j++) {
      for (int jj=1; jj<=meanSummary[j]->GetNbinsX(); jj++)
	if (meanSummary[j]->GetBinError(jj)==0.0)
	  meanSummary[j]->SetBinContent(jj,-1e6);
	else {
	  if (meanSummary[j]->GetBinContent(jj)<ml)
	    ml=meanSummary[j]->GetBinContent(jj);
	  if (meanSummary[j]->GetBinContent(jj)>mh)
	    mh=meanSummary[j]->GetBinContent(jj);
	}
    }
    meanSummary[0]->SetMaximum(mh+(mh-ml)*0.05);
    meanSummary[0]->SetMinimum(ml-(mh-ml)*0.05);

    ml=1e16,mh=-1e16;
    for (int j=1; j<5; j++) {
      for (int jj=1; jj<=RMSSummary[j]->GetNbinsX(); jj++)
	if (RMSSummary[j]->GetBinError(jj)==0.0)
	  RMSSummary[j]->SetBinContent(jj,-1e6);
	else {
	  if (RMSSummary[j]->GetBinContent(jj)<ml)
	    ml=RMSSummary[j]->GetBinContent(jj);
	  if (RMSSummary[j]->GetBinContent(jj)>mh)
	    mh=RMSSummary[j]->GetBinContent(jj);
	}
    }
    RMSSummary[0]->SetMaximum(mh+(mh-ml)*0.05);
    RMSSummary[0]->SetMinimum(ml-(mh-ml)*0.05);


    TCanvas* myplot=new TCanvas(setup.eventTypeStr.c_str(),
				setup.eventTypeStr.c_str(),
				20,20,800,600);
    myplot->Divide(1,2);

    myplot->cd(1);
    meanSummary[0]->Draw("P");
    for (int j=1; j<5; j++)
      meanSummary[j]->Draw("SAMEP");
    myplot->cd(2);
    RMSSummary[0]->Draw("P");
    for (int j=1; j<5; j++)
      RMSSummary[j]->Draw("SAMEP");
  }
  
  // global distributions
  
  {
    double mean_lo=1e160, mean_hi=-1e160;
    double RMS_lo=1e160, RMS_hi=-1e160;
    for (ii=ids.begin(); ii!=ids.end(); ii++) {
      TH1* h=histKeys.GetAHistogram(*ii,ht,et);
      if (h==0) continue;
      double mean=h->GetMean();
      double RMS=h->GetRMS();
      if (mean<mean_lo) mean_lo=mean;
      if (mean>mean_hi) mean_hi=mean;
      if (RMS<RMS_lo) RMS_lo=RMS;
      if (RMS>RMS_hi) RMS_hi=RMS;
    }

    //adjust range to include endpoints
    mean_lo = mean_lo - 0.05*(mean_hi-mean_lo);
    mean_hi = mean_hi + 0.05*(mean_hi-mean_lo);
    RMS_lo = RMS_lo - 0.05*(RMS_hi-RMS_lo);
    RMS_hi = RMS_hi + 0.05*(RMS_hi-RMS_lo);

    TH1* means=new TH1F("MEANS","MEANS",50,mean_lo,mean_hi);
    means->SetDirectory(0);

    TH1* RMSs=new TH1F("RMSs","RMSs",50,RMS_lo,RMS_hi);
    RMSs->SetDirectory(0);

    for (ii=ids.begin(); ii!=ids.end(); ii++) {
      TH1* h=histKeys.GetAHistogram(*ii,ht,et);
      if (h==0) continue;
      means->Fill(h->GetMean());
      RMSs->Fill(h->GetRMS());
    } 
    
    TCanvas * myplot = new TCanvas("Statistics","Statistics",
				   40,40,800,600);
    
    myplot->Divide(1,2);
    
    myplot->cd(1);
    means->Draw();
    myplot->cd(2);
    RMSs->Draw();
  }
}

void PlotAllDisplay::displayOne(int ieta, int iphi, int depth, int evtType, int FlavType)
{
  HistoManager::EventType et=(HistoManager::EventType)evtType;
  HistoManager::HistType ht=(HistoManager::HistType)FlavType;
  DisplaySetupStruct setup;
  setup.ieta=ieta;
  setup.iphi=iphi;
  setup.eventTypeStr=HistoManager::nameForEvent(et);
  setup.flavTypeStr=HistoManager::nameForFlavor(ht);

  MyHcalSubdetector subDet = getSubDetector(ieta,depth);

  MyHcalDetId id = {subDet,ieta,iphi,depth};

  TH1* h=histKeys.GetAHistogram(id,ht,et);
  
  if (h==0) {
    printf("The ieta and/or iphi values were not found!\n");
    printf("Make sure the correct event type is selected and the correct iphi/ieta values are entered.\n");
    return;
  }

  if (m_movie==0) {
    m_movie=new TCanvas("Selected","Selected",50,50,800,600);
    m_movie->Divide(3,2);
  }

  n_movie=(n_movie%6)+1;
  m_movie->cd(n_movie);

  if (ht==HistoManager::PULSE) 
    h->Draw("HIST");
  else 
    h->Draw();
  m_movie->Flush();
  m_movie->Update();
  m_movie->Paint();
}

void PlotAllDisplay::displaySelector(int evtType, int flavType) {
  HistoManager::EventType et=(HistoManager::EventType)evtType;
  HistoManager::HistType ht=(HistoManager::HistType)flavType;

  std::vector<MyHcalDetId> KeyIds = histKeys.getDetIdsForType(ht,et);

  if (KeyIds.empty()) {
    printf("No histograms found for specified event type and flavor type\n");
    return;
  }

  int ieta_lo=10000; 
  int ieta_hi=-10000;
  int iphi_lo=10000;
  int iphi_hi=-10000;
  for(std::vector<MyHcalDetId>::iterator jj=KeyIds.begin();
      jj!=KeyIds.end();
      jj++) {
    if( jj->ieta>ieta_hi ) ieta_hi=jj->ieta;
    if( jj->ieta<ieta_lo ) ieta_lo=jj->ieta;
    if( jj->iphi>iphi_hi ) iphi_hi=jj->iphi;
    if( jj->iphi<iphi_lo ) iphi_lo=jj->iphi;
  }

  //  printf("eta_lo=%d eta_hi=%d phi_lo=%d phi_hi=%d\n",
  //          ieta_lo,ieta_hi,iphi_lo,iphi_hi);
  
  HcalVisualSelector* vs=
    new HcalVisualSelector(new PlotAllAdapter(this,evtType,flavType),
			   ieta_lo,ieta_hi,iphi_lo,iphi_hi);

  for (std::vector<MyHcalDetId>::iterator ii=KeyIds.begin();
       ii!=KeyIds.end(); 
       ii++) {
    TH1* h=histKeys.GetAHistogram(*ii,ht,et);
    if (h==0) {
      printf("ieta=%d, iphi=%d not found\n", ii->ieta, ii->iphi);
      continue;
    }
    vs->fill(*ii,h->GetMean());
  }

  vs->Update();
}
