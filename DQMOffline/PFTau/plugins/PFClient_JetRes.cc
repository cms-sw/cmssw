#include "DQMOffline/PFTau/plugins/PFClient_JetRes.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TGraph.h"
#include "TCanvas.h"
//
// -- Constructor
//
PFClient_JetRes::PFClient_JetRes(const edm::ParameterSet& parameterSet)  
{
  folderNames_ = parameterSet.getParameter< std::vector<std::string> >( "FolderNames" );
  histogramNames_  = parameterSet.getParameter< std::vector<std::string> >( "HistogramNames" );
  efficiencyFlag_ =  parameterSet.getParameter< bool> ("CreateEfficiencyPlots" );
  effHistogramNames_  = parameterSet.getParameter< std::vector<std::string> >( "HistogramNamesForEfficiencyPlots" );
  PtBins_ = parameterSet.getParameter< std::vector<int> >( "VariablePtBins" ) ;

}
//
// -- BeginJob
//
void PFClient_JetRes::beginJob() {

  dqmStore_ = edm::Service<DQMStore>().operator->();
}
//
// -- EndJobBegin Run
// 
void PFClient_JetRes::endRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  doSummaries();
  if (efficiencyFlag_) doEfficiency();
}
//
// -- EndJob
// 
void PFClient_JetRes::endJob() {

}
//
// -- Create Summaries
//
void PFClient_JetRes::doSummaries() {

  for (std::vector<std::string>::const_iterator ifolder = folderNames_.begin();
                                    ifolder != folderNames_.end(); ifolder++) {
    std::string path  = "ParticleFlow/"+(*ifolder);

    for (std::vector<std::string>::const_iterator ihist = histogramNames_.begin();
       ihist != histogramNames_.end(); ihist++) {
      std::string hname = (*ihist); 
      createResolutionPlots(path, hname);
    }
  }
}
//
// -- Create Summaries
//
void PFClient_JetRes::doEfficiency() {
  for (std::vector<std::string>::const_iterator ifolder = folderNames_.begin();
                                    ifolder != folderNames_.end(); ifolder++) {
    std::string path  = "ParticleFlow/"+(*ifolder);

    for (std::vector<std::string>::const_iterator ihist = effHistogramNames_.begin();
	 ihist != effHistogramNames_.end(); ihist++) {
      std::string hname = (*ihist);
      createEfficiencyPlots(path, hname);
    }
  }
}
//
// -- Create Resolution Plots
//
void PFClient_JetRes::createResolutionPlots(std::string& folder, std::string& name) {     
  MonitorElement* me = dqmStore_->get(folder+"/"+name);
  if (!me) return;

  MonitorElement* pT[PtBins_.size() -1];
  std::vector<double> pTEntries(PtBins_.size()-1, 0) ;

  //std::vector<std::string> pTRange (PtBins_.size() -1) ;
  //char* pTRange[PtBins_.size() -1] ;
  TString pTRange[PtBins_.size() -1] ;
  //float pTCenter[PtBins_.size() -1] ;

  MonitorElement* me_average;
  MonitorElement* me_rms;
  MonitorElement* me_mean;
  MonitorElement* me_sigma;

  if ( (me->kind() == MonitorElement::DQM_KIND_TH2F) ||
       (me->kind() == MonitorElement::DQM_KIND_TH2S) ||
       (me->kind() == MonitorElement::DQM_KIND_TH2D) ) {
    TH2* th = me->getTH2F();
    //size_t nbinx = me->getNbinsX();
    size_t nbinx = PtBins_.size() -1;
    size_t nbiny = me->getNbinsY();
    
    float ymin = th->GetYaxis()->GetXmin();
    float ymax = th->GetYaxis()->GetXmax();
    std::string xtit = th->GetXaxis()->GetTitle();
    //std::string ytit = th->GetYaxis()->GetTitle();
    std::string ytit = "#Deltap_{T}/p_{T}";

    float* xbins = new float[nbinx+1];
    for (size_t ix = 1; ix < nbinx+1; ++ix) {
      //xbins[ix-1] = th->GetBinLowEdge(ix);
       xbins[ix-1] = PtBins_[ix-1];
       //snprintf(pTRange[ix-1].data(), 15, "Pt%d_%d", PtBins_[ix-1], PtBins_[ix]);
       pTRange[ix-1] = TString::Format("Pt%d_%d", PtBins_[ix-1], PtBins_[ix]) ;
       if (name == "BRdelta_et_Over_et_VS_et_") pTRange[ix-1] = TString::Format("BRPt%d_%d", PtBins_[ix-1], PtBins_[ix]) ;
       else if (name == "ERdelta_et_Over_et_VS_et_") pTRange[ix-1] = TString::Format("ERPt%d_%d", PtBins_[ix-1], PtBins_[ix]) ;

       //pTCenter[ix-1] = (PtBins_[ix] - PtBins_[ix-1]) / 2. ;
       if (ix == nbinx) {
	 //xbins[ix] = th->GetXaxis()->GetBinUpEdge(ix);
	 xbins[ix] = PtBins_[ix];
       }
    }    

    std::string tit_new;
    dqmStore_->setCurrentFolder(folder);
    //MonitorElement* me_slice = dqmStore_->book1D("PFlowSlice", "PFlowSlice", nbiny, ymin, ymax); 
    
    tit_new = "Average "+ytit+";"+xtit+";Average_"+ytit ; 
    me_average = dqmStore_->book1D("average_"+name,tit_new, nbinx, xbins); 
    tit_new = "RMS "+ytit+";"+xtit+";RMS_"+ytit ; 
    me_rms     = dqmStore_->book1D("rms_"+name,tit_new, nbinx, xbins); 
    tit_new = "Mean "+ytit+";"+xtit+";Mean_"+ytit ; 
    me_mean    = dqmStore_->book1D("mean_"+name,tit_new, nbinx, xbins); 
    tit_new = "Resolution "+ytit+";"+xtit+";Sigma_"+ytit ; 				 
    me_sigma   = dqmStore_->book1D("sigma_"+name,tit_new, nbinx, xbins) ; 
    
    double  average, rms, mean, sigma;
    for (size_t ix = 1; ix < nbinx+1; ++ix) {
      //me_slice->Reset();
      if (name == "delta_et_Over_et_VS_et_") pT[ix-1] = dqmStore_->book1D(pTRange[ix-1], TString::Format("Total %s;%s;Events", ytit.data(), ytit.data() ), nbiny, ymin, ymax) ;
      if (name == "BRdelta_et_Over_et_VS_et_") pT[ix-1] = dqmStore_->book1D(pTRange[ix-1], TString::Format("Barrel %s;%s;Events", ytit.data(), ytit.data()), nbiny, ymin, ymax) ;
      else if (name == "ERdelta_et_Over_et_VS_et_") pT[ix-1] = dqmStore_->book1D(pTRange[ix-1], TString::Format("Endcap %s;%s;Events", ytit.data(), ytit.data() ), nbiny, ymin, ymax) ;

      for (size_t iy = 0; iy <= nbiny+1; ++iy)  // add under and overflow
	if (th->GetBinContent(ix,iy)) {
	  //me_slice->setBinContent(iy,th->GetBinContent(ix,iy)); 
	  pT[ix-1]->setBinContent(iy, th->GetBinContent(ix,iy));
	  pT[ix-1]->setBinError(iy, th->GetBinError(ix,iy));
	  pTEntries[ix-1] += th->GetBinContent(ix,iy); }
      
      pT[ix-1]->setEntries( pTEntries[ix-1] ) ;

      //getHistogramParameters(me_slice, average, rms, mean, sigma);
      getHistogramParameters(pT[ix-1], average, rms, mean, sigma);
      me_average->setBinContent(ix,average);
      me_rms->setBinContent(ix,rms);
      me_mean->setBinContent(ix,mean);
      me_sigma->setBinContent(ix,sigma);
    }
    //if (me_slice) dqmStore_->removeElement(me_slice->getName());
    delete [] xbins;
  }
}
//
// -- Get Histogram Parameters
//
void PFClient_JetRes::getHistogramParameters(MonitorElement* me_slice, double& average, 
					     double& rms,double& mean, double& sigma) {
  average = 0.0;
  rms     = 0.0;
  mean    = 0.0;
  sigma   = 0.0;

  if (!me_slice) return;
  if  (me_slice->kind() == MonitorElement::DQM_KIND_TH1F) {
    average = me_slice->getMean();
    rms     = me_slice->getRMS();  
    TH1F* th_slice = me_slice->getTH1F();
    if (th_slice && th_slice->GetEntries() > 0) {
      th_slice->Fit( "gaus","Q0");
      TF1* gaus = th_slice->GetFunction( "gaus" );
      if (gaus) {
	sigma = gaus->GetParameter(2);
        mean  = gaus->GetParameter(1);
      }
    }
  }
}
//
// -- Create Resolution Plots
//
void PFClient_JetRes::createEfficiencyPlots(std::string& folder, std::string& name) {     
  MonitorElement* me1 = dqmStore_->get(folder+"/"+name);
  MonitorElement* me2 = dqmStore_->get(folder+"/"+name+"ref_");
  if (!me1 || !me2) return;
  MonitorElement* me_eff;
  if ( (me1->kind() == MonitorElement::DQM_KIND_TH1F) &&
       (me1->kind() == MonitorElement::DQM_KIND_TH1F) ) {
    TH1* th1 = me1->getTH1F();
    size_t nbinx = me1->getNbinsX();
    
    float xmin = th1->GetXaxis()->GetXmin();
    float xmax = th1->GetXaxis()->GetXmax();
    std::string xtit = me1->getAxisTitle(1);
    std::string tit_new;
    tit_new = ";"+xtit+";Efficiency"; 

    dqmStore_->setCurrentFolder(folder);
    me_eff = dqmStore_->book1D("efficiency_"+name,tit_new, nbinx, xmin, xmax); 
				 
    double  efficiency;
    me_eff->Reset();
    for (size_t ix = 1; ix < nbinx+1; ++ix) {
      float val1 = me1->getBinContent(ix);
      float val2 = me2->getBinContent(ix);
      if (val2 > 0.0) efficiency = val1/val2;
      else efficiency = 0;   
      me_eff->setBinContent(ix,efficiency);
    }
  }
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE (PFClient_JetRes) ;
