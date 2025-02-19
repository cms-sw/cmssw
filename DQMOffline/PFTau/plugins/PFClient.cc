#include "DQMOffline/PFTau/plugins/PFClient.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// -- Constructor
//
PFClient::PFClient(const edm::ParameterSet& parameterSet)  
{
  folderNames_ = parameterSet.getParameter< std::vector<std::string> >( "FolderNames" );
  histogramNames_  = parameterSet.getParameter< std::vector<std::string> >( "HistogramNames" );
  efficiencyFlag_ =  parameterSet.getParameter< bool> ("CreateEfficiencyPlots" );
  effHistogramNames_  = parameterSet.getParameter< std::vector<std::string> >( "HistogramNamesForEfficiencyPlots" );
}
//
// -- BeginJob
//
void PFClient::beginJob() {

  dqmStore_ = edm::Service<DQMStore>().operator->();
}
//
// -- EndJobBegin Run
// 
void PFClient::endRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  doSummaries();
  if (efficiencyFlag_) doEfficiency();
}
//
// -- EndJob
// 
void PFClient::endJob() {

}
//
// -- Create Summaries
//
void PFClient::doSummaries() {

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
void PFClient::doEfficiency() {
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
void PFClient::createResolutionPlots(std::string& folder, std::string& name) {     
  MonitorElement* me = dqmStore_->get(folder+"/"+name);
  if (!me) return;
  MonitorElement* me_average;
  MonitorElement* me_rms;
  MonitorElement* me_mean;
  MonitorElement* me_sigma;
  if ( (me->kind() == MonitorElement::DQM_KIND_TH2F) ||
       (me->kind() == MonitorElement::DQM_KIND_TH2S) ||
       (me->kind() == MonitorElement::DQM_KIND_TH2D) ) {
    TH2* th = me->getTH2F();
    size_t nbinx = me->getNbinsX();
    size_t nbiny = me->getNbinsY();
    
    float ymin = th->GetYaxis()->GetXmin();
    float ymax = th->GetYaxis()->GetXmax();
    std::string xtit = th->GetXaxis()->GetTitle();
    std::string ytit = th->GetYaxis()->GetTitle();
    float* xbins = new float[nbinx+1];
    for (size_t ix = 1; ix < nbinx+1; ++ix) {
       xbins[ix-1] = th->GetBinLowEdge(ix);
       if (ix == nbinx) xbins[ix] = th->GetXaxis()->GetBinUpEdge(ix);
    }    

    std::string tit_new;
    dqmStore_->setCurrentFolder(folder);
    MonitorElement* me_slice = dqmStore_->book1D("PFlowSlice","PFlowSlice",nbiny,ymin,ymax); 
    
    tit_new = ";"+xtit+";Average_"+ytit; 
    me_average = dqmStore_->book1D("average_"+name,tit_new, nbinx, xbins); 
    tit_new = ";"+xtit+";RMS_"+ytit; 
    me_rms     = dqmStore_->book1D("rms_"+name,tit_new, nbinx, xbins); 
    tit_new = ";"+xtit+";Mean_"+ytit; 
    me_mean    = dqmStore_->book1D("mean_"+name,tit_new, nbinx, xbins); 
    tit_new = ";"+xtit+";Sigma_"+ytit; 				 
    me_sigma   = dqmStore_->book1D("sigma_"+name,tit_new, nbinx, xbins); 
				 
    double  average, rms, mean, sigma;
    for (size_t ix = 1; ix < nbinx+1; ++ix) {
      me_slice->Reset();
      for (size_t iy = 1; iy < nbiny+1; ++iy) {
	me_slice->setBinContent(iy,th->GetBinContent(ix,iy)); 
      }
      getHistogramParameters(me_slice, average, rms, mean, sigma);
      me_average->setBinContent(ix,average);
      me_rms->setBinContent(ix,rms);
      me_mean->setBinContent(ix,mean);
      me_sigma->setBinContent(ix,sigma);
    }
    if (me_slice) dqmStore_->removeElement(me_slice->getName());
    delete [] xbins;
  }
}
//
// -- Get Histogram Parameters
//
void PFClient::getHistogramParameters(MonitorElement* me_slice, double& average, 
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
void PFClient::createEfficiencyPlots(std::string& folder, std::string& name) {     
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
DEFINE_FWK_MODULE (PFClient) ;
