#include "DQMOffline/Trigger/interface/HLTTauDQMSummaryPlotter.h"

#include "TEfficiency.h"

#include<tuple>

namespace {
  std::tuple<float, float> calcEfficiency(float num, float denom) {
    if(denom == 0.0f)
      return std::make_tuple(0.0f, 0.0f);

    //float eff = num/denom;
    constexpr double cl = 0.683f; // "1-sigma"
    const float eff = num/denom;
    const float errDown = TEfficiency::ClopperPearson(denom, num, cl, false);
    const float errUp = TEfficiency::ClopperPearson(denom, num, cl, true);

    // Because of limitation of TProfile, just take max
    return std::make_tuple(eff, std::max(eff-errDown, errUp-eff));
  }
}

HLTTauDQMSummaryPlotter::HLTTauDQMSummaryPlotter(const edm::ParameterSet& ps, const std::string& dqmBaseFolder):
  HLTTauDQMPlotter(ps, dqmBaseFolder),
  store_(nullptr)
{
  if(!configValid_)
    return;

  // no run concept in summary plotter
  runValid_ = true;
    
  //Process PSet
  try {
    type_ = ps.getUntrackedParameter<std::string>("ConfigType");
  } catch ( cms::Exception &e ) {
    edm::LogWarning("HLTTauDQMOfflineSource") << "HLTTauDQMSummaryPlotter::HLTTauDQMSummaryPlotter(): " << e.what();
    configValid_ = false;
    return;
  }
  configValid_ = true;
}

HLTTauDQMSummaryPlotter::~HLTTauDQMSummaryPlotter() {}

void HLTTauDQMSummaryPlotter::bookPlots() {
  if(!configValid_)
    return;

  edm::Service<DQMStore> store;
  if(store.isAvailable()) {
    store_ = store.operator->();
        //Path Summary 
        if ( type_ == "Path" ) {
            bookTriggerBitEfficiencyHistos(triggerTag(), "EventsPerFilter");

            bookEfficiencyHisto(triggerTag(), "L2TrigTauEtEff",  "helpers/L2TrigTauEtEffNum"); 
            bookEfficiencyHisto(triggerTag(), "L2TrigTauHighEtEff",  "helpers/L2TrigTauHighEtEffNum");
            bookEfficiencyHisto(triggerTag(), "L2TrigTauEtaEff", "helpers/L2TrigTauEtaEffNum");
            bookEfficiencyHisto(triggerTag(), "L2TrigTauPhiEff", "helpers/L2TrigTauPhiEffNum");

            bookEfficiencyHisto(triggerTag(), "L3TrigTauEtEff",  "helpers/L3TrigTauEtEffNum");
            bookEfficiencyHisto(triggerTag(), "L3TrigTauHighEtEff",  "helpers/L3TrigTauHighEtEffNum");
            bookEfficiencyHisto(triggerTag(), "L3TrigTauEtaEff", "helpers/L3TrigTauEtaEffNum");
            bookEfficiencyHisto(triggerTag(), "L3TrigTauPhiEff", "helpers/L3TrigTauPhiEffNum");
        }
        
        //L1 Summary
        else if ( type_ == "L1" ) {
            bookEfficiencyHisto(triggerTag(),"L1TauEtEff","EfficiencyHelpers/L1TauEtEffNum");
            bookEfficiencyHisto(triggerTag(),"L1TauHighEtEff","EfficiencyHelpers/L1TauHighEtEffNum");
            bookEfficiencyHisto(triggerTag(),"L1TauEtaEff","EfficiencyHelpers/L1TauEtaEffNum");
            bookEfficiencyHisto(triggerTag(),"L1TauPhiEff","EfficiencyHelpers/L1TauPhiEffNum");
            
            bookEfficiencyHisto(triggerTag(),"L1JetEtEff","EfficiencyHelpers/L1JetEtEffNum");
            bookEfficiencyHisto(triggerTag(),"L1JetHighEtEff","EfficiencyHelpers/L1JetHighEtEffNum");
            bookEfficiencyHisto(triggerTag(),"L1JetEtaEff","EfficiencyHelpers/L1JetEtaEffNum");
            bookEfficiencyHisto(triggerTag(),"L1JetPhiEff","EfficiencyHelpers/L1JetPhiEffNum");
        }

        else if(type_ == "PathSummary") {
          bookEfficiencyHisto(triggerTag(), "PathEfficiency", "helpers/PathTriggerBits");
        }
  }
  store_ = nullptr;
}

void HLTTauDQMSummaryPlotter::plot() {
  edm::Service<DQMStore> store;
  if(store.isAvailable()) {
    store_ = store.operator->();
        //Path Summary 
        if ( type_ == "Path" ) {
            plotTriggerBitEfficiencyHistos(triggerTag(), "EventsPerFilter");

            plotEfficiencyHisto(triggerTag(), "L2TrigTauEtEff",  "helpers/L2TrigTauEtEffNum",  "helpers/L2TrigTauEtEffDenom");
            plotEfficiencyHisto(triggerTag(), "L2TrigTauHighEtEff",  "helpers/L2TrigTauHighEtEffNum",  "helpers/L2TrigTauHighEtEffDenom");
            plotEfficiencyHisto(triggerTag(), "L2TrigTauEtaEff", "helpers/L2TrigTauEtaEffNum", "helpers/L2TrigTauEtaEffDenom");
            plotEfficiencyHisto(triggerTag(), "L2TrigTauPhiEff", "helpers/L2TrigTauPhiEffNum", "helpers/L2TrigTauPhiEffDenom");

            plotEfficiencyHisto(triggerTag(), "L3TrigTauEtEff",  "helpers/L3TrigTauEtEffNum",  "helpers/L3TrigTauEtEffDenom");
            plotEfficiencyHisto(triggerTag(), "L3TrigTauHighEtEff",  "helpers/L3TrigTauHighEtEffNum",  "helpers/L3TrigTauHighEtEffDenom");
            plotEfficiencyHisto(triggerTag(), "L3TrigTauEtaEff", "helpers/L3TrigTauEtaEffNum", "helpers/L3TrigTauEtaEffDenom");
            plotEfficiencyHisto(triggerTag(), "L3TrigTauPhiEff", "helpers/L3TrigTauPhiEffNum", "helpers/L3TrigTauPhiEffDenom");
        }
        
        //L1 Summary
        else if ( type_ == "L1" ) {
            plotEfficiencyHisto(triggerTag(),"L1TauEtEff","EfficiencyHelpers/L1TauEtEffNum","EfficiencyHelpers/L1TauEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1TauHighEtEff","EfficiencyHelpers/L1TauHighEtEffNum","EfficiencyHelpers/L1TauHighEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1TauEtaEff","EfficiencyHelpers/L1TauEtaEffNum","EfficiencyHelpers/L1TauEtaEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1TauPhiEff","EfficiencyHelpers/L1TauPhiEffNum","EfficiencyHelpers/L1TauPhiEffDenom");
            
            plotEfficiencyHisto(triggerTag(),"L1JetEtEff","EfficiencyHelpers/L1JetEtEffNum","EfficiencyHelpers/L1JetEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1JetHighEtEff","EfficiencyHelpers/L1JetHighEtEffNum","EfficiencyHelpers/L1JetHighEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1JetEtaEff","EfficiencyHelpers/L1JetEtaEffNum","EfficiencyHelpers/L1JetEtaEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1JetPhiEff","EfficiencyHelpers/L1JetPhiEffNum","EfficiencyHelpers/L1JetPhiEffDenom");
            
            plotEfficiencyHisto(triggerTag(),"L1ElectronEtEff","EfficiencyHelpers/L1ElectronEtEffNum","EfficiencyHelpers/L1ElectronEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1ElectronEtaEff","EfficiencyHelpers/L1ElectronEtaEffNum","EfficiencyHelpers/L1ElectronEtaEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1ElectronPhiEff","EfficiencyHelpers/L1ElectronPhiEffNum","EfficiencyHelpers/L1ElectronPhiEffDenom");
            
            plotEfficiencyHisto(triggerTag(),"L1MuonEtEff","EfficiencyHelpers/L1MuonEtEffNum","EfficiencyHelpers/L1MuonEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1MuonEtaEff","EfficiencyHelpers/L1MuonEtaEffNum","EfficiencyHelpers/L1MuonEtaEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1MuonPhiEff","EfficiencyHelpers/L1MuonPhiEffNum","EfficiencyHelpers/L1MuonPhiEffDenom");
        }

        else if(type_ == "PathSummary") {
          plotEfficiencyHisto(triggerTag(), "PathEfficiency", "helpers/PathTriggerBits", "helpers/RefEvents");
        }
  }
  store_ = nullptr;
}      

void HLTTauDQMSummaryPlotter::bookEfficiencyHisto( std::string folder, std::string name, std::string hist1 ) {
    if ( store_->dirExists(folder) ) {
        store_->setCurrentFolder(folder);
        
        MonitorElement * effnum = store_->get(folder+"/"+hist1);
        
        if ( effnum ) {            
            MonitorElement *tmp = store_->bookProfile(name,name,effnum->getTH1F()->GetNbinsX(),effnum->getTH1F()->GetXaxis()->GetXmin(),effnum->getTH1F()->GetXaxis()->GetXmax(),105,0,1.05);
            
            tmp->setTitle(name);
            tmp->setAxisTitle(effnum->getAxisTitle(), 1); // X
            tmp->setAxisTitle("Efficiency", 2);
        }
    }
}

void HLTTauDQMSummaryPlotter::plotEfficiencyHisto( std::string folder, std::string name, std::string hist1, std::string hist2 ) {
    if ( store_->dirExists(folder) ) {
        store_->setCurrentFolder(folder);
        
        MonitorElement * effnum = store_->get(folder+"/"+hist1);
        MonitorElement * effdenom = store_->get(folder+"/"+hist2);
        MonitorElement * eff = store_->get(folder+"/"+name);
        
        if(effnum && effdenom && eff) {
          const TH1F *num = effnum->getTH1F();
          const TH1F *denom = effdenom->getTH1F();
          TProfile *prof = eff->getTProfile();
          for (int i = 1; i <= num->GetNbinsX(); ++i) {
            std::tuple<float, float> effErr = calcEfficiency(num->GetBinContent(i), denom->GetBinContent(i));
            const float efficiency = std::get<0>(effErr);
            const float err = std::get<1>(effErr);
            prof->SetBinContent(i, efficiency);
            prof->SetBinEntries(i, 1);
            prof->SetBinError(i, std::sqrt(efficiency*efficiency + err*err)); // why simple SetBinError(err) does not work?
          }
        }
    }
}

void HLTTauDQMSummaryPlotter::plotIntegratedEffHisto( std::string folder, std::string name, std::string refHisto, std::string evCount, int bin ) {
    if ( store_->dirExists(folder) ) {
        store_->setCurrentFolder(folder);
        
        MonitorElement * refH = store_->get(folder+"/"+refHisto);
        MonitorElement * evC = store_->get(folder+"/"+evCount);
        MonitorElement * eff = store_->get(folder+"/"+name);
        
        if ( refH && evC && eff ) {
            TH1F *histo = refH->getTH1F();
            float nGenerated = evC->getTH1F()->GetBinContent(bin);
            // Assuming that the histogram is incremented with weight=1 for each event
            // this function integrates the histogram contents above every bin and stores it
            // in that bin.  The result is plot of integral rate versus threshold plot.
            int nbins = histo->GetNbinsX();
            double integral = histo->GetBinContent(nbins+1);  // Initialize to overflow
            if (nGenerated<=0.0) nGenerated=1.0;
            for ( int i = nbins; i >= 1; i-- ) {
                double thisBin = histo->GetBinContent(i);
                integral += thisBin;
                double integralEff;
                double integralError;
                integralEff = (integral / nGenerated);
                eff->getTProfile()->SetBinContent(i, integralEff);
                eff->getTProfile()->SetBinEntries(i, 1);
                // error
                integralError = (sqrt(integral) / nGenerated);
                
                eff->getTProfile()->SetBinError(i, sqrt(integralEff*integralEff+integralError*integralError));
            }
        }
    }
}

void HLTTauDQMSummaryPlotter::bookTriggerBitEfficiencyHistos( std::string folder, std::string histo ) {
    if ( store_->dirExists(folder) ) {
        store_->setCurrentFolder(folder);
        
        MonitorElement * eff = store_->get(folder+"/"+histo);
        
        if ( eff ) {
          //store_->bookProfile("EfficiencyRefInput","Efficiency with Matching",eff->getNbinsX()-1,0,eff->getNbinsX()-1,100,0,1);
          //store_->bookProfile("EfficiencyRefL1","Efficiency with Matching Ref to L1",eff->getNbinsX()-2,0,eff->getNbinsX()-2,100,0,1);
          MonitorElement *me_prev = store_->bookProfile("EfficiencyRefPrevious","Efficiency to Previous",eff->getNbinsX()-1,0,eff->getNbinsX()-1,100,0,1);
          const TAxis *xaxis = eff->getTH1F()->GetXaxis();
          for(int bin=1; bin < eff->getNbinsX(); ++bin) {
            me_prev->setBinLabel(bin, xaxis->GetBinLabel(bin));
          }
        }
    }
}

void HLTTauDQMSummaryPlotter::plotTriggerBitEfficiencyHistos( std::string folder, std::string histo ) {
    if ( store_->dirExists(folder) ) {
        store_->setCurrentFolder(folder);
        MonitorElement * eff = store_->get(folder+"/"+histo);
        //MonitorElement * effRefTruth = store_->get(folder+"/EfficiencyRefInput");
        //MonitorElement * effRefL1 = store_->get(folder+"/EfficiencyRefL1");
        MonitorElement * effRefPrevious = store_->get(folder+"/EfficiencyRefPrevious");
        
        //if ( eff && effRefTruth && effRefL1 && effRefPrevious ) {
        if (eff && effRefPrevious) {
          TProfile *previous = effRefPrevious->getTProfile();

          //Calculate efficiencies with ref to matched objects
          /*
            for ( int i = 2; i <= eff->getNbinsX(); ++i ) {
                double efficiency = calcEfficiency(eff->getBinContent(i),eff->getBinContent(1)).first;
                double err = calcEfficiency(eff->getBinContent(i),eff->getBinContent(1)).second;
                
                effRefTruth->getTProfile()->SetBinContent(i-1,efficiency);
                effRefTruth->getTProfile()->SetBinEntries(i-1,1);
                effRefTruth->getTProfile()->SetBinError(i-1,sqrt(efficiency*efficiency+err*err));
                effRefTruth->setBinLabel(i-1,eff->getTH1F()->GetXaxis()->GetBinLabel(i));
                
            }
            //Calculate efficiencies with ref to L1
            for ( int i = 3; i <= eff->getNbinsX(); ++i ) {
                double efficiency = calcEfficiency(eff->getBinContent(i),eff->getBinContent(2)).first;
                double err = calcEfficiency(eff->getBinContent(i),eff->getBinContent(2)).second;
                
                effRefL1->getTProfile()->SetBinContent(i-2,efficiency);
                effRefL1->getTProfile()->SetBinEntries(i-2,1);
                effRefL1->getTProfile()->SetBinError(i-2,sqrt(efficiency*efficiency+err*err));
                effRefL1->setBinLabel(i-2,eff->getTH1F()->GetXaxis()->GetBinLabel(i));
            }
          */
            //Calculate efficiencies with ref to previous
            for ( int i = 2; i <= eff->getNbinsX(); ++i ) {
              const std::tuple<float, float> effErr = calcEfficiency(eff->getBinContent(i), eff->getBinContent(i-1));
              const float efficiency = std::get<0>(effErr);
              const float err = std::get<1>(effErr);
                
              previous->SetBinContent(i-1, efficiency);
              previous->SetBinEntries(i-1, 1);
              previous->SetBinError(i-1, std::sqrt(efficiency*efficiency + err*err)); // why simple SetBinError(err) does not work?
              effRefPrevious->setBinLabel(i-1,eff->getTH1F()->GetXaxis()->GetBinLabel(i));
            }
        }
    }
}
