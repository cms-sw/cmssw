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

class HLTTauDQMSummaryPlotter::SummaryPlotter {
public:
  enum class Type {
    kL1,
    kPath,
    kPathSummary,
    kUnknown
  };

  SummaryPlotter(const std::string& type, const std::string& folder, DQMStore& store);

  void plot(DQMStore& store);

private:
  void bookEfficiencyHisto(const std::string& name, const std::string& hist1, bool copyLabels=false);
  void plotEfficiencyHisto(std::string name, std::string hist1, std::string hist2 );
  void plotIntegratedEffHisto(std::string name, std::string refHisto, std::string evCount, int bin );
  void bookTriggerBitEfficiencyHistos(std::string histo );
  void plotTriggerBitEfficiencyHistos(std::string histo );
  void bookFractionHisto(const std::string& name);
  void plotFractionHisto(const std::string& name);

  const std::string folder_;
  DQMStore *store_;
  Type type_;
};

HLTTauDQMSummaryPlotter::HLTTauDQMSummaryPlotter(const edm::ParameterSet& ps, const std::string& dqmBaseFolder, const std::string& type):
  HLTTauDQMPlotter(ps, dqmBaseFolder),
  type_(type)
{}
HLTTauDQMSummaryPlotter::HLTTauDQMSummaryPlotter(const std::string& dqmBaseFolder, const std::string& type):
  HLTTauDQMPlotter("", dqmBaseFolder),
  type_(type)
{}

HLTTauDQMSummaryPlotter::~HLTTauDQMSummaryPlotter() {}

void HLTTauDQMSummaryPlotter::bookPlots() {
  if(!configValid_)
    return;

  edm::Service<DQMStore> hstore;
  if(!hstore.isAvailable())
    return;

  DQMStore& store = *hstore;
  if(type_ == "L1") {
    plotters_.emplace_back(new SummaryPlotter(type_, triggerTag(), store));
  }
  else if(type_ == "Path") {
    std::string path = triggerTag();
    path.pop_back(); // strip of trailing /
    if(store.dirExists(path)) {
      store.setCurrentFolder(path);
      for(const std::string& subfolder: store.getSubdirs()) {
        std::size_t pos = subfolder.rfind("/");
        if(pos == std::string::npos)
          continue;
        ++pos; // position of first letter after /
        if(subfolder.compare(pos, 4, "HLT_") == 0) { // start with HLT_
          LogDebug("HLTTauDQMOffline") << "SummaryPlotter: processing path " << subfolder.substr(pos);
          plotters_.emplace_back(new SummaryPlotter(type_, subfolder, store));
        }
      }
    }
  }
  else if(type_ == "PathSummary") {
    plotters_.emplace_back(new SummaryPlotter(type_, triggerTag(), store));
  }
}

void HLTTauDQMSummaryPlotter::plot() {
  if(!isValid())
    return;

  edm::Service<DQMStore> store;
  if(store.isAvailable()) {
    for(auto& plotter: plotters_) {
      plotter->plot(*store);
    }
  }
}
 
HLTTauDQMSummaryPlotter::SummaryPlotter::SummaryPlotter(const std::string& type, const std::string& folder, DQMStore& store):
  folder_(folder),
  store_(nullptr),
  type_(Type::kUnknown)
{
  if(type == "L1") type_ = Type::kL1;
  else if(type == "Path") type_ = Type::kPath;
  else if(type == "PathSummary") type_ = Type::kPathSummary;

  store_ = &store;

  //Path Summary 
  if ( type_ == Type::kPath ) {
    bookTriggerBitEfficiencyHistos("EventsPerFilter");

    bookEfficiencyHisto("L2TrigTauEtEff",  "helpers/L2TrigTauEtEffNum"); 
    bookEfficiencyHisto("L2TrigTauHighEtEff",  "helpers/L2TrigTauHighEtEffNum");
    bookEfficiencyHisto("L2TrigTauEtaEff", "helpers/L2TrigTauEtaEffNum");
    bookEfficiencyHisto("L2TrigTauPhiEff", "helpers/L2TrigTauPhiEffNum");

    bookEfficiencyHisto("L3TrigTauEtEff",  "helpers/L3TrigTauEtEffNum");
    bookEfficiencyHisto("L3TrigTauHighEtEff",  "helpers/L3TrigTauHighEtEffNum");
    bookEfficiencyHisto("L3TrigTauEtaEff", "helpers/L3TrigTauEtaEffNum");
    bookEfficiencyHisto("L3TrigTauPhiEff", "helpers/L3TrigTauPhiEffNum");
  }

  //L1 Summary
  else if ( type_ == Type::kL1 ) {
    bookEfficiencyHisto("L1TauEtEff","EfficiencyHelpers/L1TauEtEffNum");
    bookEfficiencyHisto("L1TauHighEtEff","EfficiencyHelpers/L1TauHighEtEffNum");
    bookEfficiencyHisto("L1TauEtaEff","EfficiencyHelpers/L1TauEtaEffNum");
    bookEfficiencyHisto("L1TauPhiEff","EfficiencyHelpers/L1TauPhiEffNum");

    bookEfficiencyHisto("L1JetEtEff","EfficiencyHelpers/L1JetEtEffNum");
    bookEfficiencyHisto("L1JetHighEtEff","EfficiencyHelpers/L1JetHighEtEffNum");
    bookEfficiencyHisto("L1JetEtaEff","EfficiencyHelpers/L1JetEtaEffNum");
    bookEfficiencyHisto("L1JetPhiEff","EfficiencyHelpers/L1JetPhiEffNum");
  }

  else if(type_ == Type::kPathSummary) {
    bookEfficiencyHisto("PathEfficiency", "helpers/PathTriggerBits", true);
  }
  store_ = nullptr;
}

void HLTTauDQMSummaryPlotter::SummaryPlotter::plot(DQMStore& store) {
  store_ = &store;

  //Path Summary 
  if ( type_ == Type::kPath ) {
    plotTriggerBitEfficiencyHistos("EventsPerFilter");

    plotEfficiencyHisto("L2TrigTauEtEff",  "helpers/L2TrigTauEtEffNum",  "helpers/L2TrigTauEtEffDenom");
    plotEfficiencyHisto("L2TrigTauHighEtEff",  "helpers/L2TrigTauHighEtEffNum",  "helpers/L2TrigTauHighEtEffDenom");
    plotEfficiencyHisto("L2TrigTauEtaEff", "helpers/L2TrigTauEtaEffNum", "helpers/L2TrigTauEtaEffDenom");
    plotEfficiencyHisto("L2TrigTauPhiEff", "helpers/L2TrigTauPhiEffNum", "helpers/L2TrigTauPhiEffDenom");

    plotEfficiencyHisto("L3TrigTauEtEff",  "helpers/L3TrigTauEtEffNum",  "helpers/L3TrigTauEtEffDenom");
    plotEfficiencyHisto("L3TrigTauHighEtEff",  "helpers/L3TrigTauHighEtEffNum",  "helpers/L3TrigTauHighEtEffDenom");
    plotEfficiencyHisto("L3TrigTauEtaEff", "helpers/L3TrigTauEtaEffNum", "helpers/L3TrigTauEtaEffDenom");
    plotEfficiencyHisto("L3TrigTauPhiEff", "helpers/L3TrigTauPhiEffNum", "helpers/L3TrigTauPhiEffDenom");
  }
        
  //L1 Summary
  else if ( type_ == Type::kL1 ) {
    plotEfficiencyHisto("L1TauEtEff","EfficiencyHelpers/L1TauEtEffNum","EfficiencyHelpers/L1TauEtEffDenom");
    plotEfficiencyHisto("L1TauHighEtEff","EfficiencyHelpers/L1TauHighEtEffNum","EfficiencyHelpers/L1TauHighEtEffDenom");
    plotEfficiencyHisto("L1TauEtaEff","EfficiencyHelpers/L1TauEtaEffNum","EfficiencyHelpers/L1TauEtaEffDenom");
    plotEfficiencyHisto("L1TauPhiEff","EfficiencyHelpers/L1TauPhiEffNum","EfficiencyHelpers/L1TauPhiEffDenom");
            
    plotEfficiencyHisto("L1JetEtEff","EfficiencyHelpers/L1JetEtEffNum","EfficiencyHelpers/L1JetEtEffDenom");
    plotEfficiencyHisto("L1JetHighEtEff","EfficiencyHelpers/L1JetHighEtEffNum","EfficiencyHelpers/L1JetHighEtEffDenom");
    plotEfficiencyHisto("L1JetEtaEff","EfficiencyHelpers/L1JetEtaEffNum","EfficiencyHelpers/L1JetEtaEffDenom");
    plotEfficiencyHisto("L1JetPhiEff","EfficiencyHelpers/L1JetPhiEffNum","EfficiencyHelpers/L1JetPhiEffDenom");
            
    plotEfficiencyHisto("L1ElectronEtEff","EfficiencyHelpers/L1ElectronEtEffNum","EfficiencyHelpers/L1ElectronEtEffDenom");
    plotEfficiencyHisto("L1ElectronEtaEff","EfficiencyHelpers/L1ElectronEtaEffNum","EfficiencyHelpers/L1ElectronEtaEffDenom");
    plotEfficiencyHisto("L1ElectronPhiEff","EfficiencyHelpers/L1ElectronPhiEffNum","EfficiencyHelpers/L1ElectronPhiEffDenom");
            
    plotEfficiencyHisto("L1MuonEtEff","EfficiencyHelpers/L1MuonEtEffNum","EfficiencyHelpers/L1MuonEtEffDenom");
    plotEfficiencyHisto("L1MuonEtaEff","EfficiencyHelpers/L1MuonEtaEffNum","EfficiencyHelpers/L1MuonEtaEffDenom");
    plotEfficiencyHisto("L1MuonPhiEff","EfficiencyHelpers/L1MuonPhiEffNum","EfficiencyHelpers/L1MuonPhiEffDenom");
  }

  else if(type_ == Type::kPathSummary) {
    plotEfficiencyHisto("PathEfficiency", "helpers/PathTriggerBits", "helpers/RefEvents");
  }
  store_ = nullptr;
}      

void HLTTauDQMSummaryPlotter::SummaryPlotter::bookEfficiencyHisto(const std::string& name, const std::string& hist1, bool copyLabels) {
    if ( store_->dirExists(folder_) ) {
        store_->setCurrentFolder(folder_);
        
        MonitorElement * effnum = store_->get(folder_+"/"+hist1);
        
        if ( effnum ) {            
            MonitorElement *tmp = store_->bookProfile(name,name,effnum->getTH1F()->GetNbinsX(),effnum->getTH1F()->GetXaxis()->GetXmin(),effnum->getTH1F()->GetXaxis()->GetXmax(),105,0,1.05);
            
            tmp->setTitle(effnum->getTitle());
            tmp->setAxisTitle(effnum->getAxisTitle(), 1); // X
            tmp->setAxisTitle("Efficiency", 2);
            if(copyLabels) {
              const TAxis *xaxis = effnum->getTH1F()->GetXaxis();
              for(int bin=1; bin <= effnum->getNbinsX(); ++bin) {
                tmp->setBinLabel(bin, xaxis->GetBinLabel(bin));
              }
            }
        }
    }
}

void HLTTauDQMSummaryPlotter::SummaryPlotter::plotEfficiencyHisto(std::string name, std::string hist1, std::string hist2 ) {
    if ( store_->dirExists(folder_) ) {
        store_->setCurrentFolder(folder_);
        
        MonitorElement * effnum = store_->get(folder_+"/"+hist1);
        MonitorElement * effdenom = store_->get(folder_+"/"+hist2);
        MonitorElement * eff = store_->get(folder_+"/"+name);
        
        if(effnum && effdenom && eff) {
          const TH1F *num = effnum->getTH1F();
          const TH1F *denom = effdenom->getTH1F();
          TProfile *prof = eff->getTProfile();
          for (int i = 1; i <= num->GetNbinsX(); ++i) {
            if(denom->GetBinContent(i) < num->GetBinContent(i)) {
              edm::LogError("HLTTauDQMOffline") << "Encountered denominator < numerator with efficiency plot " << name << " in folder " << folder_ << ", bin " << i << " numerator " << num->GetBinContent(i) << " denominator " << denom->GetBinContent(i);
              continue;
            }
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

void HLTTauDQMSummaryPlotter::SummaryPlotter::plotIntegratedEffHisto(std::string name, std::string refHisto, std::string evCount, int bin ) {
    if ( store_->dirExists(folder_) ) {
        store_->setCurrentFolder(folder_);
        
        MonitorElement * refH = store_->get(folder_+"/"+refHisto);
        MonitorElement * evC = store_->get(folder_+"/"+evCount);
        MonitorElement * eff = store_->get(folder_+"/"+name);
        
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

void HLTTauDQMSummaryPlotter::SummaryPlotter::bookTriggerBitEfficiencyHistos(std::string histo ) {
    if ( store_->dirExists(folder_) ) {
        store_->setCurrentFolder(folder_);
        
        MonitorElement * eff = store_->get(folder_+"/"+histo);
        
        if ( eff ) {
          //store_->bookProfile("EfficiencyRefInput","Efficiency with Matching",eff->getNbinsX()-1,0,eff->getNbinsX()-1,100,0,1);
          //store_->bookProfile("EfficiencyRefL1","Efficiency with Matching Ref to L1",eff->getNbinsX()-2,0,eff->getNbinsX()-2,100,0,1);
          MonitorElement *me_prev = store_->bookProfile("EfficiencyRefPrevious","Efficiency to Previous",eff->getNbinsX()-1,0,eff->getNbinsX()-1,100,0,1);
          me_prev->setAxisTitle("Efficiency", 2);
          const TAxis *xaxis = eff->getTH1F()->GetXaxis();
          for(int bin=1; bin < eff->getNbinsX(); ++bin) {
            me_prev->setBinLabel(bin, xaxis->GetBinLabel(bin));
          }
        }
    }
}

void HLTTauDQMSummaryPlotter::SummaryPlotter::plotTriggerBitEfficiencyHistos(std::string histo ) {
    if ( store_->dirExists(folder_) ) {
        store_->setCurrentFolder(folder_);
        MonitorElement * eff = store_->get(folder_+"/"+histo);
        //MonitorElement * effRefTruth = store_->get(folder_+"/EfficiencyRefInput");
        //MonitorElement * effRefL1 = store_->get(folder_+"/EfficiencyRefL1");
        MonitorElement * effRefPrevious = store_->get(folder_+"/EfficiencyRefPrevious");
        
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
              if(eff->getBinContent(i-1) < eff->getBinContent(i)) {
                edm::LogError("HLTTauDQMOffline") << "Encountered denominator < numerator with efficiency plot EfficiencyRefPrevious in folder " << folder_ << ", bin " << i << " numerator " << eff->getBinContent(i) << " denominator " << eff->getBinContent(i-1);
                continue;
              }
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
