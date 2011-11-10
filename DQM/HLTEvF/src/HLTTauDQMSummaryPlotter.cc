#include "DQM/HLTEvF/interface/HLTTauDQMSummaryPlotter.h"

HLTTauDQMSummaryPlotter::HLTTauDQMSummaryPlotter( const edm::ParameterSet& ps, std::string dqmBaseFolder ) {
    //Initialize Plotter
    name_ = "HLTTauDQMSummaryPlotter";
    
    //Process PSet
    try {
        triggerTag_      = ps.getUntrackedParameter<std::string>("DQMFolder");
        triggerTagAlias_ = ps.getUntrackedParameter<std::string>("Alias","");
        type_            = ps.getUntrackedParameter<std::string>("ConfigType");
        dqmBaseFolder_   = dqmBaseFolder;
        validity_        = true;
    } catch ( cms::Exception &e ) {
        edm::LogWarning("HLTTauDQMSummaryPlotter::HLTTauDQMSummaryPlotter") << e.what() << std::endl;
        validity_ = false;
        return;
    }
    
    if (store_) {
        //Path Summary 
        if ( type_ == "Path" ) {
            bookTriggerBitEfficiencyHistos(triggerTag(),"MatchedTriggerBits");
        }
        
        //Lite Path Summary 
        if ( type_ == "LitePath" ) {
            bookEfficiencyHisto(triggerTag(),"PathEfficiency","MatchedPathTriggerBits");
            bookEfficiencyHisto(triggerTag(),"TrigTauEtEff","EfficiencyHelpers/TrigTauEtEffNum");
            bookEfficiencyHisto(triggerTag(),"TrigTauEtaEff","EfficiencyHelpers/TrigTauEtaEffNum");
            bookEfficiencyHisto(triggerTag(),"TrigTauPhiEff","EfficiencyHelpers/TrigTauPhiEffNum");
        }
        
        //L1 Summary
        if ( type_ == "L1" ) {
            bookEfficiencyHisto(triggerTag(),"L1TauEtEff","EfficiencyHelpers/L1TauEtEffNum");
            bookEfficiencyHisto(triggerTag(),"L1TauEtaEff","EfficiencyHelpers/L1TauEtaEffNum");
            bookEfficiencyHisto(triggerTag(),"L1TauPhiEff","EfficiencyHelpers/L1TauPhiEffNum");
            
            bookEfficiencyHisto(triggerTag(),"L1JetEtEff","EfficiencyHelpers/L1JetEtEffNum");
            bookEfficiencyHisto(triggerTag(),"L1JetEtaEff","EfficiencyHelpers/L1JetEtaEffNum");
            bookEfficiencyHisto(triggerTag(),"L1JetPhiEff","EfficiencyHelpers/L1JetPhiEffNum");
            
            bookEfficiencyHisto(triggerTag(),"L1SingleTauEff","L1LeadTauEt");
            bookEfficiencyHisto(triggerTag(),"L1DoubleTauEff","L1SecondTauEt");
        }
        
        //L2 Summary
        if ( type_ == "Calo" ) {
            bookEfficiencyHisto(triggerTag(),"L2RecoTauEtEff","EfficiencyHelpers/L2RecoTauEtEffNum");
            bookEfficiencyHisto(triggerTag(),"L2RecoTauEtaEff","EfficiencyHelpers/L2RecoTauEtaEffNum");
            bookEfficiencyHisto(triggerTag(),"L2RecoTauPhiEff","EfficiencyHelpers/L2RecoTauPhiEffNum");
            
            bookEfficiencyHisto(triggerTag(),"L2IsoTauEtEff","EfficiencyHelpers/L2IsoTauEtEffNum");
            bookEfficiencyHisto(triggerTag(),"L2IsoTauEtaEff","EfficiencyHelpers/L2IsoTauEtaEffNum");
            bookEfficiencyHisto(triggerTag(),"L2IsoTauPhiEff","EfficiencyHelpers/L2IsoTauPhiEffNum");
        }
        
        //L25/3 Summary
        if ( type_ == "Track" ) {
            bookEfficiencyHisto(triggerTag(),"L25TauEtEff","L25TauEtEffNum");
            bookEfficiencyHisto(triggerTag(),"L25TauEtaEff","L25TauEtaEffNum");
            bookEfficiencyHisto(triggerTag(),"L25TauPhiEff","L25TauPhiEffNum");
            bookEfficiencyHisto(triggerTag(),"L3TauEtEff","L3TauEtEffNum");
            bookEfficiencyHisto(triggerTag(),"L3TauEtaEff","L3TauEtaEffNum");
            bookEfficiencyHisto(triggerTag(),"L3TauPhiEff","L3TauPhiEffNum");
        }
    }
}

HLTTauDQMSummaryPlotter::~HLTTauDQMSummaryPlotter() {
}

void HLTTauDQMSummaryPlotter::plot() {
    if (store_) {
        //Path Summary 
        if ( type_ == "Path" ) {
            plotTriggerBitEfficiencyHistos(triggerTag(),"MatchedTriggerBits");
        }
        
        //Lite Path Summary 
        if ( type_ == "LitePath" ) {
            plotEfficiencyHisto(triggerTag(),"PathEfficiency","MatchedPathTriggerBits","RefEvents");
            plotEfficiencyHisto(triggerTag(),"TrigTauEtEff","EfficiencyHelpers/TrigTauEtEffNum","EfficiencyHelpers/TrigTauEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"TrigTauEtaEff","EfficiencyHelpers/TrigTauEtaEffNum","EfficiencyHelpers/TrigTauEtaEffDenom");
            plotEfficiencyHisto(triggerTag(),"TrigTauPhiEff","EfficiencyHelpers/TrigTauPhiEffNum","EfficiencyHelpers/TrigTauPhiEffDenom");
        }
        
        //L1 Summary
        if ( type_ == "L1" ) {
            plotEfficiencyHisto(triggerTag(),"L1TauEtEff","EfficiencyHelpers/L1TauEtEffNum","EfficiencyHelpers/L1TauEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1TauEtaEff","EfficiencyHelpers/L1TauEtaEffNum","EfficiencyHelpers/L1TauEtaEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1TauPhiEff","EfficiencyHelpers/L1TauPhiEffNum","EfficiencyHelpers/L1TauPhiEffDenom");
            
            plotEfficiencyHisto(triggerTag(),"L1JetEtEff","EfficiencyHelpers/L1JetEtEffNum","EfficiencyHelpers/L1JetEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1JetEtaEff","EfficiencyHelpers/L1JetEtaEffNum","EfficiencyHelpers/L1JetEtaEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1JetPhiEff","EfficiencyHelpers/L1JetPhiEffNum","EfficiencyHelpers/L1JetPhiEffDenom");
            
            plotEfficiencyHisto(triggerTag(),"L1ElectronEtEff","EfficiencyHelpers/L1ElectronEtEffNum","EfficiencyHelpers/L1ElectronEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1ElectronEtaEff","EfficiencyHelpers/L1ElectronEtaEffNum","EfficiencyHelpers/L1ElectronEtaEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1ElectronPhiEff","EfficiencyHelpers/L1ElectronPhiEffNum","EfficiencyHelpers/L1ElectronPhiEffDenom");
            
            plotEfficiencyHisto(triggerTag(),"L1MuonEtEff","EfficiencyHelpers/L1MuonEtEffNum","EfficiencyHelpers/L1MuonEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1MuonEtaEff","EfficiencyHelpers/L1MuonEtaEffNum","EfficiencyHelpers/L1MuonEtaEffDenom");
            plotEfficiencyHisto(triggerTag(),"L1MuonPhiEff","EfficiencyHelpers/L1MuonPhiEffNum","EfficiencyHelpers/L1MuonPhiEffDenom");
            
            plotIntegratedEffHisto(triggerTag(),"L1SingleTauEff","L1LeadTauEt","InputEvents",1);
            plotIntegratedEffHisto(triggerTag(),"L1DoubleTauEff","L1SecondTauEt","InputEvents",2);
        }
        
        //L2 Summary
        if ( type_ == "Calo" ) {
            plotEfficiencyHisto(triggerTag(),"L2RecoTauEtEff","EfficiencyHelpers/L2RecoTauEtEffNum","EfficiencyHelpers/L2RecoTauEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L2RecoTauEtaEff","EfficiencyHelpers/L2RecoTauEtaEffNum","EfficiencyHelpers/L2RecoTauEtaEffDenom");
            plotEfficiencyHisto(triggerTag(),"L2RecoTauPhiEff","EfficiencyHelpers/L2RecoTauPhiEffNum","EfficiencyHelpers/L2RecoTauPhiEffDenom");
            
            plotEfficiencyHisto(triggerTag(),"L2IsoTauEtEff","EfficiencyHelpers/L2IsoTauEtEffNum","EfficiencyHelpers/L2IsoTauEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L2IsoTauEtaEff","EfficiencyHelpers/L2IsoTauEtaEffNum","EfficiencyHelpers/L2IsoTauEtaEffDenom");
            plotEfficiencyHisto(triggerTag(),"L2IsoTauPhiEff","EfficiencyHelpers/L2IsoTauPhiEffNum","EfficiencyHelpers/L2IsoTauPhiEffDenom");
        }
        
        //L25/3 Summary
        if ( type_ == "Track" ) {
            plotEfficiencyHisto(triggerTag(),"L25TauEtEff","L25TauEtEffNum","L25TauEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L25TauEtaEff","L25TauEtaEffNum","L25TauEtaEffDenom");
            plotEfficiencyHisto(triggerTag(),"L25TauPhiEff","L25TauPhiEffNum","L25TauPhiEffDenom");
            plotEfficiencyHisto(triggerTag(),"L3TauEtEff","L3TauEtEffNum","L3TauEtEffDenom");
            plotEfficiencyHisto(triggerTag(),"L3TauEtaEff","L3TauEtaEffNum","L3TauEtaEffDenom");
            plotEfficiencyHisto(triggerTag(),"L3TauPhiEff","L3TauPhiEffNum","L3TauPhiEffDenom");
        }
    }
}      

void HLTTauDQMSummaryPlotter::bookEfficiencyHisto( std::string folder, std::string name, std::string hist1 ) {
    if ( store_->dirExists(folder) ) {
        store_->setCurrentFolder(folder);
        
        MonitorElement * effnum = store_->get(folder+"/"+hist1);
        
        if ( effnum ) {            
            MonitorElement *tmp = store_->bookProfile(name,name,effnum->getTH1F()->GetNbinsX(),effnum->getTH1F()->GetXaxis()->GetXmin(),effnum->getTH1F()->GetXaxis()->GetXmax(),105,0,1.05);
            
            tmp->setTitle(name);
        }
    }
}

void HLTTauDQMSummaryPlotter::plotEfficiencyHisto( std::string folder, std::string name, std::string hist1, std::string hist2 ) {
    if ( store_->dirExists(folder) ) {
        store_->setCurrentFolder(folder);
        
        MonitorElement * effnum = store_->get(folder+"/"+hist1);
        MonitorElement * effdenom = store_->get(folder+"/"+hist2);
        MonitorElement * eff = store_->get(folder+"/"+name);
        
        if ( effnum && effdenom && eff ) {            
            for ( int i = 1; i <= effnum->getTH1F()->GetNbinsX(); ++i ) {
                double efficiency = calcEfficiency(effnum->getTH1F()->GetBinContent(i),effdenom->getTH1F()->GetBinContent(i)).first;
                double err = calcEfficiency(effnum->getTH1F()->GetBinContent(i),effdenom->getTH1F()->GetBinContent(i)).second;
                eff->getTProfile()->SetBinContent(i,efficiency);
                eff->getTProfile()->SetBinEntries(i,1);
                eff->getTProfile()->SetBinError(i,sqrt(efficiency*efficiency+err*err));
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
            store_->bookProfile("EfficiencyRefInput","Efficiency with Matching",eff->getNbinsX()-1,0,eff->getNbinsX()-1,100,0,1);
            store_->bookProfile("EfficiencyRefL1","Efficiency with Matching Ref to L1",eff->getNbinsX()-2,0,eff->getNbinsX()-2,100,0,1);
            store_->bookProfile("EfficiencyRefPrevious","Efficiency with Matching Ref to Previous",eff->getNbinsX()-1,0,eff->getNbinsX()-1,100,0,1);
        }
    }
}

void HLTTauDQMSummaryPlotter::plotTriggerBitEfficiencyHistos( std::string folder, std::string histo ) {
    if ( store_->dirExists(folder) ) {
        store_->setCurrentFolder(folder);
        MonitorElement * eff = store_->get(folder+"/"+histo);
        MonitorElement * effRefTruth = store_->get(folder+"/EfficiencyRefInput");
        MonitorElement * effRefL1 = store_->get(folder+"/EfficiencyRefL1");
        MonitorElement * effRefPrevious = store_->get(folder+"/EfficiencyRefPrevious");
        
        if ( eff && effRefTruth && effRefL1 && effRefPrevious ) {
            //Calculate efficiencies with ref to matched objects
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
            //Calculate efficiencies with ref to previous
            for ( int i = 2; i <= eff->getNbinsX(); ++i ) {
                double efficiency = calcEfficiency(eff->getBinContent(i),eff->getBinContent(i-1)).first;
                double err = calcEfficiency(eff->getBinContent(i),eff->getBinContent(i-1)).second;
                
                effRefPrevious->getTProfile()->SetBinContent(i-1,efficiency);
                effRefPrevious->getTProfile()->SetBinEntries(i-1,1);
                effRefPrevious->getTProfile()->SetBinError(i-1,sqrt(efficiency*efficiency+err*err));
                effRefPrevious->setBinLabel(i-1,eff->getTH1F()->GetXaxis()->GetBinLabel(i));
            }
        }
    }
}

std::pair<double,double> HLTTauDQMSummaryPlotter::calcEfficiency( float num, float denom ) {
    if ( denom != 0.0 ) {
        return std::pair<double,double>(num/denom,sqrt(num/denom*(1.0-num/denom)/denom));
    }
    return std::pair<double,double>(0.0,0.0);
}
