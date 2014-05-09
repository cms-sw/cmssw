#include "HLTriggerOffline/Higgs/interface/HLTHiggsHarvesting.h"

HiggsHarvesting::HiggsHarvesting(const edm::ParameterSet& iPSet)
{
    
    analysisName = iPSet.getUntrackedParameter<std::string>("analysisName");
    

}

HiggsHarvesting::~HiggsHarvesting() {}

void HiggsHarvesting::beginJob()
{
    return;
}

void HiggsHarvesting::dqmEndJob(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter)
{

    
    //define type of sources :
    std::vector<std::string> sources(2);
    sources[0] = "gen";
    sources[1] = "rec";
    for(size_t i = 0; i < sources.size(); i++) {
        // monitoring element numerator and denominator histogram
        MonitorElement *meN =
        iGetter.get("HLT/Higgs/"+analysisName+"/SummaryPaths_"+analysisName+"_"+sources[i]+"_passingHLT");
        MonitorElement *meD =
        iGetter.get("HLT/Higgs/"+analysisName+"/SummaryPaths_"+analysisName+"_"+sources[i]);
        
        if (meN && meD) {
            // get the numerator and denominator histogram
            TH1F *numerator = meN->getTH1F();
            TH1F *denominator = meD->getTH1F();
            
            // set the current directory
            iBooker.setCurrentFolder("HLT/Higgs/"+analysisName);
            
            // booked the new histogram to contain the results
            TString nameEffHisto = "efficiencySummary_"+sources[i];
            TH1F *efficiencySummary = (TH1F*) numerator->Clone(nameEffHisto);
            std::string histoTitle = "efficiency of paths used in " + analysisName;
            efficiencySummary->SetTitle(histoTitle.c_str());
            MonitorElement *me = iBooker.book1D(nameEffHisto, efficiencySummary );
            
                // Calculate the efficiency
            me->getTH1F()->Divide(numerator, denominator, 1., 1., "B");
        
        } else {
            std::cout << "Monitor elements don't exist" << std::endl;
        }
    }
    
    return;
}

void HiggsHarvesting::beginRun(const edm::Run& iRun,
                                  const edm::EventSetup& iSetup)
{
    return;
}

void HiggsHarvesting::endRun(const edm::Run& iRun, 
                                const edm::EventSetup& iSetup)
{
    return;
}

