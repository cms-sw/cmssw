#ifndef RecoTrackerDeDx_ASmirnovDeDxDiscriminator_h
#define RecoTrackerDeDx_ASmirnovDeDxDiscriminator_h

#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"

class ASmirnovDeDxDiscriminator: public BaseDeDxEstimator
{
public: 
 ASmirnovDeDxDiscriminator(const edm::ParameterSet& iConfig){
   meVperADCStrip      = iConfig.getParameter<double>("MeVperADCStrip"); //currently needed until the map on the database are redone
   Reccord             = iConfig.getParameter<std::string>  ("Reccord");
   ProbabilityMode     = iConfig.getParameter<std::string>  ("ProbabilityMode");
   Prob_ChargePath     = nullptr;
 }

 void beginRun(edm::Run const& run, const edm::EventSetup& iSetup) override{
    DeDxTools::buildDiscrimMap(run, iSetup, Reccord,  ProbabilityMode, Prob_ChargePath);
 }

 std::pair<float,float> dedx(const reco::DeDxHitCollection& Hits) override{
    std::vector<float> vect_probs;
    for(size_t i = 0; i< Hits.size(); i ++){
       float path   = Hits[i].pathLength() * 10.0;  //x10 in order to be compatible with the map content
       float charge = Hits[i].charge() / (10.0*meVperADCStrip); // 10/meVperADCStrip in order to be compatible with the map content in ADC/mm instead of MeV/cm

       int    BinX  = Prob_ChargePath->GetXaxis()->FindBin(Hits[i].momentum());
       int    BinY  = Prob_ChargePath->GetYaxis()->FindBin(path);
       int    BinZ  = Prob_ChargePath->GetZaxis()->FindBin(charge);
       float  prob  = Prob_ChargePath->GetBinContent(BinX,BinY,BinZ);
       if(prob>=0)vect_probs.push_back(prob);
    }

    size_t size  = vect_probs.size();
    if(size<=0) return std::make_pair( -1 , -1);     
    std::sort(vect_probs.begin(), vect_probs.end(), std::less<float>() );
    float TotalProb = 1.0/(12*size);
    for(size_t i=1;i<=size;i++){
       TotalProb += vect_probs[i-1] * pow(vect_probs[i-1] - ((2.0*i-1.0)/(2.0*size)),2);
    }
    TotalProb *= (3.0/size);
    return std::make_pair( TotalProb , -1);
 }

private:
  float             meVperADCStrip;
  std::string       Reccord;
  std::string       ProbabilityMode;
  TH3F*             Prob_ChargePath;
};

#endif
