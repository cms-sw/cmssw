#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "Math/GenVector/VectorUtil.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include <algorithm>
#include <iostream>

class JetBProbabilityComputer : public JetTagComputer
{
 public:
  JetBProbabilityComputer(const edm::ParameterSet  & parameters )
  { 
     m_ipType           = parameters.getParameter<int>("impactParamterType");
     m_minTrackProb     = parameters.getParameter<double>("minimumProbability");
     m_deltaR           = parameters.getParameter<double>("deltaR");

  }
 
  float discriminator(const reco::BaseTagInfo & ti) const 
   {
      const reco::TrackIPTagInfo * tkip = dynamic_cast<const reco::TrackIPTagInfo *>(&ti);
      if(tkip!=0)  {
          const edm::RefVector<reco::TrackCollection> & tracks(tkip->selectedTracks());
          const std::vector<float> & allProbabilities((tkip->probabilities(m_ipType)));
          std::vector<float> probabilities;
          std::vector<float> probabilitiesB;
          int i=0;
          for(std::vector<float>::const_iterator it = allProbabilities.begin(); it!=allProbabilities.end(); ++it, i++)
           {
              // Use only positive tracks for B
              if (*it >=0){probabilitiesB.push_back(*it);}
               
               float p=fabs(*it);
               double delta = -2;
               if(m_deltaR > 0)   delta  = ROOT::Math::VectorUtil::DeltaR((*tkip->jet()).p4().Vect(), (*tracks[i]).momentum());
               if(delta < m_deltaR || m_deltaR < 0)
                   probabilities.push_back(p);
           }

          float all = jetProbability(probabilities); 
          std::sort(probabilitiesB.begin(), probabilitiesB.end());
          if(probabilitiesB.size() > 4 )  probabilitiesB.resize(4);
          float b = jetProbability(probabilitiesB);
        
	  return -log(b)/4-log(all)/4; ///log(all);
          }
        else { 
                 //FIXME: report an  error?
                return 0;
      }
   }

double jetProbability( const std::vector<float> & v ) const
{
   int ngoodtracks=v.size();
   double SumJet=0.;

  for(std::vector<float>::const_iterator q = v.begin(); q != v.end(); q++){
    SumJet+=(*q>m_minTrackProb)?log(*q):log(m_minTrackProb);
  }

  double ProbJet;
  double Loginvlog=0;

  if(SumJet<0.){
    if(ngoodtracks>=2){
      Loginvlog=log(-SumJet);
    }
    double Prob=1.;
    double lfact=1.;
    for(int l=1; l!=ngoodtracks; l++){
       lfact*=l;
      Prob+=exp(l*Loginvlog-log(1.*lfact));
    }
    double LogProb=log(Prob);
    ProbJet=
      std::min(exp(std::max(LogProb+SumJet,-30.)),1.);
  }else{
    ProbJet=1.;
  }
  if(ProbJet>1)
   std::cout << "ProbJet too high: "  << ProbJet << std::endl;

  //double LogProbJet=-log(ProbJet);
  //  //return 1.-ProbJet;
      return ProbJet;
  }
 private:
   double m_minTrackProb;
   int m_ipType;
   double m_deltaR;
   int m_trackSign;
};
