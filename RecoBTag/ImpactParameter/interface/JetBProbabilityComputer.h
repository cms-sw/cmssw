#ifndef ImpactParameter_JetBProbabilityComputer_h
#define ImpactParameter_JetBProbabilityComputer_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "Math/GenVector/VectorUtil.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include <algorithm>
#include <iostream>

class JetBProbabilityComputer : public JetTagComputer
{
 public:
  JetBProbabilityComputer(const edm::ParameterSet  & parameters )
  { 
     m_ipType           = parameters.getParameter<int>("impactParameterType");
     m_minTrackProb     = parameters.getParameter<double>("minimumProbability");
     m_deltaR           = parameters.getParameter<double>("deltaR");
     m_trackSign        = parameters.getParameter<int>("trackIpSign");
     m_nbTracks         = parameters.getParameter<unsigned int>("numberOfBTracks");
     m_cutMaxDecayLen   = parameters.getParameter<double>("maximumDecayLength");
     m_cutMaxDistToAxis = parameters.getParameter<double>("maximumDistanceToJetAxis");
 
     //
     // access track quality class; "any" takes everything
     //
     std::string trackQualityType = parameters.getParameter<std::string>("trackQualityClass"); //used
     m_trackQuality =  reco::TrackBase::qualityByName(trackQualityType);
     m_useAllQualities = false;
     if (trackQualityType == "any" || 
	 trackQualityType == "Any" || 
	 trackQualityType == "ANY" ) m_useAllQualities = true;

     useVariableJTA_    = parameters.getParameter<bool>("useVariableJTA");
     if(useVariableJTA_) 
       varJTApars = {
	 parameters.getParameter<double>("a_dR"),
	 parameters.getParameter<double>("b_dR"),
	 parameters.getParameter<double>("a_pT"),
	 parameters.getParameter<double>("b_pT"),
	 parameters.getParameter<double>("min_pT"),  
	 parameters.getParameter<double>("max_pT"),
	 parameters.getParameter<double>("min_pT_dRcut"),  
	 parameters.getParameter<double>("max_pT_dRcut"),
	 parameters.getParameter<double>("max_pT_trackPTcut") };
     
     uses("ipTagInfos");
  }
 
  float discriminator(const TagInfoHelper & ti) const 
   {
      const reco::TrackIPTagInfo & tkip = ti.get<reco::TrackIPTagInfo>();
      const edm::RefVector<reco::TrackCollection> & tracks(tkip.selectedTracks());
      const std::vector<float> & allProbabilities((tkip.probabilities(m_ipType)));
      const std::vector<reco::TrackIPTagInfo::TrackIPData> & impactParameters((tkip.impactParameterData()));

      if(tkip.primaryVertex().isNull()) return 0;

      GlobalPoint pv(tkip.primaryVertex()->position().x(),tkip.primaryVertex()->position().y(),tkip.primaryVertex()->position().z());

      std::vector<float> probabilities;
      std::vector<float> probabilitiesB;
      int i=0;
      for(std::vector<float>::const_iterator it = allProbabilities.begin(); it!=allProbabilities.end(); ++it, i++)
       {
        if (fabs(impactParameters[i].distanceToJetAxis.value()) < m_cutMaxDistToAxis  &&        // distance to JetAxis
            (impactParameters[i].closestToJetAxis - pv).mag() < m_cutMaxDecayLen  &&      // max decay len
            (m_useAllQualities  == true || (*tracks[i]).quality(m_trackQuality)) // use selected track qualities
           )
         {
    // Use only positive(or negative) tracks for B
           float p=fabs(*it);
	   if (useVariableJTA_) {
	     if (tkip.variableJTA( varJTApars  )[i]) {
	       if(m_trackSign>0 || *it <0 ) probabilities.push_back(p); //Use all tracks for positive tagger and only negative for negative tagger      
	       if(m_trackSign>0 && *it >=0){probabilitiesB.push_back(*it);} //Use only positive tracks for positive tagger
	       if(m_trackSign<0 && *it <=0){probabilitiesB.push_back(- *it);} //Use only negative tracks for negative tagger 
	     }
	   }
	   else
	     if( m_deltaR < 0 || ROOT::Math::VectorUtil::DeltaR((*tkip.jet()).p4().Vect(), (*tracks[i]).momentum()) < m_deltaR)
	       {
		 //if(m_trackSign>0 || *it >0 ) probabilities.push_back(p); //Use all tracks for positive tagger and only negative for negative tagger
		 if(m_trackSign>0 || *it <0 ) probabilities.push_back(p); //Use all tracks for positive tagger and only negative for negative tagger
		 
		 if(m_trackSign>0 && *it >=0){probabilitiesB.push_back(*it);} //Use only positive tracks for positive tagger
		 if(m_trackSign<0 && *it <=0){probabilitiesB.push_back(- *it);} //Use only negative tracks for negative tagger 
	       }
         }
       }

      float all = jetProbability(probabilities); 
      std::sort(probabilitiesB.begin(), probabilitiesB.end());
      if(probabilitiesB.size() > m_nbTracks )  probabilitiesB.resize(m_nbTracks);
      float b = jetProbability(probabilitiesB);
        
      return -log(b)/4-log(all)/4; ///log(all);
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
 bool useVariableJTA_;
 reco::TrackIPTagInfo::variableJTAParameters varJTApars;
   double m_minTrackProb;
   int m_ipType;
   double m_deltaR;
   int m_trackSign;
   unsigned int m_nbTracks;
   double  m_cutMaxDecayLen;
   double m_cutMaxDistToAxis;
   reco::TrackBase::TrackQuality   m_trackQuality;
   bool m_useAllQualities;

};

#endif // ImpactParameter_JetBProbabilityComputer_h
