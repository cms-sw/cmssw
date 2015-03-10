#include <vector>
#include <iostream>
#include <algorithm>
#include <boost/tokenizer.hpp>
#include "TTree.h"

#include "CalibCalorimetry/EcalTiming/interface/EcalTimeTreeContent.h"

// ****************************************************************
class TimingEvent
{
  public:
    float amplitude;
    float time;
    float chi2;
    float sigmaTime;
    float expectedPrecision;

    TimingEvent() :
      amplitude(-1),
      time(-1),
      chi2(-1),
      sigmaTime(-1),
      expectedPrecision(-1)
    {
    }
    
    TimingEvent(float amp,float t,float sigmaT, bool ee) : 
      amplitude(amp), 
      time(t), 
      chi2(-1),
      sigmaTime(sigmaT)
    {
      if(ee)
        expectedPrecision = 33/(amplitude/2.0);
      else
        expectedPrecision = 33/(amplitude/1.2);
    }

};

// ****************************************************************
class CrystalCalibration
{
  public:
    float mean;
    float meanE; // error on the mean
    float rms;
    float stdDev;
    float totalChi2;
    std::vector<TimingEvent> timingEvents;
    std::vector<TimingEvent>::iterator maxChi2Itr;

    CrystalCalibration() :
      mean(-1),
      meanE(-1),
      rms(-1),
      totalChi2(-1),
      useWeightedMean(true)
    {
    }
    
    CrystalCalibration(bool weightMean) :
      mean(-1),
      meanE(-1),
      rms(-1),
      totalChi2(-1),
      useWeightedMean(weightMean)
    {
    }
      
    CrystalCalibration(float m, float me, float r, float tc, std::vector<TimingEvent> te) :
      mean(m),
      meanE(me),
      rms(r),
      totalChi2(tc)
    {
      timingEvents = te;
    }

    CrystalCalibration(float m, float me, float r, float tc, std::vector<TimingEvent> te, bool wm) :
      mean(m),
      meanE(me),
      rms(r),
      totalChi2(tc),
      useWeightedMean(wm)
    {
      timingEvents = te;
    }

    bool insertEvent(float amp, float t, float sigmaT, bool ee)
    {
      if(sigmaT > 0) // throw away events with zero or negative errors
      {
        timingEvents.push_back(TimingEvent(amp,t,sigmaT,ee));
        updateChi2();
        return true;
      }
      else
        return false;
    }

    bool insertEvent(TimingEvent te)
    {
      if(te.sigmaTime > 0)
      {
        timingEvents.push_back(te);
        updateChi2();
        return true;
      }
      else
        return false;
    }

    int filterOutliers(float threshold = 0.5)
    {
      int numPointsErased = 0;
      while(timingEvents.size() > 4)
      {
        updateChi2();
        float oldMean = mean;
        // Erase largest chi2 event
        TimingEvent toRemove = *maxChi2Itr;
        timingEvents.erase(maxChi2Itr);
        //Calculate new mean/error
        updateChi2();
        //Compare to old mean and break if |(newMean-oldMean)| < newSigma
        //TODO: study acceptance threshold
        if(fabs(mean-oldMean) < threshold*meanE)
        {
          insertEvent(toRemove);
          break;
        }
        else
        {
          numPointsErased++;
        }
      }
      return numPointsErased;
    }

  private:
    bool useWeightedMean;

    void updateChi2() // update individual, total, maxChi2s
    {
      if(useWeightedMean)
        updateChi2Weighted();
      else
        updateChi2Unweighted();
    }

    void updateChi2Weighted()
    {
      updateMeanWeighted();
      float chi2 = 0;
      maxChi2Itr = timingEvents.begin();
      for(std::vector<TimingEvent>::iterator itr = timingEvents.begin();
          itr != timingEvents.end(); ++itr)
      {
        float singleChi = (itr->time-mean)/itr->sigmaTime;
        itr->chi2 = singleChi*singleChi;
        chi2+=singleChi*singleChi;
        if(itr->chi2 > maxChi2Itr->chi2)
          maxChi2Itr = itr;
      }
      totalChi2 = chi2;
    }

    void updateChi2Unweighted()
    {
      updateMeanUnweighted();
      float chi2 = 0;
      maxChi2Itr = timingEvents.begin();
      for(std::vector<TimingEvent>::iterator itr = timingEvents.begin();
          itr != timingEvents.end(); ++itr)
      {
        float singleChi = (itr->time-mean);
        itr->chi2 = singleChi*singleChi;
        chi2+=singleChi*singleChi;
        if(itr->chi2 > maxChi2Itr->chi2)
          maxChi2Itr = itr;
      }
      totalChi2 = chi2;
    }

    void updateMeanWeighted()
    {
      float meanTmp = 0;
      float mean2Tmp = 0;
      float sigmaTmp = 0;
      for(std::vector<TimingEvent>::const_iterator itr = timingEvents.begin();
          itr != timingEvents.end(); ++itr)
      {
        float sigmaT2 = itr->sigmaTime;
        sigmaT2*=sigmaT2;
        sigmaTmp+=1/(sigmaT2);
        meanTmp+=(itr->time)/(sigmaT2);
        mean2Tmp+=((itr->time)*(itr->time))/(sigmaT2);
      }
      meanE = sqrt(1/sigmaTmp);
      mean = meanTmp/sigmaTmp;
      rms = sqrt(mean2Tmp/sigmaTmp);
      stdDev = sqrt(rms*rms-mean*mean);
    }

    void updateMeanUnweighted()
    {
      float meanTmp = 0;
      float mean2Tmp = 0;
      for(std::vector<TimingEvent>::const_iterator itr = timingEvents.begin();
          itr != timingEvents.end(); ++itr)
      {
        meanTmp+=itr->time;
        mean2Tmp+=(itr->time)*(itr->time);
      }
      mean = meanTmp/timingEvents.size();
      rms = sqrt(mean2Tmp/timingEvents.size());
      stdDev = sqrt(rms*rms-mean*mean);
      meanE = stdDev/sqrt(timingEvents.size()); // stdDev/sqrt(n)
    }
    
};


// ****************************************************************
//
std::vector<std::string> split(std::string msg, std::string separator)
{
  boost::char_separator<char> sep(separator.c_str());
  boost::tokenizer<boost::char_separator<char> > tok(msg, sep );
  std::vector<std::string> token ;
  for ( boost::tokenizer<boost::char_separator<char> >::const_iterator i = tok.begin(); i != tok.end(); ++i ) {
    token.push_back(std::string(*i)) ;
  }
  return token ;
}


//
void genIncludeExcludeVectors(std::string optionString,
    std::vector<std::vector<double> >& includeVector,
    std::vector<std::vector<double> >& excludeVector)
{
  std::vector<std::string> rangeStringVector;
  std::vector<double> rangeIntVector;

  if(optionString != "-1"){
    std::vector<std::string> stringVector = split(optionString,",") ;

    for (uint i=0 ; i<stringVector.size() ; i++) {
      bool exclude = false;

      if(stringVector[i].at(0)=='x'){
        exclude = true;
        stringVector[i].erase(0,1);
      }
      rangeStringVector = split(stringVector[i],"-") ;

      rangeIntVector.clear();
      for(uint j=0; j<rangeStringVector.size();j++) {
        rangeIntVector.push_back(atof(rangeStringVector[j].c_str()));
      }
      if(exclude) excludeVector.push_back(rangeIntVector);
      else includeVector.push_back(rangeIntVector);

    }
  }
}


//
bool includeEvent(double eventParameter,
    std::vector<std::vector<double> > includeVector,
    std::vector<std::vector<double> > excludeVector)
{
  bool keepEvent = false;
  if(includeVector.size()==0) keepEvent = true;
  for(uint i=0; i!=includeVector.size();++i){
    if(includeVector[i].size()==1 && eventParameter==includeVector[i][0])
      keepEvent=true;
    else if(includeVector[i].size()==2 && (eventParameter>=includeVector[i][0] && eventParameter<=includeVector[i][1]))
      keepEvent=true;
  }
  if(!keepEvent) // if it's not in our include list, skip it
    return false;

  keepEvent = true;
  for(uint i=0; i!=excludeVector.size();++i){
    if(excludeVector[i].size()==1 && eventParameter==excludeVector[i][0])
      keepEvent=false;
    else if(excludeVector[i].size()==2 && (eventParameter>=excludeVector[i][0] && eventParameter<=excludeVector[i][1]))
      keepEvent=false;
  }

  return keepEvent; // if someone includes and excludes, exclusion will overrule

}


//
bool includeEvent(int* triggers,
    int numTriggers,
    std::vector<std::vector<double> > includeVector,
    std::vector<std::vector<double> > excludeVector)
{
  bool keepEvent = false;
  if(includeVector.size()==0) keepEvent = true;
  for (int ti = 0; ti < numTriggers; ++ti) {
    for(uint i=0; i!=includeVector.size();++i){
      if(includeVector[i].size()==1 && triggers[ti]==includeVector[i][0]) keepEvent=true;
      else if(includeVector[i].size()==2 && (triggers[ti]>=includeVector[i][0] && triggers[ti]<=includeVector[i][1])) keepEvent=true;
    }
  }
  if(!keepEvent)
    return false;

  keepEvent = true;
  for (int ti = 0; ti < numTriggers; ++ti) {
    for(uint i=0; i!=excludeVector.size();++i){
      if(excludeVector[i].size()==1 && triggers[ti]==excludeVector[i][0]) keepEvent=false;
      else if(excludeVector[i].size()==2 && (triggers[ti]>=excludeVector[i][0] && triggers[ti]<=excludeVector[i][1])) keepEvent=false;
    }
  }

  return keepEvent;
}
