#include "CalibCalorimetry/EcalTiming/interface/EcalTimingEvent.h"

#include <vector>

class EcalCrystalTimingCalibration
{
  public:
    float mean;
    float meanE; // error on the mean
    float rms;
    float stdDev;
    float totalChi2;
    std::vector<EcalTimingEvent> timingEvents;
    std::vector<EcalTimingEvent>::iterator maxChi2Itr;

    EcalCrystalTimingCalibration() :
      mean(-1),
      meanE(-1),
      rms(-1),
      totalChi2(-1),
      useWeightedMean(true)
    {
    }
    
    EcalCrystalTimingCalibration(bool weightMean) :
      mean(-1),
      meanE(-1),
      rms(-1),
      totalChi2(-1),
      useWeightedMean(weightMean)
    {
    }
      
    EcalCrystalTimingCalibration(float m, float me, float r, float tc, std::vector<EcalTimingEvent> te) :
      mean(m),
      meanE(me),
      rms(r),
      totalChi2(tc)
    {
      timingEvents = te;
    }

    EcalCrystalTimingCalibration(float m, float me, float r, float tc, std::vector<EcalTimingEvent> te, bool wm) :
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
        timingEvents.push_back(EcalTimingEvent(amp,t,sigmaT,ee));
        updateChi2();
        return true;
      }
      else
        return false;
    }

    bool insertEvent(EcalTimingEvent te)
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
        EcalTimingEvent toRemove = *maxChi2Itr;
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
      for(std::vector<EcalTimingEvent>::iterator itr = timingEvents.begin();
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
      for(std::vector<EcalTimingEvent>::iterator itr = timingEvents.begin();
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
      for(std::vector<EcalTimingEvent>::const_iterator itr = timingEvents.begin();
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
      for(std::vector<EcalTimingEvent>::const_iterator itr = timingEvents.begin();
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
