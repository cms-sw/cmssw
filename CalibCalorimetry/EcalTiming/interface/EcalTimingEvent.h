class EcalTimingEvent
{
  public:
    float amplitude;
    float time;
    float chi2;
    float sigmaTime;
    float expectedPrecision;

    EcalTimingEvent() :
      amplitude(-1),
      time(-1),
      chi2(-1),
      sigmaTime(-1),
      expectedPrecision(-1)
    {
    }
    
    EcalTimingEvent(float amp,float t,float sigmaT, bool ee) : 
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

    bool operator==(const EcalTimingEvent &first) const
    {
      // only check amp, time, sigmaT
      if(first.amplitude==this->amplitude &&
         first.time==this->time &&
         first.sigmaTime==this->sigmaTime)
        return true;

      return false;
    }

};
