#ifndef CONDTOOLS_SISTRIP_SISTRIPMISCALIBRATEHELPER
#define CONDTOOLS_SISTRIP_SISTRIPMISCALIBRATEHELPER

#include <numeric>
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h" 

namespace SiStripMiscalibrate {

  /*-----------------
  / Auxilliary class to store averages and std. deviations
  /------------------*/
  class Entry{
  public:
    Entry():
      entries(0),
      sum(0),
      sq_sum(0){}

    double mean() {return sum / entries;}
    double std_dev() {
      double tmean = mean();
      return (sq_sum - entries*tmean*tmean)>0 ? sqrt((sq_sum - entries*tmean*tmean)/(entries-1)) : 0.;
    }
    double mean_rms() { return std_dev()/sqrt(entries); }

    void add(double val){
      entries++;
      sum += val;
      sq_sum += val*val;
    }

    void reset() {
      entries = 0;
      sum = 0;
      sq_sum = 0;
    }
  private:
    long int entries;
    double sum, sq_sum;
  };

  /*-----------------
  / Auxilliary struct to store scale & smear factors
  /------------------*/
  struct Smearings{
    Smearings(){
      m_doScale = false;
      m_doSmear = false;
      m_scaleFactor = 1.;
      m_smearFactor = 0.;
    }
    ~Smearings(){}
    
    void setSmearing(bool doScale,bool doSmear,double the_scaleFactor,double the_smearFactor){
      m_doScale = doScale;
      m_doSmear = doSmear;
      m_scaleFactor = the_scaleFactor;
      m_smearFactor = the_smearFactor;
    }
    
    bool m_doScale;
    bool m_doSmear;
    double m_scaleFactor;
    double m_smearFactor;
  };

  /*-----------------
  / Methods used in the miscalibration tools
  /------------------*/
  
  std::pair<float,float> getTruncatedRange(const TrackerMap* theMap);  
  sistripsummary::TrackerRegion getRegionFromString(std::string region);
  std::vector<sistripsummary::TrackerRegion> getRegionsFromDetId(const TrackerTopology* m_trackerTopo,DetId detid);

};

#endif
