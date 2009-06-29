#ifndef SiPixelSummary_h
#define SiPixelSummary_h

#include<vector>
#include<map>
#include <string>
#include<iostream>
#include<boost/cstdint.hpp>
#include "FWCore/Utilities/interface/Exception.h"

/**
  @class SiPixelSummary
  @author Dean Andrew Hidas
  @class to hold historic DQM summary informations
 */

namespace sipixelsummary {
  enum TrackerRegion {
    TRACKER = 0, 
    Barrel = 1,
    Shell_mI = 2,
    Shell_mO = 3,
    Shell_pI = 4,
    Shell_pO = 5,
    Endcap = 6,
    HalfCylinder_mI = 7,
    HalfCylinder_mO = 8,
    HalfCylinder_pI = 9,
    HalfCylinder_pO = 10
  };



}

class SiPixelSummary {

  public:

    struct DetRegistry{
      uint32_t detid;
      uint32_t ibegin;
    };

    class StrictWeakOrdering{
      public:
        bool operator() (const DetRegistry& p,const uint32_t& i) const {return p.detid < i;}
    };



    // SOME DEFINITIONS
    //
    typedef std::vector<float>::const_iterator               ContainerIterator;  
    typedef std::pair<ContainerIterator, ContainerIterator>  Range; 		     
    typedef std::vector<DetRegistry>                         Registry;
    typedef Registry::const_iterator                         RegistryIterator;
    typedef std::vector<float>		                 InputVector;


    SiPixelSummary(std::vector<std::string>& userDBContent);
    SiPixelSummary(const SiPixelSummary& input);
    SiPixelSummary(){};
    ~SiPixelSummary(){};


    ContainerIterator getDataVectorBegin()     const {return v_sum_.begin();  }
    ContainerIterator getDataVectorEnd()       const {return v_sum_.end();    } 
    RegistryIterator  getRegistryVectorBegin() const {return indexes_.begin();}
    RegistryIterator  getRegistryVectorEnd()   const {return indexes_.end();  }

    // RETURNS POSITION OF DETID IN v_sum_
    //
    const Range getRange(const uint32_t& detID) const;


    // RETURNS LIST OF DETIDS 
    //
    std::vector<uint32_t> getDetIds() const;


    // INSERT SUMMARY OBJECTS...
    //
    bool put(const uint32_t& detID, InputVector &input, std::vector<std::string>& userContent );
    bool put(sipixelsummary::TrackerRegion region, InputVector &input, std::vector<std::string>& userContent );
    void setObj(const uint32_t& detID, std::string elementName, float value);


    // RETRIEVE SUMMARY OBJECTS...
    //

    // returns a vector of selected infos related to a given detId 
    std::vector<float> getSummaryObj(uint32_t& detID, std::vector<std::string> list) const; 
    std::vector<float> getSummaryObj(sipixelsummary::TrackerRegion region, std::vector<std::string> list) const; 

    // returns a vector filled with "info elementName" for each detId 
    // The order is SORTED according to the one used in getDetIds() !
    std::vector<float> getSummaryObj(std::string elementName) const;     

    // returns the entire SummaryObj related to one detId
    std::vector<float> getSummaryObj(uint32_t& detID) const;

    // returns everything, all SummaryObjects for all detIds (unsorted !)
    std::vector<float> getSummaryObj() const;		      


    // INLINE METHODS ABOUT RUN, TIME VALUE...
    //
    inline void setUserDBContent(std::vector<std::string> userDBContent)  { userDBContent_ = userDBContent;}
    inline void setRunNr(int inputRunNr)                       { runNr_ = inputRunNr;      }
    inline void setTimeValue(unsigned long long inputTimeValue){ timeValue_=inputTimeValue;}

    inline unsigned long long getTimeValue() const             { return timeValue_;        }
    inline std::vector<std::string>  getUserDBContent() const  { return userDBContent_;    }
    inline int getRunNr() const                                { return runNr_;            }


    // PRINT METHOD...
    //
    void print();


    // SISTRIPSUMMARY MEMBERS...
    //
    std::vector<std::string>        userDBContent_;
    std::vector<float> 	        v_sum_; 
    std::vector<DetRegistry> 	indexes_;

    int runNr_; 
    unsigned long long timeValue_;


  protected:	

    // RETURNS POSITION OF ELEMENTNAME IN userDBContent_
    const short getPosition(std::string elementName) const;


};


#endif
