#ifndef L1TRateHelper_H
#define L1TRateHelper_H
/**
\author Tomasz Fruboes (SINS Warsaw)
 */
 
#include <map>

namespace L1TRateHelper {

      struct TRateStruct {
        TRateStruct() : m_orbitLow(-1), m_orbitHigh(-1), m_events(0) {}; 
        void add(int orbit) {
          if (orbit < m_orbitLow || m_orbitLow==-1) m_orbitLow=orbit;
          if (orbit > m_orbitHigh || m_orbitHigh==-1) m_orbitHigh=orbit;
          ++m_events;
        };
        int getTime() { return (m_orbitHigh+m_orbitLow)/2/m_timeBin; } 
        static  int getTimeForOrbit(const  int &orbit) {return orbit/m_timeBin; };
        static const   int m_timeBin = 11224; 
        int m_orbitLow; 
        int m_orbitHigh; 
        int m_events;
        // (1 s) / (25 ns)) / 3564 = 11 223,3446
        bool operator()(const   int &o1, const  int &o2) const {
          return getTimeForOrbit(o1) < getTimeForOrbit(o2); 
        };

      };
      
      class L1TRateHelper  {
      
        public:
          L1TRateHelper() : m_lastRemovedOrbit(-1), m_timeStart(-1) {};
            
         
          /**
          returns time (in pair.first) and rate for earliest record, removes earliest record from map. 
          Time here is measured wrt to first record removed (first call of removeAndGetRateForEarliestTime will get time value equal to 1). As consequence: time returned in pair.second is shifted wrt to time from getEarliestTime, getLastTime, getTimeForOrbit methods
          */
          std::pair<int, int> removeAndGetRateForEarliestTime();
            
          /// Adds event with specified orbit. If the event for some reason comes late (after removal of coresponding entry from m_rateMap), it wont be accounted 
          void addOrbit(int orbit) { if (orbit > m_lastRemovedOrbit) m_rateMap[orbit].add(orbit);};
          
          /// gets time of earliest event (orbit) recorded in event. Returned time -1 is invalid
          int  getEarliestTime() {
            if (m_rateMap.begin() == m_rateMap.end() ) return -1;
            return m_rateMap.begin()->second.getTime();
          };
          
          /// gets time of latest event (orbit) recorded in event. Returned time -1 is invalid
          int  getLastTime() {
            if (m_rateMap.begin() == m_rateMap.end() ) return -1;
            return m_rateMap.rbegin()->second.getTime();
          };
          
          ///
          int  getTimeForOrbit(int orbit) {return TRateStruct::getTimeForOrbit(orbit); };
                
        private:
          typedef std::map <int, TRateStruct, TRateStruct > TRateMap;
          TRateMap m_rateMap;
          int m_lastRemovedOrbit;
          int m_timeStart; 
              
      
      
      };      
}



#endif
