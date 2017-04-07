#ifndef PCC_PCC_h
#define PCC_PCC_h
/** \class reco::PixelClusterCounts
 *  
 * Reconstructed PCC object that will contain the moduleID, BX, and counts.
 *
 * \authors: Sam Higginbotham shiggib@cern.ch and Chris Palmer capalmer@cern.ch
 * 
 *
 */
#include <algorithm>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

namespace reco {
class PixelClusterCounts {
static constexpr unsigned int nEmBX=3564;//Empty BX to fill with counts
    public:
        PixelClusterCounts() : m_events(nEmBX){}
        ////////////////////////////////////////////
        void increment(int mD,int BXid,int count){
            std::vector<int>::iterator it;
            it = std::find(m_ModID.begin(), m_ModID.end(), mD);
            size_t modIndex = it - m_ModID.begin();

            if(it == m_ModID.end()){
                std::vector<int> m_empBX(nEmBX, 0);
                m_ModID.push_back(mD);
                m_counts.insert(m_counts.begin(),m_empBX.begin(),m_empBX.end()); 
            }
            m_counts[nEmBX*modIndex+BXid] += count; 
        }
        ////////////////////////////////////////////
        void eventCounter(int BXid){
            m_events[BXid]++;
        }
        ////////////////////////////////////////////
        std::vector<int> const & readCounts() const {
            return(m_counts);
        }
        ////////////////////////////////////////////
  

      private:
        std::vector<int> m_counts;
        std::vector<int> m_events;
        std::vector<int> m_ModID;
        std::vector<int> m_empBX;
            

};

}
#endif
