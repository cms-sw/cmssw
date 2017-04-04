#ifndef PCC_h
#define PCC_h
/** \class reco::PCC
 *  
 * Reconstructed PCC object that will contain the moduleID, BX, and counts.
 *
 * \authors: Sam Higginbotham shiggib@cern.ch and Chris Palmer capalmer@cern.ch
 * 
 *
 *
 */
#include <algorithm>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

namespace reco {
class PCC {
static constexpr unsigned int nEmBX=3564;//Empty BX to fill with counts
    public:
        PCC() : m_events(nEmBX){}
        ////////////////////////////////////////////
        void Increment(int mD,int BXid,int count){
            std::vector<int>::iterator it;
            it = std::find(m_ModID.begin(), m_ModID.end(), mD);
            size_t modIndex = it - m_ModID.begin();

            if(it == m_ModID.end()){
                std::vector<int> m_empBX(nEmBX, 0);
                m_ModID.push_back(mD);
                m_counts.push_back(m_empBX); 
            }
           m_counts[modIndex][BXid] += count; 
        }
        ////////////////////////////////////////////
        void eventCounter(int BXid){
            m_events[BXid]++;
        }
        ////////////////////////////////////////////
        std::vector<std::vector<int>> const & read_counts() const {
                return(m_counts);
            }
        ////////////////////////////////////////////
        void printVector()
        { 
            int irow = 0;
            std::vector< std::vector<int> >::const_iterator row; 
            std::vector<int>::const_iterator col; 

            for (row = m_counts.begin(); row != m_counts.end(); ++row)
            { 
            
               std::cout << m_ModID[irow] << " :";
               for (col = row->begin(); col != row->end(); ++col)
               { 
                  std::cout << *col << " "; 
               } 
               irow++;
               std::cout<<std::endl;
            } 

        }

      private:
        std::vector<std::vector<int> > m_counts;
        std::vector<int> m_events;
        std::vector<int> m_ModID;
        std::vector<int> m_empBX;
            

};

}
#endif
