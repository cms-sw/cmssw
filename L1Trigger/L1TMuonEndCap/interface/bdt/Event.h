//////////////////////////////////////////////////////////////////////////
// ---------------------------Event.h------------------------------------
//////////////////////////////////////////////////////////////////////////

#ifndef L1Trigger_L1TMuonEndCap_emtf_Event
#define L1Trigger_L1TMuonEndCap_emtf_Event

#include <vector>
#include <iostream>

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

namespace emtf {

struct Event
{

    double trueValue;
    double predictedValue;
    double DTPt;
    double CSCPt;
    double tmvaPt;
    double tmvaPt1;
    int Mode;
    int Quality;

    static int sortingIndex;
    int id;
    std::vector<double> data;

    bool operator< (const Event &rhs) const
    {
        return data[sortingIndex] < rhs.data[sortingIndex];
    }

    void outputEvent()
    {
        std::cout << "trueValue = " << trueValue << std::endl;
        std::cout << "predictedValue = " << predictedValue << std::endl;
        std::cout << "id = " << id << std::endl;
        for(unsigned int i=0; i<data.size(); i++)
        {
            std::cout << "x"<< i << "=" << data[i] << ", ";
        }
        std::cout << std::endl;

    }

    void resetPredictedValue(){ predictedValue = 0; }
};

} // end of emtf namespace

#endif
