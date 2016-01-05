//////////////////////////////////////////////////////////////////////////
// ---------------------------Event.h------------------------------------
//////////////////////////////////////////////////////////////////////////

#ifndef ADD_EVENT
#define ADD_EVENT

#include "TMath.h"
#include <vector>
#include <iostream>

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

struct Event
{

    Double_t trueValue;
    Double_t predictedValue;
    Double_t DTPt;
    Double_t CSCPt;
    Double_t tmvaPt;
    Double_t tmvaPt1;
    Int_t Mode;
    Int_t Quality;

    static Int_t sortingIndex;
    Int_t id;    
    std::vector<Double_t> data;         

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

#endif
