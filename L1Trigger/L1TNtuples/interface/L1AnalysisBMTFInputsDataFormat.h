#ifndef __L1Analysis_L1AnalysisBMTFInputsDataFormat_H__
#define __L1Analysis_L1AnalysisBMTFInputsDataFormat_H__

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisBMTFInputsDataFormat
  {

    L1AnalysisBMTFInputsDataFormat(){Reset();};
    ~L1AnalysisBMTFInputsDataFormat(){};

    void Reset()
    {

    phSize = 0;

    phBx.clear();
    phWh.clear();
    phSe.clear();
    phSt.clear();
    phAng.clear();
    phBandAng.clear();
    phCode.clear();
    phTs2Tag.clear();


    thSize = 0;

    thBx.clear();
    thWh.clear();
    thSe.clear();
    thSt.clear();
    thCode.clear();
    thTheta.clear();
    }

    // ---- L1AnalysisBMTFDataFormat information.

    int phSize;
    std::vector<int>    phBx;
    std::vector<int>    phWh;
    std::vector<int>    phSe;
    std::vector<int>    phSt;
    std::vector<float>  phAng;
    std::vector<float>  phBandAng;
    std::vector<int>    phCode;
    std::vector<int>    phTs2Tag;


    int thSize;
    std::vector<int>   thBx;
    std::vector<int>   thWh;
    std::vector<int>   thSe;
    std::vector<int>   thSt;
    std::vector<int> thTheta;
    std::vector<int> thCode; 

  };
}
#endif


