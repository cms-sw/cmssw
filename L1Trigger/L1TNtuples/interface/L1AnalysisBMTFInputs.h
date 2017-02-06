#ifndef __L1Analysis_L1AnalysisBMTFInputs_H__
#define __L1Analysis_L1AnalysisBMTFInputs_H__

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/Common/interface/Handle.h"

#include <vector>

#include "L1AnalysisBMTFInputsDataFormat.h"

//#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/EDAnalyzer.h"
//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"


namespace L1Analysis
{
  class L1AnalysisBMTFInputs
  {
  public:
    L1AnalysisBMTFInputs();
    ~L1AnalysisBMTFInputs();

    void SetBMPH(const edm::Handle<L1MuDTChambPhContainer > L1MuDTChambPhContainer, unsigned int maxDTPH);
    void SetBMTH(const edm::Handle<L1MuDTChambThContainer > L1MuDTChambThContainer, unsigned int maxDTTH);

    void Reset() {bmtf_.Reset();}
    L1AnalysisBMTFInputsDataFormat * getData() {return &bmtf_;}

  private :
    L1AnalysisBMTFInputsDataFormat bmtf_;
  };
}
#endif


