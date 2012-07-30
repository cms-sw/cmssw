#ifndef HLTHtMhtFilter_h
#define HLTHtMhtFilter_h

/** \class HLTHtMhtFilter
 *
 *  \author Steven Lowette
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
  class ConfigurationDescriptions;
}


class HLTHtMhtFilter : public HLTFilter {

  public:

    explicit HLTHtMhtFilter(const edm::ParameterSet &);
    ~HLTHtMhtFilter();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual bool filter(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    std::string moduleLabel_;
    std::vector<edm::InputTag> htLabels_;
    std::vector<edm::InputTag> mhtLabels_;
    std::vector<double> minHt_;
    std::vector<double> minMht_;
    std::vector<double> minMeff_;
    std::vector<double> meffSlope_;
    unsigned int nOrs_;
    bool saveTags_;

};

#endif
