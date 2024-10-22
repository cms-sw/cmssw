#ifndef DataFormatsHcalCalibObjectsHcalIsoTrkEventVariables_h
#define DataFormatsHcalCalibObjectsHcalIsoTrkEventVariables_h
#include <string>
#include <vector>

class HcalIsoTrkEventVariables {
public:
  HcalIsoTrkEventVariables() { clear(); }

  void clear() {
    allvertex_ = 0;
    tracks_ = tracksProp_ = tracksSaved_ = tracksLoose_ = tracksTight_ = 0;
    l1Bit_ = trigPass_ = trigPassSel_ = false;
    hltbits_.clear();
    ietaAll_.clear();
    ietaGood_.clear();
    trackType_.clear();
  };

  int allvertex_, tracks_, tracksProp_, tracksSaved_;
  int tracksLoose_, tracksTight_;
  bool l1Bit_, trigPass_, trigPassSel_;
  std::vector<int> ietaAll_, ietaGood_, trackType_;
  std::vector<bool> hltbits_;
};

typedef std::vector<HcalIsoTrkEventVariables> HcalIsoTrkEventVariablesCollection;
#endif
