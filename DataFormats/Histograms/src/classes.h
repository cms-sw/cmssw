#include "DataFormats/Common/interface/Wrapper.h"
#include "TH1C.h"
#include "TH1D.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TH1S.h"
#include "TH2C.h"
#include "TH2D.h"
#include "TH2F.h"
#include "TH2I.h"
#include "TH2S.h"
#include "TH3C.h"
#include "TH3D.h"
#include "TH3F.h"
#include "TH3I.h"
#include "TH3S.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TProfile3D.h"
#include "DataFormats/Histograms/interface/MEtoROOTFormat.h"

namespace {
  struct dictionary {
    edm::Wrapper<TH1C> dummy1C;
    edm::Wrapper<TH1D> dummy1D;
    edm::Wrapper<TH1F> dummy1F;
    edm::Wrapper<TH1I> dummy1I;
    edm::Wrapper<TH1S> dummy1S;
    edm::Wrapper<TH2C> dummy2C;
    edm::Wrapper<TH2D> dummy2D;
    edm::Wrapper<TH2F> dummy2F;
    edm::Wrapper<TH2I> dummy2I;
    edm::Wrapper<TH2S> dummy2S;
    edm::Wrapper<TH3C> dummy3C;
    edm::Wrapper<TH3D> dummy3D;
    edm::Wrapper<TH3F> dummy3F;
    edm::Wrapper<TH3I> dummy3I;
    edm::Wrapper<TH3S> dummy3K;
    edm::Wrapper<TProfile> dummyPr;
    edm::Wrapper<TProfile2D> dummyPr2D;
    edm::Wrapper<TProfile3D> dummyPr3D;
    edm::Wrapper<TObject> dummyObject;
  };

  namespace {
    std::vector<uint32_t> dummy1;
    std::vector<std::vector<uint32_t> > dummy2;
    MEtoROOT<TH1F> dummy3;
    MEtoROOT<TH2F> dummy4;
    MEtoROOT<TH3F> dummy5;
    MEtoROOT<TProfile> dummy6;
    MEtoROOT<TProfile2D> dummy7;
    MEtoROOT<float> dummy8;
    MEtoROOT<int> dummy9;
    MEtoROOT<std::string> dummy10;
    std::vector<MEtoROOT<TH1F>::MEROOTObject> dummy11;
    std::vector<MEtoROOT<TH2F>::MEROOTObject> dummy12;
    std::vector<MEtoROOT<TH3F>::MEROOTObject> dummy13;
    std::vector<MEtoROOT<TProfile>::MEROOTObject> dummy14;
    std::vector<MEtoROOT<TProfile2D>::MEROOTObject> dummy15;
    std::vector<MEtoROOT<float>::MEROOTObject> dummy16;
    std::vector<MEtoROOT<int>::MEROOTObject> dummy17;
    std::vector<MEtoROOT<std::string>::MEROOTObject> dummy18;
    edm::Wrapper<MEtoROOT<TH1F> > theValidData1;
    edm::Wrapper<MEtoROOT<TH2F> > theValidData2;
    edm::Wrapper<MEtoROOT<TH3F> > theValidData3;
    edm::Wrapper<MEtoROOT<TProfile> > theValidData4;
    edm::Wrapper<MEtoROOT<TProfile2D> > theValidData5;
    edm::Wrapper<MEtoROOT<float> > theValidData6;
    edm::Wrapper<MEtoROOT<int> > theValidData7;
    edm::Wrapper<MEtoROOT<std::string> > theValidData8;
  }
}
