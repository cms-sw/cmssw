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
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"
#include "TString.h"
#include <stdint.h>

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

    MEtoEDM<TH1F> dummy3;
    MEtoEDM<TH1S> dummy3s;
    MEtoEDM<TH1D> dummy3d;
    MEtoEDM<TH2F> dummy4;
    MEtoEDM<TH2S> dummy4s;
    MEtoEDM<TH2D> dummy4d;
    MEtoEDM<TH3F> dummy5;
    MEtoEDM<TProfile> dummy6;
    MEtoEDM<TProfile2D> dummy7;
    MEtoEDM<double> dummy8;
    MEtoEDM<int> dummy9;
    MEtoEDM<long long> dummy9a;
    MEtoEDM<TString> dummy10;
    MEtoEDM<TH1F>::MEtoEDMObject blah1;
    MEtoEDM<TH1S>::MEtoEDMObject blah2;
    MEtoEDM<TH1D>::MEtoEDMObject blah3;
    MEtoEDM<TH2F>::MEtoEDMObject blah4;
    MEtoEDM<TH2S>::MEtoEDMObject blah5;
    MEtoEDM<TH2D>::MEtoEDMObject blah6;
    MEtoEDM<TH3F>::MEtoEDMObject blah7;
    MEtoEDM<TProfile>::MEtoEDMObject blah8;
    MEtoEDM<TProfile2D>::MEtoEDMObject blah9;
    MEtoEDM<double>::MEtoEDMObject blah10;
    MEtoEDM<TString>::MEtoEDMObject blah11;
    std::vector<MEtoEDM<TH1F>::MEtoEDMObject> dummy11;
    std::vector<MEtoEDM<TH1S>::MEtoEDMObject> dummy11s;
    std::vector<MEtoEDM<TH1D>::MEtoEDMObject> dummy11d;
    std::vector<MEtoEDM<TH2F>::MEtoEDMObject> dummy12;
    std::vector<MEtoEDM<TH2S>::MEtoEDMObject> dummy12s;
    std::vector<MEtoEDM<TH2D>::MEtoEDMObject> dummy12d;
    std::vector<MEtoEDM<TH3F>::MEtoEDMObject> dummy13;
    std::vector<MEtoEDM<TProfile>::MEtoEDMObject> dummy14;
    std::vector<MEtoEDM<TProfile2D>::MEtoEDMObject> dummy15;
    std::vector<MEtoEDM<double>::MEtoEDMObject> dummy16;
    std::vector<MEtoEDM<int>::MEtoEDMObject> dummy17;
    std::vector<MEtoEDM<long long>::MEtoEDMObject> dummy17a;
    std::vector<MEtoEDM<TString>::MEtoEDMObject> dummy18;
    edm::Wrapper<MEtoEDM<TH1F> > theValidData1;
    edm::Wrapper<MEtoEDM<TH1S> > theValidData1s;
    edm::Wrapper<MEtoEDM<TH1D> > theValidData1d;
    edm::Wrapper<MEtoEDM<TH2F> > theValidData2;
    edm::Wrapper<MEtoEDM<TH2S> > theValidData2s;
    edm::Wrapper<MEtoEDM<TH2D> > theValidData2d;
    edm::Wrapper<MEtoEDM<TH3F> > theValidData3;
    edm::Wrapper<MEtoEDM<TProfile> > theValidData4;
    edm::Wrapper<MEtoEDM<TProfile2D> > theValidData5;
    edm::Wrapper<MEtoEDM<double> > theValidData6;
    edm::Wrapper<MEtoEDM<int> > theValidData7;
    edm::Wrapper<MEtoEDM<long long> > theValidData7a;
    edm::Wrapper<MEtoEDM<TString> > theValidData8;
  };
}
