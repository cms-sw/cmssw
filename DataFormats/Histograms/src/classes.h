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
  };

  namespace {
    edm::Wrapper<MEtoROOT> theValidData1;
  }
}
