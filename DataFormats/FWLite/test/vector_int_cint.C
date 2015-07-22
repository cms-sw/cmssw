#include <vector>
#include <typeinfo>
#include <TFile.h>
#include <TClass.h>
#include <TROOT.h>

using namespace std;

#if defined(__CINT__) && !defined(__MAKECINT__)
class loadFWLite {
   public:
      loadFWLite() {
         gSystem->Load("libFWCoreFWLite");
         FWLiteEnabler::enable();
      }
};

static loadFWLite lfw;

#if 0
namespace edm {
   typedef
     edm::Wrapper<vector<int> >
     Wrapper<vector<int,allocator<int> > >;
}
#endif
#endif

#include "DataFormats/FWLite/interface/Handle.h"

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/Common/interface/Wrapper.h"
#endif

void vector_int_cint()
{
  gROOT->ProcessLine(".autodict"); // disable auto building of dictionaries.
  const std::type_info& t = edm::Wrapper<vector<int> >::typeInfo();
  TClass* tc = TClass::GetClass(t);
  std::cout << tc->GetName() << std::endl;

  TFile f1("vectorinttest.root");
  fwlite::Event ev(&f1);
  fwlite::Handle<vector<int> > vip;

  for (ev.toBegin(); ! ev.atEnd(); ++ev) {
    edm::EventID id = ev.id();
    cout << "Run " << id.run() << " event " << id.event() << endl;
    vip.getByLabel(ev,"VIP");
    for (unsigned int i=0; i != vip.ref().size(); ++i) {
      cout <<vip.ref().at(i)<<" ";
    }
    cout << endl;
  }
}
