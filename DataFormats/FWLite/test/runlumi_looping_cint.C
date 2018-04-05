#include <vector>
#include <TFile.h>
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
#endif

#include "DataFormats/FWLite/interface/Handle.h"

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#endif

void runlumi_looping_cint()
{
    TFile f("prodmerge.root");
    fwlite::Run r(&f);
    fwlite::LuminosityBlock l(&f);

    int i =0;
    int returnValue = 0;
    for( ;r.isValid();++r,++i) {
        cout << r.run() << endl;
        fwlite::Handle<vector<edmtest::Thing> > pThing;
        pThing.getByLabel(r,"Thing","beginRun");
   
        for(int i=0; i!=pThing.ref().size();++i) {
            cout <<pThing.ref().at(i).a<<" ";
        }
        cout << endl;
    }  
    if (i==0) {
        cout <<"First run loop failed!"<<endl;
        returnValue = 1;
    }

    i =0;
    returnValue = 0;
    for( ;l.isValid();++l,++i) {
        cout << l.id().run() << " " << l.id().luminosityBlock() << endl;
        fwlite::Handle<vector<edmtest::Thing> > pThing;
        pThing.getByLabel(l,"Thing","beginLumi");
   
        for(int i=0; i!=pThing.ref().size();++i) {
            cout <<pThing.ref().at(i).a<<" ";
        }
        cout << endl;
    }  
    if (i==0) {
        cout <<"First lumi loop failed!"<<endl;
        returnValue = 1;
    }


    r.toBegin();
    i=0;
    for( ;r;++r,++i) { 
    }
    if (i==0) {
        cout <<"Second run loop failed!"<<endl;
        returnValue = 1;
    }

    l.toBegin();
    i=0;
    for( ;l;++l,++i) { 
    }
    if (i==0) {
        cout <<"Second lumi loop failed!"<<endl;
        returnValue = 1;
    }

    i=0;
    for(r.toBegin(); !r.atEnd();++r,++i) {
        cout << r.run() << endl;
        fwlite::Handle<vector<edmtest::Thing> > pThing;
        pThing.getByLabel(r,"Thing","endRun");
   
        for(int i=0; i!=pThing.ref().size();++i) {
            cout <<pThing.ref().at(i).a<<" ";
        }
        cout << endl;
    }
    if (i==0) {
        cout <<"Third run loop failed!"<<endl;
        returnValue = 1;
    }

    i=0;
    for(l.toBegin(); !l.atEnd();++l,++i) {
        cout << l.id().run() << " " << l.id().luminosityBlock() << endl;
        fwlite::Handle<vector<edmtest::Thing> > pThing;
        pThing.getByLabel(l,"Thing","endLumi");
   
        for(int i=0; i!=pThing.ref().size();++i) {
            cout <<pThing.ref().at(i).a<<" ";
        }
        cout << endl;
    }
    if (i==0) {
        cout <<"Third lumi loop failed!"<<endl;
        returnValue = 1;
    }

    exit(returnValue);
}
