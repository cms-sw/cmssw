{
gSystem->Load("libFWCoreFWLite");
AutoLibraryLoader::enable();
vector<string>  files;
files.push_back("good.root");
files.push_back("good_delta5.root");
fwlite::ChainEvent e(files);
#include "DataFormats/FWLite/interface/Handle.h" 

int i =0;
int returnValue = 0;
TFile* f = 0;

for( ;e.isValid();++e,++i) {
  if (e.getTFile() != f) {
    f = e.getTFile();
    cout << "New file " << f->GetName() << endl;
  }

  fwlite::Handle<vector<edmtest::Thing> > pThing;
  //pThing.getByLabel(e,"Thing","","TEST"); //WORKS
  pThing.getByLabel(e,"Thing");
  
  for(i=0; i!=pThing.ref().size();++i) {
    cout <<pThing.ref().at(i).a<<endl;
  }
}  
if (i==0) {
  cout <<"First loop failed!"<<endl;
  returnValue = 1;
}
e.toBegin();

i=0;
for( ;e;++e,++i) { 
}

if (i==0) {
  cout <<"Second loop failed!"<<endl;
  returnValue = 1;
}

i=0;
for(e.toBegin(); !e.atEnd();++e,++i) {
   fwlite::Handle<vector<edmtest::Thing> > pThing;
   //pThing.getByLabel(e,"Thing","","TEST"); //WORKS
   pThing.getByLabel(e,"Thing");
   
   for(i=0; i!=pThing.ref().size();++i) {
      cout <<pThing.ref().at(i).a<<endl;
   }

   //DOES NOT WORK
   //for(vector<edmtest::Thing>::const_iterator it = pThing.data()->begin(); it != pThing.data()->end();++it) {
   //   cout <<(*it).a<<endl;
   //}
}
if (i==0) {
  cout <<"Third loop failed!"<<endl;
  returnValue = 1;
}
e.to(0);
for (int j = 0; j<20; ++j) {
  int k = rand() % 10;
  e.to(k);
  edm::EventID id = e.id();
  cout << "Entry " << k << " Run " << id.run() << " event " << id.event() << endl;
}

exit(returnValue);
}
