#include <TProof.h>
#include <TDSet.h>
#include <TEnv.h>

using namespace std;

#if defined(__CINT__) && !defined(__MAKECINT__)
class loadFWLite {
   public:
      loadFWLite() {
         gSystem->Load("libFWCoreFWLite");
         AutoLibraryLoader::enable();
      }
};

static loadFWLite lfw;
#else
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#endif

void proof_thing2_sel()
{
  if (gSystem->Getenv("TMPDIR")) {
    std::string t(gSystem->Getenv("TMPDIR"));
    if (t.size() > 80)
      t = "/tmp";
    t += "/proof";
    gEnv->SetValue("Proof.Sandbox", t.c_str());
    gEnv->SetValue("ProofLite.SockPathDir", t.c_str());
  }

  //Setup the proof server
  TProof *myProof=TProof::Open( "" );
  
  // This makes sure the TSelector library and dictionary are properly
  // installed in the remote PROOF servers

  // This works, but results in an annoying error message from 'cp',
  // something not right with the how the macro is sent?
  //myProof->Exec( ".x proof_remote.C" );

  // So inline it...
  myProof->Exec("gSystem->Load(\"libFWCoreFWLite\"); "
               "AutoLibraryLoader::enable(); "
  // Have to load library manually since Proof does not use the 
  // mechanism used by TFile to find class dictionaries and therefore
  // the AutoLibraryLoader can not help
               "gSystem->Load(\"libFWCoreTFWLiteSelectorTest\");");
  
  //This creates the 'data set' which defines what files we need to process
  // NOTE: the files given must be accessible by the remote systems
  TDSet c( "TTree", "Events");
  c.Add("$CMSSW_BASE/test.root");
  
  //This makes the actual processing happen
  c.Process( "tfwliteselectortest::ThingsTSelector2" );
}
