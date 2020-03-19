#include <TProof.h>
#include <TDSet.h>
#include <TEnv.h>

using namespace std;

#include "FWCore/FWLite/interface/FWLiteEnabler.h"

void proof_thing2_sel() {
  if (gSystem->Getenv("TMPDIR")) {
    std::string t(gSystem->Getenv("TMPDIR"));
    if (t.size() > 80)
      t = "/tmp";
    t += "/proof";
    gEnv->SetValue("Proof.Sandbox", t.c_str());
    gEnv->SetValue("ProofLite.SockPathDir", t.c_str());
  }

  //Setup the proof server
  TProof *myProof = TProof::Open("", "workers=2");

  // This makes sure the TSelector library and dictionary are properly
  // installed in the remote PROOF servers

  // This works, but results in an annoying error message from 'cp',
  // something not right with the how the macro is sent?
  //myProof->Exec( ".x proof_remote.C" );

  // So inline it...
  myProof->Exec(
      "gSystem->Load(\"libFWCoreFWLite\"); "
      "FWLiteEnabler::enable(); "
      "gSystem->Load(\"libFWCoreTFWLiteSelectorTest\");");

  //This creates the 'data set' which defines what files we need to process
  // NOTE: the files given must be accessible by the remote systems
  TDSet c("TTree", "Events");
  c.Add("testTFWLiteSelector.root");

  //This makes the actual processing happen
  c.Process("tfwliteselectortest::ThingsTSelector2");

  myProof->Close("S");
}
