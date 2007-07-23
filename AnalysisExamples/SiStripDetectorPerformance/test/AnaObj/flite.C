// Use this to set the environment to compile FWLite macros for CMSSW versions prior to 1_5_0
{
  gROOT->Reset();

  gSystem->Load("libFWCoreFWLite.so"); 
  AutoLibraryLoader::enable();

  const char* env = gSystem->Getenv("CMSSW_BASE");
  if( 0 != env) {
    string dir("-I\"");
    dir += env;
    dir += "/src\"";
    gSystem->AddIncludePath(dir.c_str());
  }

  env = gSystem->Getenv("CMSSW_RELEASE_BASE");
  if( 0 != env) {
    string dir("-I\"");
    dir += env;
    dir += "/src\"";
    gSystem->AddIncludePath(dir.c_str());
  }

  gSystem->AddIncludePath("-I\"/afs/cern.ch/cms/sw/slc3_ia32_gcc323/external/boost/1.33.1/include\"");

  gSystem->AddIncludePath("-I\"/afs/cern.ch/cms/sw/slc3_ia32_gcc323/external/clhep/1.9.2.3/include\"");

  gSystem->AddLinkedLibs("-L$CMSSW_BASE/lib/$SCRAM_ARCH");
  gSystem->AddLinkedLibs("-L$CMSSW_RELEASE_BASE/lib/$SCRAM_ARCH");
  gSystem->AddLinkedLibs("-lSimDataFormatsHepMCProduct");
}
