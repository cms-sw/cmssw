{
   gSystem->Load( "libFWCoreFWLite" );
   gSystem->Load("libDataFormatsFWLite");
   gSystem->Load("libDataFormatsCommon");
   gSystem->Load("libDataFormatsCaloTowers");
   gSystem->Load("libDataFormatsHeavyIonEvent");
   gSystem->Load("libSimDataFormatsHiGenData");
   gSystem->AddIncludePath("-I$CMSSW_BASE/src/");
   gSystem->AddIncludePath("-I$CMSSW_RELEASE_BASE/src/");
   AutoLibraryLoader::enable();

}
