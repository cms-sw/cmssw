{
/*
TString includeString= TString(" -I")+getenv("CMSSW_BASE")+"/src"
+" -I"+getenv("CMSSW_RELEASE_BASE")+"/src"
+" -I/afs/cern.ch/cms/sw/"+getenv("SCRAM_ARCH")+"/external/boost/1.34.1-cms/include"
;

cout << "Include path\n\t "<< includeString.Data() << " \n\n" << endl;  

gSystem->SetIncludePath(includeString);
*/

gSystem->Load("libFWCoreFWLite");  
gSystem->Load("libtestSiStripHistoricDQM"); 
gSystem->Load("libCondFormatsSiStripObjects"); 

AutoLibraryLoader::enable();
}
