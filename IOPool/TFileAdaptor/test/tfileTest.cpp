#include "TFile.h"
#include <TROOT.h>
#include <TSystem.h>
#include <TPluginManager.h>
#include <TEnv.h>

#include <string>
#include <iostream>

#include "PluginManager/PluginManager.h"
#include "SealBase/Error.h"

int main (int argc, char *argv[]) 
{
    /*
      TestMain tfileTest zip-member:file:bha.zip#bha

      TestMain tfileTest rfio:suncmsc.cern.ch:/data/valid/test/vincenzo/testPool/evData/EVD0_EventData.56709894d26a11d78dd20040f45cca94.1.h300eemm.TestSignalHits
      TestMain tfileTest zip-member:rfio:suncmsc.cern.ch:/data/valid/test/vincenzo/testZip/test1.zip#file.5

    */

    seal::PluginManager::get ()->initialise ();

    gEnv->SetValue("Root.Stacktrace", "0");
    // set our own root plugin
    gROOT->GetPluginManager()->AddHandler("TFile", "^file:",     
					  "TStorageFactoryFile", "TFileAdaptorModule",     
					  "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)");   
    gROOT->GetPluginManager()->AddHandler("TFile", "^http:",     
					  "TStorageFactoryFile", "TFileAdaptorModule",     
					  "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)");   
    gROOT->GetPluginManager()->AddHandler("TFile", "^ftp:",     
					  "TStorageFactoryFile", "TFileAdaptorModule",     
					  "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)");   
    gROOT->GetPluginManager()->AddHandler("TFile", "^web:",     
					  "TStorageFactoryFile", "TFileAdaptorModule",     
					  "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)");   

    gROOT->GetPluginManager()->AddHandler("TFile", "^gsiftp:",     
					  "TStorageFactoryFile", "TFileAdaptorModule",     
					  "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)");   

    gROOT->GetPluginManager()->AddHandler("TFile", "^sfn:",     
					  "TStorageFactoryFile", "TFileAdaptorModule",     
					  "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)");   

    gROOT->GetPluginManager()->AddHandler("TFile", "^zip-member:",     
					  "TStorageFactoryFile", "TFileAdaptorModule",     
					  "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)");   
    gROOT->GetPluginManager()->AddHandler("TFile", "^rfio:",     
					  "TStorageFactoryFile", "TFileAdaptorModule",     
					  "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)");   
    gROOT->GetPluginManager()->AddHandler("TFile", "^dcache:",     
					  "TStorageFactoryFile", "TFileAdaptorModule",     
					  "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)");   
    gROOT->GetPluginManager()->AddHandler("TFile", "^dcap:",     
					  "TStorageFactoryFile", "TFileAdaptorModule",     
					  "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)");   
    gROOT->GetPluginManager()->AddHandler("TFile", "^gsidcap:",     
					  "TStorageFactoryFile", "TFileAdaptorModule",     
					  "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)");   
    
    gROOT->GetPluginManager()->AddHandler("TSystem", "^file:",     
					  "TStorageFactorySystem", "TFileAdaptorModule",     
					  "TStorageFactorySystem()");
    gROOT->GetPluginManager()->AddHandler("TSystem", "^http:",     
					  "TStorageFactorySystem", "TFileAdaptorModule",     
					  "TStorageFactorySystem()");
    gROOT->GetPluginManager()->AddHandler("TSystem", "^ftp:",     
					  "TStorageFactorySystem", "TFileAdaptorModule",     
					  "TStorageFactorySystem()");
    gROOT->GetPluginManager()->AddHandler("TSystem", "^web:",     
					  "TStorageFactorySystem", "TFileAdaptorModule",     
					  "TStorageFactorySystem()");

    gROOT->GetPluginManager()->AddHandler("TSystem", "^gsiftp:",     
					  "TStorageFactorySystem", "TFileAdaptorModule",     
					  "TStorageFactorySystem()");

    gROOT->GetPluginManager()->AddHandler("TSystem", "^sfn:",     
					  "TStorageFactorySystem", "TFileAdaptorModule",     
					  "TStorageFactorySystem()");
    
    gROOT->GetPluginManager()->AddHandler("TSystem", "^zip-member:",     
					  "TStorageFactorySystem", "TFileAdaptorModule",     
					  "TStorageFactorySystem()");   
    gROOT->GetPluginManager()->AddHandler("TSystem", "^rfio:",     
					  "TStorageFactorySystem", "TFileAdaptorModule",     
					  "TStorageFactorySystem()");
    gROOT->GetPluginManager()->AddHandler("TSystem", "^dcache:",     
					  "TStorageFactorySystem", "TFileAdaptorModule",     
					  "TStorageFactorySystem()");  
    gROOT->GetPluginManager()->AddHandler("TSystem", "^dcap:",     
					  "TStorageFactorySystem", "TFileAdaptorModule",     
					  "TStorageFactorySystem()");
    gROOT->GetPluginManager()->AddHandler("TSystem", "^gsidcap:",     
					  "TStorageFactorySystem", "TFileAdaptorModule",     
					  "TStorageFactorySystem()");
                                                                        
    gROOT->GetPluginManager()->Print(); // use option="a" to see ctors 
 
    std::string fname("file:file:bha");

    if (argc>1) fname =  argv[1];

    TFile * f=0;
    try {
	Bool_t result = gSystem->AccessPathName(fname.c_str(), kFileExists);
	std::cout<< "file " << fname 
		 << ( result ? " does not exist" : " exists" ) << std::endl;
	if (const char *err = gSystem->GetErrorStr ())
	    std::cout<< "error was " << err << "\n";

	f = TFile::Open(fname.c_str());
	std::cout<< "file size " << f->GetSize()<< std::endl;
	f->ls();
    }
    catch(seal::Error &e) {
	std::cout<< "exception: " << e.explain() << std::endl;
    }
    catch(...) {
	std::cout<< "exception...." << std::endl;
    }

    delete f;
    return 0;
}
