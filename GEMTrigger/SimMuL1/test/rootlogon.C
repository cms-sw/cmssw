{
char *hosttype = gSystem->Getenv( "HOSTTYPE" );
char *rootsys  = gSystem->Getenv( "ROOTSYS" );

// gROOT->Reset();                // Reseting ROOT
gROOT->LoadMacro("tdrstyle.C");
setTDRStyle();

//gSystem->Load("libFWCoreFWLite") ;
//AutoLibraryLoader::enable() ;

printf( "libraries loaded\n" );

}

