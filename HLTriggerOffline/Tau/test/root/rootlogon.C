{
char *hosttype = gSystem->Getenv( "HOSTTYPE" );
char *rootsys  = gSystem->Getenv( "ROOTSYS" );

// gROOT->Reset();                // Reseting ROOT
gROOT->LoadMacro("tdrstyle.C");
setTDRStyle();
printf( "libraries loaded\n" );

}

