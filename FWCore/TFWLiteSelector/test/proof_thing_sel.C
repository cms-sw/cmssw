{
    gROOT->Proof( "lnx7108.lns.cornell.edu" );

    gSystem->Load("libFWCoreFWLite");
    AutoLibraryLoader::enable();
    gSystem->Load("libFWCoreTFWLiteSelectorTest");
    gProof->Exec( ".x Pthing_sel_Remote.C" );
    TDSet c( "TTree", "Events");
    c.Add("/home/gregor/cms/CMSSW_0_7_2/src/FWCore/TFWLiteSelector/test/test.root");
    c.Process( "ThingsTSelector" );
}
