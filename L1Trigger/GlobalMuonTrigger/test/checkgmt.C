{
using namespace std;

   gROOT->Reset();

   gSystem->Load("libPhysicsToolsFWLite") ;
   FWLiteEnabler::enable() ;
   
   TFile* f = new TFile("runGMT.root") ;
   
   TTree* tevt = f->Get("Events") ;
   
   int nevt = tevt->GetEntries() ;
   std::cout << "Number of Events = " << nevt << std::endl ;
   
   std::vector<L1MuRegionalCand> EvtInDT;
   std::vector<L1MuRegionalCand> EvtInCSC;
   std::vector<L1MuRegionalCand> EvtInRPCb;
   std::vector<L1MuRegionalCand> EvtInRPCf;
   std::vector<L1MuGMTCand> EvtOutGMT;

   TBranch* dtprod = tevt->GetBranch("L1MuRegionalCands_L1MuGMTHWFileReader_DT.obj") ;
   TBranch* cscprod = tevt->GetBranch("L1MuRegionalCands_L1MuGMTHWFileReader_CSC.obj") ;
   TBranch* rpcbprod = tevt->GetBranch("L1MuRegionalCands_L1MuGMTHWFileReader_RPCb.obj") ;
   TBranch* rpcfprod = tevt->GetBranch("L1MuRegionalCands_L1MuGMTHWFileReader_RPCf.obj") ;
   TBranch* gmtprod = tevt->GetBranch("L1MuGMTCands_gmt.obj") ;

   gmtprod->SetAddress( & EvtOutGMT );
   dtprod->SetAddress( & EvtInDT );
   cscprod->SetAddress( & EvtInCSC );
   rpcbprod->SetAddress( & EvtInRPCb );
   rpcfprod->SetAddress( & EvtInRPCf );
   
   
   //
   //   Loop over events
   //
   for ( int iev=0; iev<nevt; iev++ )
   {

     cout << "event " << iev << endl;

     dtprod->GetEntry(iev);
     cscprod->GetEntry(iev);
     rpcbprod->GetEntry(iev);
     rpcfprod->GetEntry(iev);
     gmtprod->GetEntry(iev);

     for(int i=0; i<4; i++) {
       cout << "DT:   " << EvtInDT[i].phiValue() << " " <<
                           EvtInDT[i].etaValue() << " " <<
                           EvtInDT[i].ptValue() << " " <<
                           EvtInDT[i].quality() << endl; 
     }

     for(int i=0; i<4; i++) {
       cout << "CSC:  " << EvtInCSC[i].phiValue() << " " <<
                           EvtInCSC[i].etaValue() << " " <<
                           EvtInCSC[i].ptValue() << " " <<
                           EvtInCSC[i].quality() << endl;
     }

     for(int i=0; i<4; i++) {
       cout << "RPCb: " << EvtInRPCb[i].phiValue() << " " <<
                           EvtInRPCb[i].etaValue() << " " <<
                           EvtInRPCb[i].ptValue() << " " <<
                           EvtInRPCb[i].quality() << endl;
     }

     for(int i=0; i<4; i++) {
       cout << "RPBf: " << EvtInRPCf[i].phiValue() << " " <<
                           EvtInRPCf[i].etaValue() << " " <<
                           EvtInRPCf[i].ptValue() << " " <<
                           EvtInRPCf[i].quality() << endl;
     }

     for(int i=0; i<4; i++) {
       cout << "GMT:  " << EvtOutGMT[i].phiValue() << " " <<
                           EvtOutGMT[i].etaValue() << " " <<
                           EvtOutGMT[i].ptValue() << " " <<
                           EvtOutGMT[i].quality() << endl;
     }


   }
   
}
