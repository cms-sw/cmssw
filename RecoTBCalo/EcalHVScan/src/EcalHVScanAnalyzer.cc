/**\class EcalHVScanAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Shahram RAHATLOU
//         Created:  Tue Aug  2 16:15:01 CEST 2005
// $Id: EcalHVScanAnalyzer.cc,v 1.2 2006/01/10 13:37:37 rahatlou Exp $
//
//
#include "RecoTBCalo/EcalHVScan/src/EcalHVScanAnalyzer.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalPnDiodeDigi.h"

//#include<fstream>
#include <iostream>
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TGraph.h"
#include "TF1.h"
#include<string>
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//


//========================================================================
EcalHVScanAnalyzer::EcalHVScanAnalyzer( const edm::ParameterSet& iConfig )
//========================================================================
{
   //now do what ever initialization is needed
   rootfile_          = iConfig.getUntrackedParameter<std::string>("rootfile","hvscan.root");
   hitCollection_     = iConfig.getParameter<std::string>("hitCollection");
   hitProducer_       = iConfig.getParameter<std::string>("hitProducer");
   pndiodeProducer_   = iConfig.getParameter<std::string>("pndiodeProducer");

  initPNTTMap();


   std::cout << "EcalHVScanAnalyzer: fetching hitCollection: " << hitCollection_.c_str()
	<< " produced by " << hitProducer_.c_str() << std::endl;

   // initialize the tree
   tree_ = new TTree("hvscan","HV Scan Analysis");
   tree_->Branch("ampl",tAmpl,"ampl[85][20]/F");
   tree_->Branch("jitter",tJitter,"jitter[85][20]/F");
   tree_->Branch("chi2",tChi2,"chi2[85][20]/F");
   tree_->Branch("PN1",tAmplPN1,"PN1[85][20]/F");
   tree_->Branch("PN2",tAmplPN2,"PN2[85][20]/F");
   tree_->Branch("iC",tiC,"iC[85][20]/I");
   tree_->Branch("iEta",tiEta,"iEta[85][20]/I");
   tree_->Branch("iPhi",tiPhi,"iPhi[85][20]/I");
   tree_->Branch("iTT",tiTT,"iTT[85][20]/I");

}


//========================================================================
EcalHVScanAnalyzer::~EcalHVScanAnalyzer()
//========================================================================
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  delete tree_;
}

//========================================================================
void
EcalHVScanAnalyzer::beginJob(edm::EventSetup const&) {
//========================================================================
  h_ampl_=TH1F("amplitude","Fitted Pulse Shape Amplitude",800,300,2700);
  h_jitt_=TH1F("jitter","Fitted Pulse Shape Jitter",500,0.,5.);

  // histo for each PN diode
  for(int i=0; i<10; ++i) {
    char htit[100];
    char pndiodename[100];
    sprintf(pndiodename,"average ADC count vs. sample - pndiode_%2d",i+1);
    sprintf(htit,"pndiode_%2d",i+1);
    h1d_pnd_[i] = TH1F(htit,pndiodename,50,0.5,50.5);

    sprintf(htit,"max_pndiode_%2d",i+1);
    sprintf(pndiodename,"max fitted amplitude - pndiode_%2d",i+1);
    h_pnd_max_[i] = TH1F(htit,pndiodename,1000,700,1050);

    sprintf(htit,"t0_pndiode_%2d",i+1);
    sprintf(pndiodename,"fitted t0 - pndiode_%2d",i+1);
    h_pnd_t0_[i] = TH1F(htit,pndiodename,100,20.5,50.5);
  }

  // histo for each xtal
  for(int i=0; i<85; ++i) {
    for(int j=0; j<20; ++j) {
      char htit[100];
      char hdesc[100];

      sprintf(htit,"amplitude_xtal_%d_%d",i+1,j+1);
      sprintf(hdesc,"fitted amplitudes xtal eta:%d phi:%d",i+1,j+1);
      h_ampl_xtal_[i][j] =  TH1F(htit,hdesc,2400,300.5,2700.5);

      sprintf(htit,"norm_ampl_xtal_%d_%d",i+1,j+1);
      sprintf(hdesc,"normalized amplitudes xtal eta:%d phi:%d",i+1,j+1);
      h_norm_ampl_xtal_[i][j] =  TH1F(htit,hdesc,200,1.2,2.0);


      sprintf(htit,"jitter_xtal_%d_%d",i+1,j+1);
      sprintf(hdesc,"fitted jitters xtal eta:%d phi:%d",i+1,j+1);
      h_jitt_xtal_[i][j] = TH1F(htit,hdesc,500,0.,5.);

/*
      sprintf(htit,"alpha_xtal_%d_%d",i+1,j+1);
      sprintf(hdesc,"fitted \\alpha xtal eta:%d phi:%d",i+1,j+1);
      h_alpha_xtal_[i][j] = TH1F(htit,hdesc,200,1.,3.);

      sprintf(htit,"tp_xtal_%d_%d",i+1,j+1);
      sprintf(hdesc,"fitted t_{p} xtal eta:%d phi:%d",i+1,j+1);
      h_tp_xtal_[i][j] = TH1F(htit,hdesc,200,1.,3.);
*/

    }
  }

  h2d_anfit_tot_ = TH2F("anfit_tot","analytic fit total",85,0.5,85.5,20,0.5,20.5);
  h2d_anfit_bad_ = TH2F("anfit_bad","analytic fit failed",85,0.5,85.5,20,0.5,20.5);

}

//========================================================================
void
EcalHVScanAnalyzer::endJob() {
//========================================================================

  TFile f(rootfile_.c_str(),"RECREATE");
  h_ampl_.Write();
  h_jitt_.Write();

  for(int i=0; i<10; ++i) {
    h1d_pnd_[i].Write();
    h_pnd_max_[i].Write();
    h_pnd_t0_[i].Write();
  }

  TH1F h_norm_ampl("norm_ampl","normalized amplitudes all xtals",110,1.0,2.1);
  for(int i=0; i<85; ++i) {
    for(int j=0; j<20; ++j) {
      //if(h_ampl_xtal_[i][j].GetEntries()>0) h_ampl_xtal_[i][j].Fit("gaus","QL");
      h_ampl_xtal_[i][j].Write();

      h_jitt_xtal_[i][j].Write();

      if(h_norm_ampl_xtal_[i][j].GetEntries()>10.) {
         h_norm_ampl_xtal_[i][j].Fit("gaus","QL");
         TF1* f1 = h_norm_ampl_xtal_[i][j].GetFunction("gaus");
         h_norm_ampl.Fill( f1->GetParameter(1)); // mean of distributions for each xtal
      }
      h_norm_ampl_xtal_[i][j].Write();
    }
  }
  h_norm_ampl.Fit("gaus","QL");
  h_norm_ampl.Write();

  h2d_anfit_tot_.SetOption("COLZ");
  h2d_anfit_tot_.SetXTitle("\\eta index");
  h2d_anfit_tot_.SetYTitle("\\phi index");
  h2d_anfit_tot_.Write();
  h2d_anfit_bad_.SetOption("COLZ");
  h2d_anfit_bad_.SetXTitle("\\eta index");
  h2d_anfit_bad_.SetYTitle("\\phi index");
  h2d_anfit_bad_.Write();



  TH2F h2d_anfit_failure_("anfit_failure","fraction of failed analytic fits",85,0.5,85.5,20,0.5,20.5);
  for(int i=1; i <= (int)h2d_anfit_tot_.GetNbinsX(); ++i) {
    for(int j=1; j <= (int)h2d_anfit_tot_.GetNbinsY(); ++j) {
      h2d_anfit_failure_.SetBinContent(i,j,0.);
      if( h2d_anfit_tot_.GetBinContent(i,j)>0. )
        h2d_anfit_failure_.SetBinContent(i,j, h2d_anfit_bad_.GetBinContent(i,j)/h2d_anfit_tot_.GetBinContent(i,j));
    }
  }
  h2d_anfit_failure_.SetOption("COLZ");
  h2d_anfit_failure_.SetXTitle("\\eta index");
  h2d_anfit_failure_.SetYTitle("\\phi index");
  h2d_anfit_failure_.Write();

  // store the tree in the output file
  tree_->Write();

  f.Close();
}

//
// member functions
//

//========================================================================
void
EcalHVScanAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {
//========================================================================

   using namespace edm;
   using namespace cms;

   // take a look at PNdiodes
   Handle<EcalPnDiodeDigiCollection> h_pndiodes;
   try {
     iEvent.getByLabel( pndiodeProducer_,h_pndiodes);
   } catch ( std::exception& ex ) {
     std::cerr << "Error! can't get the EcalPnDiodeDigiCollection object " << std::endl;
   }
   const EcalPnDiodeDigiCollection* pndiodes = h_pndiodes.product();
   std::cout << "length of EcalPnDiodeDigiCollection: " << pndiodes->size() << std::endl;

   // find max of PND signal
   std::vector<double> maxAmplPN;
   for(EcalPnDiodeDigiCollection::const_iterator ipnd=pndiodes->begin(); ipnd!=pndiodes->end(); ++ipnd) {
     //std::cout << "PNDiode Id: " << ipnd->id() << "\tsize: " << ipnd->size() << std::endl;
     for(int is=0; is < ipnd->size() ; ++is) {
       float x1 =  h1d_pnd_[ipnd->id().iPnId()-1].GetBinContent(is+1);
       h1d_pnd_[ipnd->id().iPnId()-1].SetBinContent(is+1, x1+ipnd->sample(is).adc());
     }

     // 1D fit to PN diode to find the max
     PNfit res  = maxPNDiode( *ipnd );
     maxAmplPN.push_back( res.max );
     h_pnd_max_[ipnd->id().iPnId()-1].Fill( res.max );
     h_pnd_t0_[ipnd->id().iPnId()-1].Fill( res.t0 );

     //if(ipnd->id().iPnId() == 2 )
     //std::cout << "PNDiode " << ipnd->id() << "\t max amplitude: " << res.max << ", t0: " << res.t0 << std::endl;
   }

   // fetch the digis and compute signal amplitude
   Handle<EcalUncalibratedRecHitCollection> phits;
   try {
     //std::cout << "EcalHVScanAnalyzer::analyze getting product with label: " << digiProducer_.c_str()<< " prodname: " << digiCollection_.c_str() << endl;
     iEvent.getByLabel( hitProducer_, hitCollection_,phits);
     //iEvent.getByLabel( hitProducer_, phits);
   } catch ( std::exception& ex ) {
     std::cerr << "Error! can't get the product " << hitCollection_.c_str() << std::endl;
   }

   // reset tree variables
   for(int i=0;i<85;++i) {
     for(int j=0;j<20;++j) {
       tAmpl[i][j] = 0.;
       tJitter[i][j] = 0.;
       tChi2[i][j] = 0.;
       tAmplPN1[i][j] = 0.;
       tAmplPN2[i][j] = 0.;
       tAmplPN2[i][j] = 0.;
       tiC[i][j] = 0;
       tiEta[i][j] = 0;
       tiPhi[i][j] = 0;
       tiTT[i][j] = 0;
     }
   }

   // loop over hits
   const EcalUncalibratedRecHitCollection* hits = phits.product(); // get a ptr to the product
   std::cout << "# of EcalUncalibratedRecHits hits: " << hits->size() << std::endl;
   for(EcalUncalibratedRecHitCollection::const_iterator ithit = hits->begin(); ithit != hits->end(); ++ithit) {


     EBDetId anid(ithit->id());
     if(ithit->chi2()>0.) { // make sure fit has converged
       h_ampl_.Fill(ithit->amplitude());
       h_jitt_.Fill(ithit->jitter());

       // std::cout << "det Id: " << anid << "\t xtal # " <<anid.ic() << std::endl;
       h_ampl_xtal_[anid.ieta()-1][anid.iphi()-1].Fill(ithit->amplitude());
       h_jitt_xtal_[anid.ieta()-1][anid.iphi()-1].Fill(ithit->jitter());

       // normalized amplitude - use average of 2 PNdiodes for each group of TTs
       double averagePNdiode = 0.5*( maxAmplPN[ pnFromTT(anid.tower()).first ] +
                                     maxAmplPN[ pnFromTT(anid.tower()).second ] );

       double normAmpl = ithit->amplitude() / averagePNdiode;
       h_norm_ampl_xtal_[anid.ieta()-1][anid.iphi()-1].Fill(normAmpl);


      /*
       std::cout << anid.tower() << " iTT: " << anid.tower().iTT()
                                 << " PNdiode: " << pnFromTT(anid.tower()).first << pnFromTT(anid.tower()).second
                                 << std::endl;
      */
     }

     //  compute fraction of bad analytic fits
     h2d_anfit_tot_.Fill(anid.ieta(),anid.iphi());
     if(ithit->chi2()<0) {
       h2d_anfit_bad_.Fill(anid.ieta(),anid.iphi());
     }

     // fill the tree arrays
     //tAmpl[tnXtal] = ithit->amplitude();
     tAmpl[anid.ietaAbs()-1][anid.iphi()-1] = ithit->amplitude();
     tJitter[anid.ietaAbs()-1][anid.iphi()-1]  = ithit->jitter();
     tChi2[anid.ietaAbs()-1][anid.iphi()-1]  = ithit->chi2();
     tAmplPN1[anid.ietaAbs()-1][anid.iphi()-1]  = maxAmplPN[ pnFromTT(anid.tower()).first ];
     tAmplPN2[anid.ietaAbs()-1][anid.iphi()-1]  = maxAmplPN[ pnFromTT(anid.tower()).second ];
     tiC[anid.ietaAbs()-1][anid.iphi()-1]  = anid.ic();
     tiTT[anid.ietaAbs()-1][anid.iphi()-1]  = anid.tower().iTT();
     tiEta[anid.ietaAbs()-1][anid.iphi()-1]  = anid.ietaAbs();
     tiPhi[anid.ietaAbs()-1][anid.iphi()-1]  = anid.iphi();


     // debugging printout
     if(ithit->chi2()<0 && false)
     std::cout << "analytic fit failed! EcalUncalibratedRecHit  id: "
               << EBDetId(ithit->id()) << "\t"
               << "amplitude: " << ithit->amplitude() << ", jitter: " << ithit->jitter()
               << std::endl;

     /*
     std::cout << "EBDetId: " << anid
               << " iCtal: " << anid.ic() 
               //<< " trig tow: " << ( (anid.tower_ieta()-1)*4 + anid.tower_iphi())
               << "\tTrig tower ID: " << anid.tower()
               << "\tiTT: " << anid.tower().iTT()
               << "\tTT hindex: " << anid.tower().hashedIndex()
                << std::endl;
     */



   }//end of loop over hits
   // fill tree
   tree_->Fill();

}


//========================================================================
PNfit EcalHVScanAnalyzer::maxPNDiode(const EcalPnDiodeDigi& pndigi) {
//========================================================================
  const int ns = 50;
  double sample[ns],ampl[ns];
  for(int is=0; is < ns ; ++is) {
    sample[is] = is;
    ampl[is] = pndigi.sample(is).adc();
  }
  TGraph gpn(ns,sample,ampl);
  TF1  mypol3("mypol3","pol3",25,50);
  gpn.Fit("mypol3","QFR","",25,50);
  //TF1* pol3 = (TF1*) gpn.GetFunction("mypol3");
  PNfit res;
  res.max = mypol3.GetMaximum();
  res.t0  = mypol3.GetMaximumX();

  return res;
}

//========================================================================
std::pair<int,int> EcalHVScanAnalyzer::pnFromTT(const EcalTrigTowerDetId& tt) {
//========================================================================

  return pnFromTTMap_[ tt.iTT() ];
  //return pnFromTTMap_[ 1 ];

}

//========================================================================
void EcalHVScanAnalyzer::initPNTTMap() {
//========================================================================

  using namespace std;

  // pairs of PN diodes for groups of trigger towers
  pair<int,int> pair05,pair16,pair27,pair38,pair49;

  pair05.first=0;
  pair05.second=5;

  pair16.first=1;
  pair16.second=6;

  pair27.first=2;
  pair27.second=7;

  pair38.first=3;
  pair38.second=8;

  pair49.first=4;
  pair49.second=9;

  pnFromTTMap_[1] = pair05;
  pnFromTTMap_[2] = pair05;
  pnFromTTMap_[3] = pair05;
  pnFromTTMap_[4] = pair05;
  pnFromTTMap_[5] = pair16;
  pnFromTTMap_[6] = pair16;
  pnFromTTMap_[7] = pair16;
  pnFromTTMap_[8] = pair16;
  pnFromTTMap_[9] = pair16;
  pnFromTTMap_[10] = pair16;
  pnFromTTMap_[11] = pair16;
  pnFromTTMap_[12] = pair16;
  pnFromTTMap_[13] = pair16;
  pnFromTTMap_[14] = pair16;
  pnFromTTMap_[15] = pair16;
  pnFromTTMap_[16] = pair16;
  pnFromTTMap_[17] = pair16;
  pnFromTTMap_[18] = pair16;
  pnFromTTMap_[19] = pair16;
  pnFromTTMap_[20] = pair16;
  pnFromTTMap_[21] = pair27;
  pnFromTTMap_[22] = pair27; 
  pnFromTTMap_[23] = pair27;
  pnFromTTMap_[24] = pair27;
  pnFromTTMap_[25] = pair27;
  pnFromTTMap_[26] = pair27;
  pnFromTTMap_[27] = pair27;
  pnFromTTMap_[28] = pair27;
  pnFromTTMap_[29] = pair27;
  pnFromTTMap_[30] = pair27;
  pnFromTTMap_[31] = pair27;
  pnFromTTMap_[32] = pair27; 
  pnFromTTMap_[33] = pair27;
  pnFromTTMap_[34] = pair27;
  pnFromTTMap_[35] = pair27;
  pnFromTTMap_[36] = pair27;
  pnFromTTMap_[37] = pair38;
  pnFromTTMap_[38] = pair38;
  pnFromTTMap_[39] = pair38;
  pnFromTTMap_[40] = pair38;
  pnFromTTMap_[41] = pair38;
  pnFromTTMap_[42] = pair38; 
  pnFromTTMap_[43] = pair38;
  pnFromTTMap_[44] = pair38;
  pnFromTTMap_[45] = pair38;
  pnFromTTMap_[46] = pair38;
  pnFromTTMap_[47] = pair38;
  pnFromTTMap_[48] = pair38;
  pnFromTTMap_[49] = pair38;
  pnFromTTMap_[50] = pair38;
  pnFromTTMap_[51] = pair38;
  pnFromTTMap_[52] = pair38; 
  pnFromTTMap_[53] = pair49;
  pnFromTTMap_[54] = pair49;
  pnFromTTMap_[55] = pair49;
  pnFromTTMap_[56] = pair49;
  pnFromTTMap_[57] = pair49;
  pnFromTTMap_[58] = pair49;
  pnFromTTMap_[59] = pair49;
  pnFromTTMap_[60] = pair49;
  pnFromTTMap_[61] = pair49;
  pnFromTTMap_[62] = pair49; 
  pnFromTTMap_[63] = pair49;
  pnFromTTMap_[64] = pair49;
  pnFromTTMap_[65] = pair49;
  pnFromTTMap_[66] = pair49;
  pnFromTTMap_[67] = pair49;
  pnFromTTMap_[68] = pair49;

}
