#include <iostream>
#include <string>


#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"


namespace hvscan {

  class  Analysis {
    public:
    Analysis() {};
    ~Analysis() {};
    void bookHisto() {};

  };

}

int main(int argc, char *argv[]) {

  using namespace hvscan;

  if(argc<2) {
    std::cout << "Usage: RunHVScanAnalysis <rootfile produced with cmsrun>\n"
              << "rootfile must contain a root tree names hvscan" << std::endl;
    return -1;
  }


  // input file
  std::string rootfile(argv[1]);//("/u1/rahatlou/condDB/hv-CMSSW_0_3_0_pre5/src/RecoTBCalo/EcalHVScan/test/hvscan-first600-6711.root");

  std::string ofname("analysis-"+rootfile);

  // open output file and dump histograms
  std::cout << "open output file <" << ofname.c_str() << ">..." << std::endl;
  //TFile of("analysis-hvscan.root","RECREATE");
  TFile of(ofname.c_str(),"RECREATE");

  // histo for each PN diode
  std::cout << "book histograms" << std::endl;
  TH1F h1d_pnd_[10];
  TH1F h_pnd_max_[10];
  TH1F h_pnd_t0_[10];
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
  TH1F h_ampl_xtal_[85][20];
  TH1F h_jitt_xtal_[85][20];
  TH1F h_norm_ampl_xtal_[85][20];
  int nbinAmpl(2400);
  float minAmpl(300.5), maxAmpl(2700.5);
  int nbinNormAmpl(200);
  float minNormAmpl(0.), maxNormAmpl(1.5);

  for(int i=0; i<85; ++i) {
    for(int j=0; j<20; ++j) {
      char htit[100];
      char hdesc[100];

      sprintf(htit,"amplitude_xtal_%d_%d",i+1,j+1);
      sprintf(hdesc,"fitted amplitudes xtal eta:%d phi:%d",i+1,j+1);
      h_ampl_xtal_[i][j] =  TH1F(htit,hdesc,nbinAmpl,minAmpl,maxAmpl);

      sprintf(htit,"norm_ampl_xtal_%d_%d",i+1,j+1);
      sprintf(hdesc,"normalized amplitudes xtal eta:%d phi:%d",i+1,j+1);
      h_norm_ampl_xtal_[i][j] =  TH1F(htit,hdesc,nbinNormAmpl,minNormAmpl,maxNormAmpl);


      sprintf(htit,"jitter_xtal_%d_%d",i+1,j+1);
      sprintf(hdesc,"fitted jitters xtal eta:%d phi:%d",i+1,j+1);
      h_jitt_xtal_[i][j] = TH1F(htit,hdesc,500,0.,5.);
    }
  }

  //open TFile and point to the tree
  std::cout << "reading tree from " << rootfile.c_str() << std::endl;
  TFile* f = TFile::Open(rootfile.c_str());
  TTree* t = (TTree*) f->Get("hvscan");

  // set branch address
  float ampl[85][20],jitter[85][20],chi2[85][20], PN1[85][20], PN2[85][20];
  int   iEta[85][20],iPhi[85][20],iTT[85][20], iC[85][20];

  t->SetBranchAddress("ampl",ampl);
  t->SetBranchAddress("jitter",jitter);
  t->SetBranchAddress("chi2",chi2);
  t->SetBranchAddress("PN1",PN1);
  t->SetBranchAddress("PN2",PN2);
  t->SetBranchAddress("iC",iC);
  t->SetBranchAddress("iEta",iEta);
  t->SetBranchAddress("iPhi",iPhi);
  t->SetBranchAddress("iTT",iTT);

  //counters
  float ntot[85][20], nGood[85][20];


  // loop over tree entries
  for(int iEvt=0; iEvt<t->GetEntries(); ++iEvt) {
    t->GetEntry(iEvt);

    for(int i=0; i<85; ++i) {
      for(int j=0; j<20; ++j) {
        ntot[i][j]++; // all events

        if(0==1 && ampl[i][j]>0. && chi2[i][j]<=0.)
          std::cout << "eta: " << i << " phi: " << j
                    << " ampl: " << ampl[i][j] << " fit failed." << std::endl;

        if(chi2[i][j]>0 && ampl[i][j]>0.) {
          nGood[i][j]++;
          h_ampl_xtal_[i][j].Fill( ampl[i][j] );
          h_norm_ampl_xtal_[i][j].Fill( ampl[i][j]/(0.5*(PN1[i][j]+PN2[i][j]))  );
          h_jitt_xtal_[i][j].Fill( jitter[i][j] );
        }
      }
    }

  } //end of loop

  // close input file
  std::cout << "close input file" << std::endl;
  f->Close();

  // go back to output file
  of.cd();

  // compute goodness of analytic fit
  std::cout << "compute failure rate of analytic fit" << std::endl;
  TH2F h2d_anfit_failure_("anfit_failure","fraction of failed analytic fits",85,0.5,85.5,20,0.5,20.5);
  TH2F h2d_norm_ampl_("h2d_norm_ampl","fitted normalized amplitude",85,0.5,85.5,20,0.5,20.5);
  for(int i=0; i < 85; ++i) {
    for(int j=0; j < 20; ++j) {
      h2d_anfit_failure_.SetBinContent(i+1,j+1,0.);
      if(0==2)
      std::cout << "eta: " << i << " phi: " << j
                << " total: " << ntot[i][j] << " good: " << nGood[i][j]
                << " succesful fraction: " << nGood[i][j]/ntot[i][j]
                << std::endl;

      h2d_anfit_failure_.SetBinContent(i+1,j+1, 1.- nGood[i][j]/ntot[i][j] );
      h2d_norm_ampl_.SetBinContent(i+1,j+1, h_norm_ampl_xtal_[i][j].GetMean() );
    }
  }
  h2d_anfit_failure_.SetOption("COLZ");
  h2d_anfit_failure_.SetXTitle("\\eta index");
  h2d_anfit_failure_.SetYTitle("\\phi index");
  h2d_anfit_failure_.Write();

  h2d_norm_ampl_.SetOption("COLZ");
  h2d_norm_ampl_.SetXTitle("\\eta index");
  h2d_norm_ampl_.SetYTitle("\\phi index");
  h2d_norm_ampl_.Write();

  std::cout << "dump histos for individual xtals" << std::endl;
  for(int i=0; i < 85; ++i) {
    for(int j=0; j < 20; ++j) {
      h_ampl_xtal_[i][j].Write();
      h_norm_ampl_xtal_[i][j].Write();
      h_jitt_xtal_[i][j].Write();
    }
  }

  std::cout << "close output file" << std::endl;
  of.Close();


}
