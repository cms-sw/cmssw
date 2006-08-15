#include "L1Trigger/L1GctAnalyzer/interface/L1GctJetCheckHistogrammer.h"


//--------------Histogram checks against reconstructed jets---------------------------
//
using namespace reco;

const int L1GctJetCheckHistogrammer::NJETET =50;
const double L1GctJetCheckHistogrammer::MINJETET =   0.;
const double L1GctJetCheckHistogrammer::MAXJETET = 100.; 
const int L1GctJetCheckHistogrammer::NJETETA=50;
const double L1GctJetCheckHistogrammer::MINJETETA=  -5.;
const double L1GctJetCheckHistogrammer::MAXJETETA=   5.; 
const int L1GctJetCheckHistogrammer::NJETPHI=36;
const double L1GctJetCheckHistogrammer::MINJETPHI=-M_PI;
const double L1GctJetCheckHistogrammer::MAXJETPHI= M_PI; 

///constructor
L1GctJetCheckHistogrammer::L1GctJetCheckHistogrammer(TFile* tf, const std::string dir) :
  L1GctCorrelator<reco::GenJetCollection>(tf, dir),
  topJetRankVsGenEt( "topJetRankVsGenEt", "Gct top jet rank vs genJet et",
		     NRANK,  MINRANK,  MAXRANK,
		     NJETET, MINJETET, MAXJETET),
  topJetEtaVsGen   ( "topJetEtaVsGen",    "Gct eta top jet vs genJet eta",
		     NGCTETA, MINGCTETA, MAXGCTETA,
		     NJETETA, MINJETETA, MAXJETETA),
  topJetPhiVsGen   ( "topJetPhiVsGen",    "Gct phi top jet vs genJet phi",
		     NPHI,    MINPHI,    MAXPHI,
		     NJETPHI, MINJETPHI, MAXJETPHI)
{
}

///destructor
L1GctJetCheckHistogrammer::~L1GctJetCheckHistogrammer()
{
  if (setHistogramDirectory()) {
    topJetRankVsGenEt.Write();
    topJetEtaVsGen.Write();
    topJetPhiVsGen.Write();
  }
}

///event processor
void L1GctJetCheckHistogrammer::fillHistograms(const GctOutputData gct)
{
  GenJetCollection::const_iterator gJet=prod().begin();
  topJetRankVsGenEt.Fill(topJet(gct).rank(),     gJet->et());
  topJetEtaVsGen.Fill   (gctEta(topJet(gct)),    gJet->eta());
  topJetPhiVsGen.Fill   (topJet(gct).phiIndex(), gJet->phi());
}

