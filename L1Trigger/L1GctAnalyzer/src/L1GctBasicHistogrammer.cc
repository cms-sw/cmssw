#include "L1Trigger/L1GctAnalyzer/interface/L1GctBasicHistogrammer.h"


//--------------Histogram basic gct quantities---------------------------
//
///constructor
L1GctBasicHistogrammer::L1GctBasicHistogrammer(TFile* tf, const std::string dir) :
  L1GctHistogrammer(tf, dir),
  allJetsRank  ( "allJetsRank",    "Jet rank",       NRANK, MINRANK, MAXRANK), 
  allJetsEta   ( "allJetsEta",     "Jet eta",        NETA,  MINETA,  MAXETA ), 
  allJetsPhi   ( "allJetsPhi",     "Jet phi",        NPHI,  MINPHI,  MAXPHI ), 
  allJetsGctEta( "allJetsGctEta",  "Global jet eta", NGCTETA,   MINGCTETA,   MAXGCTETA  ), 
  metValue     ( "metValue",       "Missing Et",     NGCTMETVALUE, MINGCTMETVALUE, MAXGCTMETVALUE), 
  metAngle     ( "metAngle",       "Missing Et phi", NGCTMETPHI,   MINGCTMETPHI,   MAXGCTMETPHI  ) 
{
}

///destructor
L1GctBasicHistogrammer::~L1GctBasicHistogrammer()
{
  setHistogramDirectory();
  allJetsRank.Write();
  allJetsEta.Write();
  allJetsPhi.Write();
  allJetsGctEta.Write();
  metValue.Write();
  metAngle.Write();
}

///event processor
void L1GctBasicHistogrammer::fillHistograms(const GctOutputData gct)
{
  L1GctJetCandCollection::const_iterator jet;
  for (unsigned i=0; i<gct.jets.size(); ++i) {
    for (jet=gct.jets.at(i)->begin(); jet!=gct.jets.at(i)->end(); ++jet) {
      if (!jet->empty()) {
	allJetsRank.Fill(jet->rank());
	allJetsEta.Fill (jet->etaIndex());
	allJetsPhi.Fill (jet->phiIndex());
	allJetsGctEta.Fill(gctEta(*jet));
      }
    }
  }
  metValue.Fill(gct.etMiss->et());
  metAngle.Fill(gct.etMiss->phi());
}

