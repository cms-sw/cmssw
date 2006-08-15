#include "L1Trigger/L1GctAnalyzer/interface/L1GctHistogrammer.h"


//--------------Histogrammer base class---------------------------
//
///constructor
L1GctHistogrammer::L1GctHistogrammer(TFile* tf, const std::string dir) :
  m_file(tf),
  m_dir(dir)
{
  m_file->cd();
  m_file->mkdir(HistogramDirectory(), HistogramDirectory());
}

///destructor
L1GctHistogrammer::~L1GctHistogrammer()
{
}

//-------------------------------------------------------------
//
// Protected member functions, for use by histogrammers

///Find the top jet in the event
L1GctJetCand
L1GctHistogrammer::topJet(const GctOutputData gct)
{
  L1GctJetCand result;
  for (unsigned i=0; i<gct.jets.size(); ++i) {
    L1GctJetCandCollection::const_iterator jet=gct.jets.at(i)->begin();
    if (jet->rank()>result.rank()) {
      result = *jet;
    }
  }
  return result;
}

//Find the global eta from the encoded hardware eta
unsigned
L1GctHistogrammer::gctEta(L1GctJetCand jet)
{
  unsigned rctEta = (jet.etaIndex() & 0x7) + (jet.isForward() ? 7 : 0);
  return ((jet.etaSign()==0) ? (rctEta+11) : (10-rctEta));
}

//-------------------------------------------------------------
//
// Protected constants

// Histogram bin numbers and ranges for GCT quantities
const int L1GctHistogrammer::NRANK=64; const double L1GctHistogrammer::MINRANK=0.; const double L1GctHistogrammer::MAXRANK=64.;
const int L1GctHistogrammer::NETA =14; const double L1GctHistogrammer::MINETA =0.; const double L1GctHistogrammer::MAXETA =14.;
const int L1GctHistogrammer::NPHI =18; const double L1GctHistogrammer::MINPHI =0.; const double L1GctHistogrammer::MAXPHI =18.;

const int L1GctHistogrammer::NGCTETA=22; const double L1GctHistogrammer::MINGCTETA=0.; const double L1GctHistogrammer::MAXGCTETA=22.;

const int L1GctHistogrammer::NGCTMETVALUE=64; const double L1GctHistogrammer::MINGCTMETVALUE=0.; const double L1GctHistogrammer::MAXGCTMETVALUE=256.;
const int L1GctHistogrammer::NGCTMETPHI  =72; const double L1GctHistogrammer::MINGCTMETPHI  =0.; const double L1GctHistogrammer::MAXGCTMETPHI  =72.;

