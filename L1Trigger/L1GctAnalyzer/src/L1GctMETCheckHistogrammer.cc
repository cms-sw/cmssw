#include "L1Trigger/L1GctAnalyzer/interface/L1GctMETCheckHistogrammer.h"


//--------------Histogram checks against reconstructed missing Et---------------------------
//
using namespace reco;

const    int L1GctMETCheckHistogrammer::NGENMETVALUE=  NGCTMETVALUE;
const double L1GctMETCheckHistogrammer::MINGENMETVALUE=MINGCTMETVALUE;
const double L1GctMETCheckHistogrammer::MAXGENMETVALUE=MAXGCTMETVALUE; 
const    int L1GctMETCheckHistogrammer::NGENMETPHI=    NGCTMETPHI;
const double L1GctMETCheckHistogrammer::MINGENMETPHI=-M_PI;
const double L1GctMETCheckHistogrammer::MAXGENMETPHI= M_PI; 

///constructor
L1GctMETCheckHistogrammer::L1GctMETCheckHistogrammer(TFile* tf, const std::string dir) :
  L1GctCorrelator<reco::METCollection>(tf, dir),
  missingEtValueGCTVsGen( "missingEtValueGCTVsGen", "Gct missing et value vs generated MET",
			  NGCTMETVALUE, MINGCTMETVALUE, MAXGCTMETVALUE,
			  NGENMETVALUE, MINGENMETVALUE, MAXGENMETVALUE),
  missingEtPhiGCTVsGen  ( "missingEtPhiGCTVsGen",   "Gct missing et phi vs generated MET phi",
			  NGCTMETPHI, MINGCTMETPHI, MAXGCTMETPHI,
			  NGENMETPHI, MINGENMETPHI, MAXGENMETPHI),
  missingEtRatioVsGCTJetEta ( "missingEtRatioVsGCTJetEta", "Ratio GCT MET/gen vs GCT jet eta",
			      NGCTETA, MINGCTETA, MAXGCTETA,
			      50, 0., 5.)
{
}

///destructor
L1GctMETCheckHistogrammer::~L1GctMETCheckHistogrammer()
{
  if (setHistogramDirectory()) {
    missingEtValueGCTVsGen.Write();
    missingEtPhiGCTVsGen.Write();
    missingEtRatioVsGCTJetEta.Write();
  }
}

///event processor
void L1GctMETCheckHistogrammer::fillHistograms(const GctOutputData gct)
{
  METCollection::const_iterator gMET=prod().begin();
  missingEtValueGCTVsGen.Fill(gct.etMiss->et(),  gMET->sumEt());
  missingEtPhiGCTVsGen.Fill  (gct.etMiss->phi(), gMET->phi());
  missingEtRatioVsGCTJetEta.Fill (gctEta(topJet(gct)), (gct.etMiss->et()/gMET->sumEt()));
}
