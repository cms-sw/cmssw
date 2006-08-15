#ifndef L1GCTHISTOGRAMMER_H
#define L1GCTHISTOGRAMMER_H

/** \class L1GctHistogrammer
 *
 * Books and fills histograms to check GCT
 * and calo trigger performance
 *
 * \author Greg Heath
 *
 * \date August 2006
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

#include <vector>
#include <string>
#include "TFile.h"

typedef struct {
  std::vector<edm::Handle<L1GctEmCandCollection> > electrons;
  std::vector<edm::Handle<L1GctJetCandCollection> > jets;
  edm::Handle<L1GctEtTotal> etTotal;
  edm::Handle<L1GctEtHad> etHad;
  edm::Handle<L1GctEtMiss> etMiss;
} GctOutputData;

class L1GctHistogrammer {

 public:

  ///constructor
  L1GctHistogrammer(TFile* tf=0, const std::string dir="default");

  ///destructor
  virtual ~L1GctHistogrammer();

  ///event processor
  virtual void fillHistograms(const GctOutputData gct)=0;

 protected:

  static const int      NRANK,   NETA,   NPHI,   NGCTETA;
  static const double MINRANK, MINETA, MINPHI, MINGCTETA;
  static const double MAXRANK, MAXETA, MAXPHI, MAXGCTETA;
  static const int      NGCTMETVALUE,   NGCTMETPHI;
  static const double MINGCTMETVALUE, MINGCTMETPHI;
  static const double MAXGCTMETVALUE, MAXGCTMETPHI;

  const char* HistogramDirectory() const { return m_dir.c_str(); }
  bool   setHistogramDirectory() { return ( resetHistogramDirectory() ? m_file->cd(HistogramDirectory()) : false ); }
  bool resetHistogramDirectory() { return ( (m_file==0) ? false : m_file->cd() ); }

  L1GctJetCand topJet(const GctOutputData gct);
  unsigned gctEta(L1GctJetCand jet);

 private:

  TFile* m_file;
  const std::string m_dir;

};


template<class prodType>
class L1GctCorrelator : public L1GctHistogrammer {

 public:

  ///constructor
  L1GctCorrelator(TFile* tf, const std::string dir="default") : L1GctHistogrammer(tf, dir) {}

  ///destructor
  virtual ~L1GctCorrelator() {}

  ///event processor
  virtual void fillHistograms(const GctOutputData gct) = 0;

  ///set the product to be correlated with gct output
  void setInputProduct(const edm::Handle<prodType> prod) { m_prod = *prod; }

 protected:

  prodType prod() const { return m_prod; }

 private:

  prodType m_prod;

};

#endif
