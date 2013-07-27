//
// $Id: Centrality.h,v 1.12 2010/08/23 16:42:01 nart Exp $
//

#ifndef DataFormats_Centrality_h
#define DataFormats_Centrality_h

#include <string>
#include <vector>

namespace reco { class Centrality {
public:
  Centrality(double d=0, std::string label = "");
  virtual ~Centrality();

  friend class CentralityProducer;

  double    raw()               const { return value_; }
  std::string label()           const { return label_; }

  double EtHFhitSum() const {return etHFhitSumPlus_ + etHFhitSumMinus_;}
  double EtHFhitSumPlus() const {return etHFhitSumPlus_;}
  double EtHFhitSumMinus() const {return etHFhitSumMinus_;}
  double EtHFtowerSum() const {return etHFtowerSumPlus_ + etHFtowerSumMinus_;}
  double EtHFtowerSumPlus() const {return etHFtowerSumPlus_;}
  double EtHFtowerSumMinus() const {return etHFtowerSumMinus_;}
  double EtHFtruncated() const {return etHFtruncatedPlus_ + etHFtruncatedMinus_;}
  double EtHFtruncatedPlus() const {return etHFtruncatedPlus_;}
  double EtHFtruncatedMinus() const {return etHFtruncatedMinus_;}
  double EtEESum() const {return etEESumPlus_ + etEESumMinus_;}
  double EtEESumPlus() const {return etEESumPlus_;}
  double EtEESumMinus() const {return etEESumMinus_;}
  double EtEEtruncated() const {return etEEtruncatedPlus_ + etEEtruncatedMinus_;}
  double EtEEtruncatedPlus() const {return etEEtruncatedPlus_;}
  double EtEEtruncatedMinus() const {return etEEtruncatedMinus_;}
  double EtEBSum() const {return etEBSum_;}
  double EtEBtruncated() const {return etEBtruncated_;}
  double EtEcalSum() const {return etEBSum_ + EtEESum();}
  double EtEcaltruncated() const {return etEBtruncated_ + EtEEtruncated();}
  double multiplicityPixel() const {return pixelMultiplicity_;}
  double Ntracks() const {return trackMultiplicity_;}
  double NtracksPtCut() const {return ntracksPtCut_;}
  double NtracksEtaCut() const {return ntracksEtaCut_;}
  double NtracksEtaPtCut() const {return ntracksEtaPtCut_;}
  double NpixelTracks() const {return nPixelTracks_;}

  double zdcSum() const {return zdcSumPlus_ + zdcSumMinus_;}
  double zdcSumPlus() const {return zdcSumPlus_;}
  double zdcSumMinus() const {return zdcSumMinus_;}
  double EtMidRapiditySum() const {return etMidRapiditySum_;}

protected:
  double value_;
  std::string label_;

  double etHFhitSumPlus_;
  double etHFtowerSumPlus_;
  double etHFtruncatedPlus_;

  double etHFhitSumMinus_;
  double etHFtowerSumMinus_;
  double etHFtruncatedMinus_;

  double etEESumPlus_;
  double etEEtruncatedPlus_;
  double etEESumMinus_;
  double etEEtruncatedMinus_;
  double etEBSum_;
  double etEBtruncated_;

  double pixelMultiplicity_;
  double trackMultiplicity_;
  double zdcSumPlus_;
  double zdcSumMinus_;
  double etMidRapiditySum_;
  double ntracksPtCut_;
  double ntracksEtaCut_;
  double ntracksEtaPtCut_;
  double nPixelTracks_;

};

 typedef std::vector<reco::Centrality> CentralityCollection;

}

#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include "FWCore/Framework/interface/EventSetup.h"
const CentralityBins* getCentralityBinsFromDB(const edm::EventSetup& iSetup);



#endif 


