//
// $Id: Centrality.h,v 1.2 2008/07/04 13:54:04 yilmaz Exp $
//

#ifndef DataFormats_Centrality_h
#define DataFormats_Centrality_h

#include <string>
#include <vector>

namespace reco { class Centrality {
public:
  Centrality(double eHF=0, double eCASTOR=0, double eZDC=0, int ZDCHits=0);
  virtual ~Centrality();

  double    raw()               const { return value_; }
  double    HFEnergy()          const { return HFEnergy_; } 
  double    CASTOREnergy()      const { return CASTOREnergy_; } 
  double    ZDCEnergy()         const { return ZDCEnergy_; } 
  int       ZDCHitCounts()      const { return ZDCHitCounts_; } 

private:
  double    HFEnergy_  ;
  double    CASTOREnergy_  ;
  double    ZDCEnergy_  ;
  int       ZDCHitCounts_  ;

  //Preparaion for new format:
  std::string label_;
  double value_;

};

 typedef std::vector<reco::Centrality> CentralityCollection;

}

#endif 


