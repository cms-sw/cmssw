//
// $Id: Centrality.h,v 1.3 2009/08/17 18:08:14 yilmaz Exp $
//

#ifndef DataFormats_Centrality_h
#define DataFormats_Centrality_h

#include <string>
#include <vector>

namespace reco { class Centrality {
public:
  Centrality(double d=0, std::string label = "");
  virtual ~Centrality();

  std::string label()           const { return label_; }
  double    raw()               const { return value_; }

private:
  std::string label_;
  double value_;

};

 typedef std::vector<reco::Centrality> CentralityCollection;

}

#endif 


