//
// $Id: Centrality.h,v 1.4 2009/08/18 14:45:31 yilmaz Exp $
//

#ifndef DataFormats_Centrality_h
#define DataFormats_Centrality_h

#include <string>
#include <vector>

namespace reco { class Centrality {
public:
  Centrality(double d=0, std::string label = "");
  virtual ~Centrality();

  double    raw()               const { return value_; }
  std::string label()           const { return label_; }

private:
  double value_;
  std::string label_;

};

 typedef std::vector<reco::Centrality> CentralityCollection;

}

#endif 


