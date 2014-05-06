#ifndef BaseTagInfoPlotter_H
#define BaseTagInfoPlotter_H

#include <vector>
#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DQMOffline/RecoB/interface/BaseBTagPlotter.h"

class BaseTagInfoPlotter : public BaseBTagPlotter {

 public:

  BaseTagInfoPlotter ( const std::string & tagName, const EtaPtBin & etaPtBin) :
	    BaseBTagPlotter(tagName, etaPtBin) {};

  virtual ~BaseTagInfoPlotter () {};
  virtual void analyzeTag(const reco::BaseTagInfo * tagInfo, const double & jec, const int & jetFlavour);
  virtual void analyzeTag(const std::vector<const reco::BaseTagInfo *> &tagInfos, const double & jec, const int & jetFlavour);
  virtual void analyzeTag(const reco::BaseTagInfo * tagInfo, const double & jec, const int & jetFlavour, const float & w);
  virtual void analyzeTag(const std::vector<const reco::BaseTagInfo *> &tagInfos, const double & jec, const int & jetFlavour, const float & w);

  virtual void setEventSetup(const edm::EventSetup & setup);
  virtual std::vector<std::string> tagInfoRequirements() const;
  
} ;

#endif
