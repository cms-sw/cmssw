#ifndef BaseTagInfoPlotter_H
#define BaseTagInfoPlotter_H

#include <vector>
#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DQMOffline/RecoB/interface/BaseBTagPlotter.h"

class BaseTagInfoPlotter: public BaseBTagPlotter {

 public:

  BaseTagInfoPlotter(const std::string & tagName, const EtaPtBin & etaPtBin) :
	    BaseBTagPlotter(tagName, etaPtBin) {};

  ~BaseTagInfoPlotter() override {};
  virtual void analyzeTag(const reco::BaseTagInfo * tagInfo, double jec, int jetFlavour, float w=1);
  virtual void analyzeTag(const std::vector<const reco::BaseTagInfo *> &tagInfos, double jec, int jetFlavour, float w=1);

  virtual void setEventSetup(const edm::EventSetup & setup);
  virtual std::vector<std::string> tagInfoRequirements() const;
  
} ;

#endif
