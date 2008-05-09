#ifndef BaseTagInfoPlotter_H
#define BaseTagInfoPlotter_H

#include <vector>
#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DQMOffline/RecoB/interface/BaseBTagPlotter.h"
// #include "RecoBTag/MCTools/interface/JetFlavour.h"

class BaseTagInfoPlotter : public BaseBTagPlotter {

 public:

  BaseTagInfoPlotter ( const TString & tagName, const EtaPtBin & etaPtBin) :
	    BaseBTagPlotter(tagName, etaPtBin) {};

  virtual ~BaseTagInfoPlotter () {};
  virtual void analyzeTag(const reco::BaseTagInfo * tagInfo, const int & jetFlavour);
  virtual void analyzeTag(const std::vector<const reco::BaseTagInfo *> &tagInfos, const int & jetFlavour);

  virtual void setEventSetup(const edm::EventSetup & setup);
  virtual vector<string> tagInfoRequirements() const;
  
} ;

#endif
