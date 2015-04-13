#ifndef RecoBTag_SoftLepton_ElectronTagger_h
#define RecoBTag_SoftLepton_ElectronTagger_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/MvaSoftElectronEstimator.h"
#include <mutex>

/** \class ElectronTagger
 *
 *
 *  \author P. Demin - UCL, Louvain-la-Neuve - Belgium
 *
 */

class ElectronTagger : public JetTagComputer {
public:

  /// explicit ctor 
 ElectronTagger(const edm::ParameterSet & );
  virtual float discriminator(const TagInfoHelper & tagInfo) const override;
//  std::vector<string> vecstr;
//  string path_mvaWeightFileEleID;
private:
  btag::LeptonSelector m_selector;
  edm::FileInPath WeightFile;
  mutable std::mutex m_mutex;
  std::unique_ptr<MvaSoftEleEstimator> mvaID;
};

ElectronTagger::ElectronTagger(const edm::ParameterSet & configuration):
    m_selector(configuration)
  {
	uses("seTagInfos");
	WeightFile=configuration.getParameter<edm::FileInPath>("weightFile");
	mvaID.reset(new MvaSoftEleEstimator(WeightFile.fullPath()));
  }

#endif
