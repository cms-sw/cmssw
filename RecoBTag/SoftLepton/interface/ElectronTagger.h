#ifndef RecoBTag_SoftLepton_ElectronTagger_h
#define RecoBTag_SoftLepton_ElectronTagger_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/MvaSoftElectronEstimator.h"
#include "TRandom3.h"

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
 ~ElectronTagger(); 
  virtual float discriminator(const TagInfoHelper & tagInfo) const;
//  std::vector<string> vecstr;
//  string path_mvaWeightFileEleID;
private:
  btag::LeptonSelector m_selector;
  TRandom3* random;
  edm::FileInPath WeightFile;
  MvaSoftEleEstimator* mvaID;
};

ElectronTagger::ElectronTagger(const edm::ParameterSet & configuration):
    m_selector(configuration)
  {
	uses("seTagInfos");
        random=new TRandom3();
	WeightFile=configuration.getParameter<edm::FileInPath>("weightFile");
	mvaID=new MvaSoftEleEstimator(WeightFile.fullPath());
  }


ElectronTagger::~ElectronTagger() {
        delete random;
	delete mvaID;
  }
#endif
