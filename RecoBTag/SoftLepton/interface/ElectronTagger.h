#ifndef RecoBTag_SoftLepton_ElectronTagger_h
#define RecoBTag_SoftLepton_ElectronTagger_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/MvaSoftElectronEstimator.h"

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

//  	path_mvaWeightFileEleID =edm::FileInPath("RecoBTag/SoftLepton/data/MVA_SE_BDT_weight.weights.xml" ).fullPath();
//  	vecstr.push_back(path_mvaWeightFileEleID);
//	mvaID_=new MvaSoftEleEstimator(vecstr);  
  }


ElectronTagger::~ElectronTagger() {
	delete mvaID;
  }
#endif
