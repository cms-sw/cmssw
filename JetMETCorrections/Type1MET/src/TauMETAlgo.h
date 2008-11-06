#ifndef TauMETAlgo_h
#define TauMETAlgo_h

// Authors: Alfredo Gurrola, Chi Nhan Nguyen

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <math.h>
#include <vector>
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/TauReco/interface/PFTau.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CorrMETData.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

typedef math::XYZTLorentzVector LorentzVector;
typedef math::XYZPoint Point;

using namespace std;
class TauMETAlgo 
{
 public:
  TauMETAlgo();
  virtual ~TauMETAlgo();

  virtual void run(edm::Event&, const edm::EventSetup&,
		   edm::Handle<PFTauCollection>,edm::Handle<CaloJetCollection>,double,double,
                   const JetCorrector&,const std::vector<CaloMET>&,double,double,double,
                   bool,double,bool,double,bool,double,bool,std::vector<CaloMET>* corrMET);

  virtual void run(edm::Event&, const edm::EventSetup&,
		   edm::Handle<PFTauCollection>,edm::Handle<CaloJetCollection>,double,double,
                   const JetCorrector&,const std::vector<MET>&,double,double,double,
                   bool,double,bool,double,bool,double,bool,std::vector<MET>* corrMET);

};

#endif

