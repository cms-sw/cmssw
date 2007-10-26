#ifndef TauMETAlgo_h
#define TauMETAlgo_h

/** \class TauMETAlgo
 *
 * Correct MET for taus in the events.
 *
 * \version   1st Version August 30, 2007
 ************************************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <math.h>
#include <vector>
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CorrMETData.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

//typedef std::vector<CorrMETData> CorrMETDataCollection;
typedef math::XYZTLorentzVector LorentzVector;
typedef math::XYZPoint Point;

using namespace std;
class TauMETAlgo 
{
 public:
  TauMETAlgo();
  virtual ~TauMETAlgo();
  virtual void run(edm::Event&, const edm::EventSetup&,   
		   edm::Handle<PFJetCollection>,edm::Handle<CaloJetCollection>,
                   const JetCorrector&,bool,
		   double,METCollection* corrMET);
};

#endif

