#ifndef HLTMCTRUTH_H
#define HLTMCTRUTH_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/METReco/interface/CaloMETCollection.h"

typedef std::vector<std::string> MyStrings;

/** \class HLTMCtruth
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */
class HLTMCtruth {
public:
  HLTMCtruth(); 

  void setup(const edm::ParameterSet& pSet, TTree* tree);

  /** Analyze the Data */
  void analyze(const edm::Handle<reco::CandidateView> & mctruth,
	       const double        & pthat,
	       const edm::Handle<std::vector<SimTrack> > & simTracks,
	       const edm::Handle<std::vector<SimVertex> > & simVertices,
	       TTree* tree);

private:

  // Tree variables
  float *mcvx, *mcvy, *mcvz, *mcpt, *mceta, *mcphi;
  int *mcpid, *mcstatus;
  int nmcpart,nmu3,nel3,nab,nbb,nwenu,nwmunu,nzee,nzmumu;
  float pthatf;
  float ptEleMax,ptMuMax;
  // input variables
  bool _Monte,_Debug;

};

#endif
