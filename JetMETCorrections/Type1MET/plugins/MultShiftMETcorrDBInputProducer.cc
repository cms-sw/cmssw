#include "JetMETCorrections/Type1MET/plugins/MultShiftMETcorrDBInputProducer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/JetMETObjects/interface/MEtXYcorrectParameters.h"
#include "CondFormats/JetMETObjects/interface/Utilities.h"
#include "JetMETCorrections/Objects/interface/MEtXYcorrectRecord.h"

#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/View.h"

#include <TString.h>

int MultShiftMETcorrDBInputProducer::translateTypeToAbsPdgId( reco::PFCandidate::ParticleType type ) {
  switch( type ) {
  case reco::PFCandidate::ParticleType::h: return 211; // pi+
  case reco::PFCandidate::ParticleType::e: return 11;
  case reco::PFCandidate::ParticleType::mu: return 13;
  case reco::PFCandidate::ParticleType::gamma: return 22;
  case reco::PFCandidate::ParticleType::h0: return 130; // K_L0
  case reco::PFCandidate::ParticleType::h_HF: return 1; // dummy pdg code
  case reco::PFCandidate::ParticleType::egamma_HF: return 2; // dummy pdg code
  case reco::PFCandidate::ParticleType::X:
  default: return 0;
  }
}


MultShiftMETcorrDBInputProducer::MultShiftMETcorrDBInputProducer(const edm::ParameterSet& cfg):
  moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  
  mPayloadName    = cfg.getParameter<std::string>("payloadName");
  mSampleType     = (cfg.exists("sampleType")) ? cfg.getParameter< std::string >("sampleType"): "MC";
  mIsData         = cfg.getParameter< bool >("isData");

  pflow_ = consumes<edm::View<reco::Candidate> >(cfg.getParameter< edm::InputTag >("srcPFlow") );
  vertices_ = consumes<edm::View<reco::Vertex> >( cfg.getParameter< edm::InputTag >("vertexCollection") );

  etaMin_.clear(); 
  etaMax_.clear();

  produces<CorrMETData>();


}

MultShiftMETcorrDBInputProducer::~MultShiftMETcorrDBInputProducer()
{
}

void MultShiftMETcorrDBInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  // Get para.s from DB
  edm::ESHandle<MEtXYcorrectParametersCollection> MEtXYcorParaColl;
  es.get<MEtXYcorrectRecord>().get(mPayloadName,MEtXYcorParaColl);

  // get the sections from Collection (pair of section and METCorr.Par class)
  std::vector<MEtXYcorrectParametersCollection::key_type> keys;
  // save level to keys for each METParameter in METParameter collection
  MEtXYcorParaColl->validKeys( keys );


  //get primary vertices
  edm::Handle<edm::View<reco::Vertex> > hpv;
  evt.getByToken( vertices_, hpv );
  if(!hpv.isValid()) {
    edm::LogError("MultShiftMETcorrDBInputProducer::produce") << "could not find vertex collection ";
  }
  std::vector<reco::Vertex> goodVertices;
  for (unsigned i = 0; i < hpv->size(); i++) {
    if ( (*hpv)[i].ndof() > 4 &&
       ( fabs((*hpv)[i].z()) <= 24. ) &&
       ( fabs((*hpv)[i].position().rho()) <= 2.0 ) )
       goodVertices.push_back((*hpv)[i]);
  }
  int ngoodVertices = goodVertices.size();

  edm::Handle<edm::View<reco::Candidate> > particleFlow;
  evt.getByToken(pflow_, particleFlow);


  //loop over all constituent types and sum each correction
  //std::unique_ptr<CorrMETData> metCorr(new CorrMETData());
  std::unique_ptr<CorrMETData> metCorr(new CorrMETData());
  
  double corx=0.;
  double cory=0.;


  // check DB
  for ( std::vector<MEtXYcorrectParametersCollection::key_type>::const_iterator ikey = keys.begin();
	   ikey != keys.end(); ++ikey ) {
    if(mIsData)
    {
      if(!MEtXYcorParaColl->isShiftData(*ikey) )
	throw cms::Exception("MultShiftMETcorrDBInputProducer::produce")
	  << "DB is not for Data. Set proper option: \"corrPfMetXYMultDB.isData\" !!\n";
    }else{
      if( MEtXYcorParaColl->isShiftData(*ikey) )
	throw cms::Exception("MultShiftMETcorrDBInputProducer::produce")
	  << "DB is for Data. Set proper option: \"corrPfMetXYMultDB.isData\" !!\n";
    }
  }

  for ( std::vector<MEtXYcorrectParametersCollection::key_type>::const_iterator ikey = keys.begin();
	   ikey != keys.end(); ++ikey ) {

    if( !mIsData){

      if(mSampleType == "MC"){
        if(!MEtXYcorParaColl->isShiftMC(*ikey)) continue;
      }else if(mSampleType == "DY"){
        if(!MEtXYcorParaColl->isShiftDY(*ikey)) continue;
      }else if(mSampleType == "TTJets"){
        if(!MEtXYcorParaColl->isShiftTTJets(*ikey)) continue;
      }else if(mSampleType == "WJets"){
        if(!MEtXYcorParaColl->isShiftWJets(*ikey)) continue;
      }else throw cms::Exception("MultShiftMETcorrDBInputProducer::produce")
	<< "SampleType: "<<mSampleType<<" is not reserved !!!\n";
    }

    std::string sectionName= MEtXYcorParaColl->findLabel(*ikey);
    MEtXYcorrectParameters const & MEtXYcorParams = (*MEtXYcorParaColl)[*ikey];

    counts_ = 0;
    sumPt_  = 0;

    for (unsigned i = 0; i < particleFlow->size(); ++i) {
      const reco::Candidate& c = particleFlow->at(i);
      if (abs(c.pdgId())== translateTypeToAbsPdgId(reco::PFCandidate::ParticleType( MEtXYcorParams.definitions().PtclType() ))) {
        if ((c.eta()>MEtXYcorParams.record(0).xMin(0)) and (c.eta()<MEtXYcorParams.record(0).xMax(0))) {
          counts_ +=1;
          sumPt_  +=c.pt();
          continue;
        }
      }
    }
    double val(0.);
    unsigned parVar = MEtXYcorParams.definitions().parVar(0);

    if ( parVar ==0) {
      val = counts_;

    }else if ( parVar ==1) { 
      val = ngoodVertices; 

    }else if ( parVar ==2) { 
      val = sumPt_;

    }else{
	throw cms::Exception("MultShiftMETcorrDBInputProducer::produce")
	  << "parVar: "<<parVar<<" is not reserved !!!\n";
    }

    formula_x_.reset( new TF1("corrPx", MEtXYcorParams.definitions().formula().c_str()));
    formula_y_.reset( new TF1("corrPy", MEtXYcorParams.definitions().formula().c_str()));

    for( unsigned i(0); i<MEtXYcorParams.record(0).nParameters(); i++)
    {
      formula_x_->SetParameter(i,MEtXYcorParams.record(0).parameter(i));
    }
    for( unsigned i(0); i<MEtXYcorParams.record(1).nParameters(); i++)
    {
      formula_y_->SetParameter(i,MEtXYcorParams.record(1).parameter(i));
    }

    corx -= formula_x_->Eval(val);
    cory -= formula_y_->Eval(val);

  } //end loop over corrections


  metCorr->mex = corx;
  metCorr->mey = cory;
  evt.put(std::move(metCorr), "");
  
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MultShiftMETcorrDBInputProducer);
