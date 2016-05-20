#include "JetMETCorrections/Type1MET/plugins/MultShiftMETcorrDBInputProducer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/JetMETObjects/interface/METCorrectorParameters.h"
#include "JetMETCorrections/Objects/interface/METCorrectionsRecord.h"

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
  
  mPayloadName    = cfg.getUntrackedParameter<std::string>("payloadName");
  mGlobalTag      = cfg.getUntrackedParameter<std::string>("globalTag");  
  //cfgCorrParameters_ = cfg.getParameter<std::vector<edm::ParameterSet> >("parameters");

  pflow_ = consumes<edm::View<reco::Candidate> >(cfg.getParameter< edm::InputTag >("srcPFlow") );
  vertices_ = consumes<edm::View<reco::Vertex> >( cfg.getParameter< edm::InputTag >("vertexCollection") );

  etaMin_.clear(); 
  etaMax_.clear();
  //varType_.clear(); 

  produces<CorrMETData>();


  /*******************
  for (std::vector<edm::ParameterSet>::const_iterator v = cfgCorrParameters_.begin(); v!=cfgCorrParameters_.end(); v++) {
    TString corrPxFormula = v->getParameter<std::string>("fx");
    TString corrPyFormula = v->getParameter<std::string>("fy");
    std::vector<double> corrPxParams = v->getParameter<std::vector<double> >("px");
    std::vector<double> corrPyParams = v->getParameter<std::vector<double> >("py");

    formula_x_.push_back( std::unique_ptr<TF1>(new TF1(std::string(moduleLabel_).append("_").append(v->getParameter<std::string>("name")).append("_corrPx").c_str(), v->getParameter<std::string>("fx").c_str()) ) );
    formula_y_.push_back( std::unique_ptr<TF1>(new TF1(std::string(moduleLabel_).append("_").append(v->getParameter<std::string>("name")).append("_corrPy").c_str(), v->getParameter<std::string>("fy").c_str()) ) );

    for (unsigned i=0; i<corrPxParams.size();i++) formula_x_.back()->SetParameter(i, corrPxParams[i]);
    for (unsigned i=0; i<corrPyParams.size();i++) formula_y_.back()->SetParameter(i, corrPyParams[i]);

    counts_.push_back(0);
    sumPt_.push_back(0.);
    etaMin_.push_back(v->getParameter<double>("etaMin"));
    etaMax_.push_back(v->getParameter<double>("etaMax"));
    type_.push_back(v->getParameter<int>("type"));
    varType_.push_back(v->getParameter<int>("varType"));
  }
  ***************/

}

MultShiftMETcorrDBInputProducer::~MultShiftMETcorrDBInputProducer()
{
}

void MultShiftMETcorrDBInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  // Get para.s from DB
  edm::ESHandle<METCorrectorParametersCollection> METCorParamsColl;
  //std::cout <<"Inspecting MET payload with label: "<< mPayloadName <<std::endl;
  es.get<METCorrectionsRecord>().get(mPayloadName,METCorParamsColl);

  // get the sections from Collection (pair of section and METCorr.Par class)
  std::vector<METCorrectorParametersCollection::key_type> keys;
  // save level to keys for each METParameter in METParameter collection
  METCorParamsColl->validKeys( keys );


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

  //for (unsigned i=0;i<counts_.size();i++) counts_[i]=0;
  //for (unsigned i=0;i<sumPt_.size();i++) sumPt_[i]=0.;

  edm::Handle<edm::View<reco::Candidate> > particleFlow;
  evt.getByToken(pflow_, particleFlow);


  //MM: loop over all constituent types and sum each correction
  std::auto_ptr<CorrMETData> metCorr(new CorrMETData());
  
  double corx=0.;
  double cory=0.;


  for ( std::vector<METCorrectorParametersCollection::key_type>::const_iterator ibegin = keys.begin(),
	  iend = keys.end(), ikey = ibegin; ikey != iend; ++ikey ) {
    //std::cout<<"--------------------------------------" << std::endl;
    //std::cout<<"Processing key = " << *ikey << std::endl;
    std::string sectionName= METCorParamsColl->findLabel(*ikey);
    //std::cout<<"object label: "<<sectionName<<std::endl;
    METCorrectorParameters const & METCorParams = (*METCorParamsColl)[*ikey];

    counts_ = 0;
    sumPt_  = 0;

    for (unsigned i = 0; i < particleFlow->size(); ++i) {
      const reco::Candidate& c = particleFlow->at(i);
      if (abs(c.pdgId())== translateTypeToAbsPdgId(reco::PFCandidate::ParticleType( METCorParams.definitions().ptclType() ))) {
        if ((c.eta()>METCorParams.record(0).xMin(0)) and (c.eta()<METCorParams.record(0).xMax(0))) {
          counts_ +=1;
          sumPt_  +=c.pt();
          continue;
        }
      }
    }
    //std::cout<<"counts: "<<counts_<<" sumPt: "<<sumPt_<<std::endl;
    //METCorParams.printScreen(sectionName);
    /**************
    if (mCreateTextFile)
      {
        if(METCorParamsColl->isXYshift(*ikey) )
        {
	  std::string outFileName(mGlobalTag+"_XYshift_"+mPayloadName+".txt");
	  std::cout<<"outFileName: "<<outFileName<<std::endl;
          //std::cout<<"Writing METCorrectorParameter to txt file: "<<mGlobalTag+"_XYshift_"+mPayloadName+".txt"<<std::endl;
          METCorParams.printFile(outFileName, sectionName);
        }
      }
  *****************/
    double val(0.);
    //std::cout<<"parVar: "<<METCorParams.definitions().parVar(0)<<std::endl;
    int parVar = METCorParams.definitions().parVar(0);
    //std::cout<<"int parVar: "<<parVar<<std::endl;

    if ( METCorParams.definitions().parVar(0) ==0) {
      val = counts_;
    } 
    if ( METCorParams.definitions().parVar(0) ==1) {
      val = ngoodVertices; 
    } 
    if ( METCorParams.definitions().parVar(0) ==2) {
      val = sumPt_;
    }
    //std::cout<<"val: "<<val<<std::endl;
    //std::cout<<"formula: "<<METCorParams.definitions().formula()<<std::endl;

    formula_x_ = new TF1("corrPx", METCorParams.definitions().formula().c_str());
    formula_y_ = new TF1("corrPy", METCorParams.definitions().formula().c_str());

    for( unsigned i(0); i<METCorParams.record(0).nParameters(); i++)
    {
      //std::cout<<"par("<<i<<"): "<<METCorParams.record(0).parameter(i)<<std::endl;
      formula_x_->SetParameter(i,METCorParams.record(0).parameter(i));
    }
    for( unsigned i(0); i<METCorParams.record(1).nParameters(); i++)
    {
      //std::cout<<"par("<<i<<"): "<<METCorParams.record(1).parameter(i)<<std::endl;
      formula_y_->SetParameter(i,METCorParams.record(1).parameter(i));
    }

    //std::cout<<"value from formula_x: "<<formula_x_->Eval(val)<<std::endl;
    //std::cout<<"value from formula_y: "<<formula_y_->Eval(val)<<std::endl;
    corx -= formula_x_->Eval(val);
    cory -= formula_y_->Eval(val);

  } //end loop over corrections

  //std::cout<<"corx: "<<corx<<"  cory: "<<cory<<std::endl;

  metCorr->mex = corx;
  metCorr->mey = cory;
  evt.put(metCorr, "");
  
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MultShiftMETcorrDBInputProducer);
