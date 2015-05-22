#include "PhysicsTools/TagAndProbe/interface/BaseTreeFiller.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "PhysicsTools/TagAndProbe/interface/ColinsSoperVariables.h"

#include <TList.h>
#include <TObjString.h>

tnp::ProbeVariable::~ProbeVariable() {}

tnp::ProbeFlag::~ProbeFlag() {}

void tnp::ProbeFlag::init(const edm::Event &iEvent) const {
    if (external_) {
        edm::Handle<edm::View<reco::Candidate> > view;
        iEvent.getByToken(srcToken_, view);
        passingProbes_.clear();
        for (size_t i = 0, n = view->size(); i < n; ++i) passingProbes_.push_back(view->refAt(i));
    }

}

void tnp::ProbeFlag::fill(const reco::CandidateBaseRef &probe) const {
    if (external_) {
        value_ = (std::find(passingProbes_.begin(), passingProbes_.end(), probe) != passingProbes_.end());
    } else {
        value_ = bool(cut_(*probe));
    }
}

tnp::BaseTreeFiller::BaseTreeFiller(const char *name, const edm::ParameterSet& iConfig, edm::ConsumesCollector & iC) {
    // make trees as requested
    edm::Service<TFileService> fs;
    tree_ = fs->make<TTree>(name,name);

    // add the branches
    addBranches_(tree_, iConfig, iC, "");

    // set up weights, if needed
    if (iConfig.existsAs<double>("eventWeight")) {
        weightMode_ = Fixed;
        weight_ = iConfig.getParameter<double>("eventWeight");
    } else if (iConfig.existsAs<edm::InputTag>("eventWeight")) {
        weightMode_ = External;
        weightSrcToken_ = iC.consumes<double>(iConfig.getParameter<edm::InputTag>("eventWeight"));
    } else {
        weightMode_ = None;
    }
    if (weightMode_ != None) {
        tree_->Branch("weight", &weight_, "weight/F");
    }
    storePUweight_ = iConfig.existsAs<edm::InputTag>("PUWeightSrc") ? true: false;
    if(storePUweight_) {
      PUweightSrcToken_   = iC.consumes<double>(iConfig.getParameter<edm::InputTag>("PUWeightSrc")); 
      tree_->Branch("PUweight", &PUweight_, "PUweight/F");
    }

    addRunLumiInfo_ = iConfig.existsAs<bool>("addRunLumiInfo") ? iConfig.getParameter<bool>("addRunLumiInfo") : false;
    if (addRunLumiInfo_) {
         tree_->Branch("run",  &run_,  "run/i");
         tree_->Branch("lumi", &lumi_, "lumi/i");
         tree_->Branch("event", &event_, "event/l");
    }
    addEventVariablesInfo_ = iConfig.existsAs<bool>("addEventVariablesInfo") ? iConfig.getParameter<bool>("addEventVariablesInfo") : false;
    if (addEventVariablesInfo_) {
      recVtxsToken_ = iC.consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
      beamSpotToken_ = iC.consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
      metToken_ = iC.consumes<reco::CaloMETCollection>(edm::InputTag("met"));
      tcmetToken_ = iC.consumes<reco::METCollection>(edm::InputTag("tcMet"));
      pfmetToken_ = iC.consumes<reco::PFMETCollection>(edm::InputTag("pfMet"));
      tree_->Branch("event_nPV"        ,&mNPV_                 ,"mNPV/I");
      tree_->Branch("event_met_calomet"    ,&mMET_                ,"mMET/F");
      tree_->Branch("event_met_calosumet"  ,&mSumET_              ,"mSumET/F");
      tree_->Branch("event_met_calometsignificance",&mMETSign_    ,"mMETSign/F");
      tree_->Branch("event_met_tcmet"    ,&mtcMET_                ,"mtcMET/F");
      tree_->Branch("event_met_tcsumet"  ,&mtcSumET_              ,"mtcSumET/F");
      tree_->Branch("event_met_tcmetsignificance",&mtcMETSign_    ,"mtcMETSign/F");
      tree_->Branch("event_met_pfmet"    ,&mpfMET_                ,"mpfMET/F");
      tree_->Branch("event_met_pfsumet"  ,&mpfSumET_              ,"mpfSumET/F");
      tree_->Branch("event_met_pfmetsignificance",&mpfMETSign_    ,"mpfMETSign/F");
      tree_->Branch("event_PrimaryVertex_x"  ,&mPVx_              ,"mPVx/F");
      tree_->Branch("event_PrimaryVertex_y"  ,&mPVy_              ,"mPVy/F");
      tree_->Branch("event_PrimaryVertex_z"  ,&mPVz_              ,"mPVz/F");
      tree_->Branch("event_BeamSpot_x"       ,&mBSx_              ,"mBSx/F");
      tree_->Branch("event_BeamSpot_y"       ,&mBSy_              ,"mBSy/F");
      tree_->Branch("event_BeamSpot_z"       ,&mBSz_              ,"mBSz/F");
    }

    ignoreExceptions_ = iConfig.existsAs<bool>("ignoreExceptions") ? iConfig.getParameter<bool>("ignoreExceptions") : false;
}

tnp::BaseTreeFiller::BaseTreeFiller(BaseTreeFiller &main, const edm::ParameterSet &iConfig, edm::ConsumesCollector && iC, const std::string &branchNamePrefix) :
  addRunLumiInfo_(false),
  addEventVariablesInfo_(false),
  tree_(0)
{
    addRunLumiInfo_ = main.addRunLumiInfo_;
    storePUweight_ = main.storePUweight_;
    addBranches_(main.tree_, iConfig, iC, branchNamePrefix);
}

void
tnp::BaseTreeFiller::addBranches_(TTree *tree, const edm::ParameterSet &iConfig, edm::ConsumesCollector & iC, const std::string &branchNamePrefix) {
    // set up variables
    edm::ParameterSet variables = iConfig.getParameter<edm::ParameterSet>("variables");
    //.. the ones that are strings
    std::vector<std::string> stringVars = variables.getParameterNamesForType<std::string>();
    for (std::vector<std::string>::const_iterator it = stringVars.begin(), ed = stringVars.end(); it != ed; ++it) {
        vars_.push_back(tnp::ProbeVariable(branchNamePrefix + *it, variables.getParameter<std::string>(*it)));
    }
    //.. the ones that are InputTags
    std::vector<std::string> inputTagVars = variables.getParameterNamesForType<edm::InputTag>();
    for (std::vector<std::string>::const_iterator it = inputTagVars.begin(), ed = inputTagVars.end(); it != ed; ++it) {
        vars_.push_back(tnp::ProbeVariable(branchNamePrefix + *it, iC.consumes<edm::ValueMap<float> >(variables.getParameter<edm::InputTag>(*it))));
    }
    // set up flags
    edm::ParameterSet flags = iConfig.getParameter<edm::ParameterSet>("flags");
    //.. the ones that are strings
    std::vector<std::string> stringFlags = flags.getParameterNamesForType<std::string>();
    for (std::vector<std::string>::const_iterator it = stringFlags.begin(), ed = stringFlags.end(); it != ed; ++it) {
        flags_.push_back(tnp::ProbeFlag(branchNamePrefix + *it, flags.getParameter<std::string>(*it)));
    }
    //.. the ones that are InputTags
    std::vector<std::string> inputTagFlags = flags.getParameterNamesForType<edm::InputTag>();
    for (std::vector<std::string>::const_iterator it = inputTagFlags.begin(), ed = inputTagFlags.end(); it != ed; ++it) {
        flags_.push_back(tnp::ProbeFlag(branchNamePrefix + *it, iC.consumes<edm::View<reco::Candidate> >(flags.getParameter<edm::InputTag>(*it))));
    }

    // then make all the variables in the trees
    for (std::vector<tnp::ProbeVariable>::iterator it = vars_.begin(), ed = vars_.end(); it != ed; ++it) {
        tree->Branch(it->name().c_str(), it->address(), (it->name()+"/F").c_str());
    }

    for (std::vector<tnp::ProbeFlag>::iterator it = flags_.begin(), ed = flags_.end(); it != ed; ++it) {
        tree->Branch(it->name().c_str(), it->address(), (it->name()+"/I").c_str());
    }

}

tnp::BaseTreeFiller::~BaseTreeFiller() { }

void tnp::BaseTreeFiller::init(const edm::Event &iEvent) const {
    run_  = iEvent.id().run();
    lumi_ = iEvent.id().luminosityBlock();
    event_ = iEvent.id().event();

    for (std::vector<tnp::ProbeVariable>::const_iterator it = vars_.begin(), ed = vars_.end(); it != ed; ++it) {
        it->init(iEvent);
    }
    for (std::vector<tnp::ProbeFlag>::const_iterator it = flags_.begin(), ed = flags_.end(); it != ed; ++it) {
        it->init(iEvent);
    }
    if (weightMode_ == External) {
        edm::Handle<double> weight;
        iEvent.getByToken(weightSrcToken_, weight);
        weight_ = *weight;
    }

    ///// ********** Pileup weight: needed for MC re-weighting for PU ************* 
     edm::Handle<std::vector<float> > weightPU;
     if(storePUweight_) {
       bool isPresent = iEvent.getByToken(PUweightSrcToken_, weightPU);
       if(isPresent) PUweight_ = (*weightPU).at(0);
       else PUweight_ = 1.0;
     }
     
    if (addEventVariablesInfo_) {
        /// *********** store some event variables: MET, SumET ******
        //////////// Primary vertex //////////////
        edm::Handle<reco::VertexCollection> recVtxs;
        iEvent.getByToken(recVtxsToken_,recVtxs);
        mNPV_ = 0;
        mPVx_ =  100.0;
        mPVy_ =  100.0;
        mPVz_ =  100.0;

        for(unsigned int ind=0;ind<recVtxs->size();ind++) {
          if (!((*recVtxs)[ind].isFake()) && ((*recVtxs)[ind].ndof()>4)
              && (fabs((*recVtxs)[ind].z())<=24.0) &&
              ((*recVtxs)[ind].position().Rho()<=2.0) ) {
            mNPV_++;
            if(mNPV_==1) { // store the first good primary vertex
              mPVx_ = (*recVtxs)[ind].x();
              mPVy_ = (*recVtxs)[ind].y();
              mPVz_ = (*recVtxs)[ind].z();
            }
          }
        }


        //////////// Beam spot //////////////
        edm::Handle<reco::BeamSpot> beamSpot;
        iEvent.getByToken(beamSpotToken_, beamSpot);
        mBSx_ = beamSpot->position().X();
        mBSy_ = beamSpot->position().Y();
        mBSz_ = beamSpot->position().Z();


        ////////////// CaloMET //////
        edm::Handle<reco::CaloMETCollection> met;
        iEvent.getByToken(metToken_,met);
        if (met->size() == 0) {
          mMET_   = -1;
          mSumET_ = -1;
          mMETSign_ = -1;
        }
        else {
          mMET_   = (*met)[0].et();
          mSumET_ = (*met)[0].sumEt();
          mMETSign_ = (*met)[0].significance();
        }

        /////// TcMET information /////
        edm::Handle<reco::METCollection> tcmet;
        iEvent.getByToken(tcmetToken_, tcmet);
        if (tcmet->size() == 0) {
          mtcMET_   = -1;
          mtcSumET_ = -1;
          mtcMETSign_ = -1;
        }
        else {
          mtcMET_   = (*tcmet)[0].et();
          mtcSumET_ = (*tcmet)[0].sumEt();
          mtcMETSign_ = (*tcmet)[0].significance();
        }

        /////// PfMET information /////
        edm::Handle<reco::PFMETCollection> pfmet;
        iEvent.getByToken(pfmetToken_, pfmet);
        if (pfmet->size() == 0) {
          mpfMET_   = -1;
          mpfSumET_ = -1;
          mpfMETSign_ = -1;
        }
        else {
          mpfMET_   = (*pfmet)[0].et();
          mpfSumET_ = (*pfmet)[0].sumEt();
          mpfMETSign_ = (*pfmet)[0].significance();
        }
    }

}

void tnp::BaseTreeFiller::fill(const reco::CandidateBaseRef &probe) const {
    for (std::vector<tnp::ProbeVariable>::const_iterator it = vars_.begin(), ed = vars_.end(); it != ed; ++it) {
        if (ignoreExceptions_)  {
            try{ it->fill(probe); } catch(cms::Exception &ex ){}
        } else {
            it->fill(probe);
        }
    }

    for (std::vector<tnp::ProbeFlag>::const_iterator it = flags_.begin(), ed = flags_.end(); it != ed; ++it) {
        if (ignoreExceptions_)  {
            try{ it->fill(probe); } catch(cms::Exception &ex ){}
        } else {
            it->fill(probe);
        }
    }
    if (tree_) tree_->Fill();
}
void tnp::BaseTreeFiller::writeProvenance(const edm::ParameterSet &pset) const {
    TList *list = tree_->GetUserInfo();
    list->Add(new TObjString(pset.dump().c_str()));
}
