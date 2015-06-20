#ifndef ComphepSingletopFilterPy8_h
#define ComphepSingletopFilterPy8_h


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TH1F.h"
#include "TTree.h"
#include "TDirectory.h"
//
// class declaration
//


class ComphepSingletopFilterPy8 : public edm::EDFilter {
public:
    explicit ComphepSingletopFilterPy8(const edm::ParameterSet&);
    ~ComphepSingletopFilterPy8();
private:
    virtual void beginJob() ;
    virtual bool filter(edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;
private:
//     edm::InputTag hepMCProductTag;
    
    TString outputFileName;
    double ptsep;
    bool iWriteFile, isPythia8;

    unsigned int read22, read23, pass22, pass23, hardLep;
    TFile * moutFile;
    TTree * Matching;
    Double_t pt_add_b;
    Double_t pt_add_b_FSR;
    Double_t eta_add_b;
    Double_t eta_add_b_FSR;
    Double_t pt_t_b;
    Double_t eta_t_b;
    Double_t pt_t_b_FSR;
    Double_t eta_t_b_FSR;
    Double_t pt_l;
    Double_t eta_l;
    Double_t pt_l_FSR;
    Double_t eta_l_FSR;
    Double_t pt_top;
    Double_t pt_top_FSR;
    Double_t eta_top;
    Double_t eta_top_FSR;
    Double_t Cos_CargedLep_LJet;
    Double_t Cos_CargedLep_LJet_FSR;
    Double_t eta_light_q;
    Double_t pt_light_q;
    Double_t eta_light_q_FSR;
    Double_t pt_light_q_FSR;
};

#endif
