#ifndef JetTSelector_h
#define JetTSelector_h
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TSelector.h>
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Wrapper.h"

typedef reco::CandidateCollection JetCollection;

class JetTSelector : public TSelector {
public :
  JetTSelector( TTree *tree = 0 ) { }
  virtual ~JetTSelector() { }
  virtual Int_t Version() const { return 1; }
  virtual void  Begin( TTree *tree );
  virtual void  SlaveBegin( TTree *tree );
  virtual void  Init(TTree *tree);
  virtual bool  Notify();
  virtual bool  Process( long long entry );
  virtual void  SetOption( const char * option ) { fOption = option; }
  virtual void  SetObject( TObject *obj ) { fObject = obj; }
  virtual void  SetInputList( TList *input ) { fInput = input; }
  virtual TList * GetOutputList() const { return fOutput; }
  virtual void  SlaveTerminate();
  virtual void  Terminate();

private:
  TTree * chain;
  JetCollection jets;
  TBranch * jetsBranch;
  TH1F * h_et, * h_eta;

  JetTSelector(JetTSelector const&);
  JetTSelector operator=(JetTSelector const&);
};

#endif
