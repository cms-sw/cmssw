/** \class PAPileUpVertexFilter
 *
 *
 *  This class is an EDFilter for rejecting  pPb events with multiple collisions
 *  (pileup). This is performed by looking at the characteristics of the
 *  reconstructed vertices.
 *
 *  $Date: 2013/02/02 22:54:49 $
 *  $Revision: 1.7 $
 *
 *  \author E. Appelt - Vanderbilt University, Wei Li - Rice University
 *
 */


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>
#include <vector>
#include <algorithm>

#include "TF2.h"

#include <DataFormats/VertexReco/interface/Vertex.h>
#include <DataFormats/VertexReco/interface/VertexFwd.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class PAPileUpVertexFilter : public edm::EDFilter {
public:
  explicit PAPileUpVertexFilter(const edm::ParameterSet&);
  ~PAPileUpVertexFilter();
  virtual void endJob() ;

  virtual bool filter(edm::Event&, const edm::EventSetup&);

private:

  edm::InputTag vtxSrc_;
  bool doDzNtrkCut_;
  bool doDxyDzCut_;
  bool doSurfaceCut_;
  double dxyVeto_;
  double dzVeto_;
  double dxyDzCutPar0_;
  double dxyDzCutPar1_;
  double surfaceMinDzEval_;
  std::vector<double> dzCutByNtrk_;
  std::string surfaceFunctionString_;
  std::vector<double> surfaceCutParameters_;
  TF2* func2D_;


};


PAPileUpVertexFilter::PAPileUpVertexFilter(const edm::ParameterSet& iConfig) :
  vtxSrc_(iConfig.getParameter<edm::InputTag>("vtxSrc")),
  doDzNtrkCut_(iConfig.getParameter<bool>("doDzNtrkCut")),
  doDxyDzCut_(iConfig.getParameter<bool>("doDxyDzCut")),
  doSurfaceCut_(iConfig.getParameter<bool>("doSurfaceCut")),
  dxyVeto_(iConfig.getParameter<double>("dxyVeto")),
  dzVeto_(iConfig.getParameter<double>("dzVeto")),
  dxyDzCutPar0_(iConfig.getParameter<double>("dxyDzCutPar0")),
  dxyDzCutPar1_(iConfig.getParameter<double>("dxyDzCutPar1")),
  surfaceMinDzEval_(iConfig.getParameter<double>("surfaceMinDzEval")),
  dzCutByNtrk_(iConfig.getParameter<std::vector<double> >("dzCutByNtrk")),
  surfaceFunctionString_(iConfig.getParameter<std::string>("surfaceFunctionString")),
  surfaceCutParameters_(iConfig.getParameter<std::vector<double> >("surfaceCutParameters"))
{
  func2D_ = new TF2("func2D",surfaceFunctionString_.c_str(),0,50.0,0,500);
  if( (int)surfaceCutParameters_.size() == func2D_->GetNpar())
  {
    for(unsigned int i=0;i<surfaceCutParameters_.size();i++) func2D_->SetParameter(i,surfaceCutParameters_[i]);
  }
  else
  {
    // really need to quit with an error that the configuration is misformed
    for(unsigned int i=0;i<surfaceCutParameters_.size();i++) func2D_->SetParameter(i,0);
  }
}

PAPileUpVertexFilter::~PAPileUpVertexFilter()
{
  delete func2D_;
}

bool
PAPileUpVertexFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  bool accepted = true;

  Handle<std::vector<reco::Vertex> > vcol;
  iEvent.getByLabel(vtxSrc_, vcol);

  std::vector<reco::Vertex> vsorted = *vcol;
  // sort the vertcies by number of tracks in descending order
  std::sort( vsorted.begin(), vsorted.end(),
	     []( reco::Vertex a, reco::Vertex b)
	     {
	       return  a.tracksSize() > b.tracksSize() ? true : false ;
	     });

  // check additional PVs
  for (unsigned int i =1; i<vsorted.size(); i++)
  {
    double dz = fabs( vsorted[i].z() - vsorted[0].z() );
    double dx = fabs( vsorted[i].x() - vsorted[0].x() );
    double dy = fabs( vsorted[i].y() - vsorted[0].y() );
    double dxy  = sqrt ( dx*dx + dy*dy );
    double nTrk = vsorted[i].tracksSize();
    double nTrkLead = vsorted[0].tracksSize();

    // dxy Veto: only filter for small dxy
    if ( dxy > dxyVeto_ ) continue;

    // dz Veto: only filter for large dz
    if ( dz < dzVeto_ ) continue;

    // DzNtrkCut:  filter on dz by number of tracks
    if ( doDzNtrkCut_ )
    {
      for( unsigned int j=0; j<dzCutByNtrk_.size() ; j++)
      {
	if ( nTrk == (int)j && dz > dzCutByNtrk_[j] )
	  accepted = false;
      }
      // last dz value in vector is applied to all greater values of Ntrk
      if ( nTrk >= (int)dzCutByNtrk_.size()
	   && dz > dzCutByNtrk_[dzCutByNtrk_.size()-1] )
	accepted = false;
    }

    // DxyDzCut: filter on PVs above a diagonal line on the Dxy vs Dz plot
    if ( doDxyDzCut_ )
    {
      if( dz > dxyDzCutPar0_ + dxyDzCutPar1_ * dxy )
	accepted = false;
    }

    // surfaceCut: filter on PVs above a curved surface of the form Ntrk(dz, NtrkLead)
    if ( doSurfaceCut_ )
    {
      if(nTrk>func2D_->Eval(dz,nTrkLead) && dz>surfaceMinDzEval_) accepted = false;
      if(nTrk>func2D_->Eval(surfaceMinDzEval_,nTrkLead) && dz<=surfaceMinDzEval_) accepted = false;
    }

  }

  return accepted;
}

void
PAPileUpVertexFilter::endJob()
{
}

DEFINE_FWK_MODULE(PAPileUpVertexFilter);
