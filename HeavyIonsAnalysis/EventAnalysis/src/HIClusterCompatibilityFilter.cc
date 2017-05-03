// Derived from HLTrigger/special/src/HLTPixelClusterShapeFilter.cc
// at version 7_5_0_pre3
//
//
// Author of Derived Filter:  Eric Appelt
//         Created:  Wed Apr 29, 2015
//
//

#include "HeavyIonsAnalysis/EventAnalysis/interface/HIClusterCompatibilityFilter.h"

HIClusterCompatibilityFilter::HIClusterCompatibilityFilter(const edm::ParameterSet& iConfig):
cluscomSrc_(consumes<reco::ClusterCompatibility>(iConfig.getParameter<edm::InputTag>("cluscomSrc"))),
minZ_(iConfig.getParameter<double>("minZ")),
maxZ_(iConfig.getParameter<double>("maxZ")),
clusterPars_(iConfig.getParameter< std::vector<double> >("clusterPars")),
nhitsTrunc_(iConfig.getParameter<int>("nhitsTrunc")),
clusterTrunc_(iConfig.getParameter<double>("clusterTrunc"))
{}

HIClusterCompatibilityFilter::~HIClusterCompatibilityFilter() {}

bool
HIClusterCompatibilityFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  bool accept = true;


  // obtain cluster compatibility scores
  Handle<reco::ClusterCompatibility> cc;
  iEvent.getByToken(cluscomSrc_, cc);

  double clusVtxQual = determineQuality(*cc, minZ_, maxZ_);
  double nPxlHits = cc->nValidPixelHits();

  // construct polynomial cut on cluster vertex quality vs. npixelhits
  double polyCut=0;
  for(unsigned int i=0; i < clusterPars_.size(); i++) {
    polyCut += clusterPars_[i]*std::pow((double)nPxlHits,(int)i);
  }
  if(nPxlHits < nhitsTrunc_)
    polyCut=0;             // don't use cut below nhitsTrunc_ pixel hits
  if(polyCut > clusterTrunc_ && clusterTrunc_ > 0)
    polyCut=clusterTrunc_; // no cut above clusterTrunc_

  if (clusVtxQual < polyCut) accept = false;

  // return with final filter decision
  return accept;

}

void
HIClusterCompatibilityFilter::beginJob()
{
}

void
HIClusterCompatibilityFilter::endJob()
{
}

double
HIClusterCompatibilityFilter::determineQuality(const reco::ClusterCompatibility & cc,
                                               double minZ, double maxZ) 
{
  // will compare cluster compatibility at a determined best 
  // z position to + and - 10 cm from the best position
  float best_z = 0.;
  int best_n= 0.,low_n = 0.,high_n = 0.;


  // look for best vertex z position within zMin to zMax range
  // best position is determined by maximum nHit with 
  // chi used for breaking a tie
  int nhits_max = 0;
  double chi_max = 1e+9;
  for( int i=0; i<cc.size(); i++ )
  {
    if( cc.z0(i) > maxZ || cc.z0(i) < minZ ) continue;
    if(cc.nHit(i) == 0) continue;
    if(cc.nHit(i) > nhits_max) {
      chi_max = 1e+9;
      nhits_max = cc.nHit(i);
    }
    if(cc.nHit(i) >= nhits_max && cc.chi(i) < chi_max) {
      chi_max = cc.chi(i);
      best_z = cc.z0(i); best_n = cc.nHit(i);
    }
  }

  // find compatible clusters at + or - 10 cm of the best, 
  // or get as close as possible in terms of z position.
  double low_target = best_z - 10.0;
  double high_target = best_z + 10.0;
  double low_match = 1000., high_match = 1000.;
  for( int i=0; i<cc.size(); i++ )
  {  
    if( fabs(cc.z0(i)-low_target) < low_match )
    {
       low_n = cc.nHit(i); 
       low_match = fabs(cc.z0(i)-low_target);
    }
    if( fabs(cc.z0(i)-high_target) < high_match )
    {
       high_n = cc.nHit(i); 
       high_match = fabs(cc.z0(i)-high_target);
    }
  }

  // determine vertex compatibility quality score
  double clusVtxQual=0.0;
  if ((low_n+high_n)> 0)
    clusVtxQual = (2.0*best_n)/(low_n+high_n);  // A/B
  else if (best_n > 0)
    clusVtxQual = 1000.0;                      // A/0 (set to arbitrarily large number)
  else
    clusVtxQual = 0;   

  return clusVtxQual;

}

void
HIClusterCompatibilityFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(HIClusterCompatibilityFilter);
