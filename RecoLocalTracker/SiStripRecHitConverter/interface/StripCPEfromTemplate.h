#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTemplate_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTemplate_H

#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripTemplate.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripTemplateReco.h"


class StripCPEfromTemplate : public StripCPE 
{

 public:
  

  StripClusterParameterEstimator::LocalValues 
    localParameters( const SiStripCluster&, const GeomDetUnit&, const LocalTrajectoryParameters&) const;

  StripCPEfromTemplate( edm::ParameterSet & conf, 
			  const MagneticField& mag, 
			  const TrackerGeometry& geom, 
			  const SiStripLorentzAngle& lorentz,
			  const SiStripBackPlaneCorrection& backplaneCorrections,
			  const SiStripConfObject& confObj,
			  const SiStripLatency& latency) 
    : StripCPE(conf, mag, geom, lorentz, backplaneCorrections, confObj, latency ),
    use_template_reco( conf.getParameter<bool>("UseTemplateReco") ),
    template_reco_speed( conf.getParameter<int>("TemplateRecoSpeed") ),
    use_strip_split_cluster_errors( conf.getParameter<bool>("UseStripSplitClusterErrors") )
    {
      templ.pushfile( 11 );
      templ.pushfile( 12 );
      templ.pushfile( 13 );
      templ.pushfile( 14 );
      templ.pushfile( 15 );
      templ.pushfile( 16 );

      //cout << "STRIPS: (int)use_template_reco = " << (int)use_template_reco << endl;
      //cout << "template_reco_speed    = " << template_reco_speed    << endl;
      //cout << "(int)use_strip_split_cluster_errors = " << (int)use_strip_split_cluster_errors << endl;
    }
  
 private:

  mutable SiStripTemplate templ;
 
  bool use_template_reco;
  int template_reco_speed;
  bool use_strip_split_cluster_errors;

};
#endif
