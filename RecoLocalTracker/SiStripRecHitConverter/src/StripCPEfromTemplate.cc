
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTemplate.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"                                                           

// #include <iostream>

namespace {
  inline
  float stripErrorSquared(const unsigned N, const float uProj) {
    if( (float(N)-uProj) > 3.5f )  
      return float(N*N)/12.f;
    else {
      const float P1=-0.339f;
      const float P2=0.90f;
      const float P3=0.279f;
      const float uerr = P1*uProj*std::exp(-uProj*P2)+P3;
      return uerr*uerr;
    }
  }
}

StripClusterParameterEstimator::LocalValues 
StripCPEfromTemplate::localParameters( const SiStripCluster& cluster, 
					 const GeomDetUnit& det, 
					 const LocalTrajectoryParameters& ltp) const 
{
  StripClusterParameterEstimator::LocalValues final_lv;

  LocalPoint final_lp;
  LocalError final_le; 

  // Default reconstruction (needed if template reco fails)

  StripCPE::Param const& p = param( det );

  LocalVector track = ltp.momentum();
  track *= 
    ( track.z()<0 ) ?  fabs( p.thickness/track.z() ) : 
    ( track.z()>0 ) ? -fabs( p.thickness/track.z() ) :  
    p.maxLength/track.mag() ;

  const unsigned N = cluster.amplitudes().size();

  const float fullProjection = p.coveredStrips( track+p.drift, ltp.position());

  const float strip = cluster.barycenter() -  0.5f*(1.f-p.backplanecorrection) * fullProjection
    + 0.5f*p.coveredStrips(track, ltp.position());
  
  LocalPoint default_lp = p.topology->localPosition( strip, ltp.vector() );




 
  // Get the error of the split cluster.
  // If the cluster is not split, then the error is -99999.9.
  // The split cluster error is in microns and it has to be transformed into centimeteres and then in measurement units
  
  float uerr2 = -99999.9;
  float split_cluster_error = cluster.getSplitClusterError(); 

  float local_pitch = p.topology->localPitch( default_lp );

  if ( split_cluster_error > 0.0 && use_strip_split_cluster_errors )
    {
      //cout << endl;
      //cout << "Assign split rechit errors" << endl;
      // go from microns to cm and then to strip units...
      
      uerr2 = 
	(split_cluster_error/10000.0 / local_pitch ) * 
	(split_cluster_error/10000.0 / local_pitch ); 
    }
  else
    {
      //cout << endl;
      //cout << "Assign default rechit errors" << endl;
      uerr2 = stripErrorSquared( N, fabs(fullProjection) );
    }
  
  
  LocalError default_le = p.topology->localError( strip, uerr2, ltp.vector() );



  // Template reconstruction
 
  int ierr = 9999999; // failed template reco ( if ierr = 0, then, template reco was successful )
  float template_x_pos = -9999999.9;
  float template_x_err = -9999999.9;

  // int cluster_size = (int)cluster.amplitudes().size();
  
  // do not use template reco for huge clusters
  if ( use_template_reco )
    {    

      int id = -9999999;
      
      SiStripDetId ssdid = SiStripDetId( det.geographicalId() );

      int is_stereo = (int)( ssdid.stereo() ); 
      
      if      ( p.moduleGeom == 1 ) // IB1 
	{
	  if ( !is_stereo ) 
	    id = 11;
	  else
	    id = 12;
	}
      else if ( p.moduleGeom == 2 ) // IB2
	{
	  id = 13;
	}
      else if ( p.moduleGeom == 3 ) // OB1
	{
	  id = 16; 
	}
      else if ( p.moduleGeom == 4 ) // OB2
	{
	  if ( !is_stereo )
	    id = 14;
	  else
	    id = 15;
	}
      //else 
      //cout << "Do not use templates for strip modules other than IB1, IB2, OB1 and OB2" << endl;
      
      StripGeomDetUnit* stripdet = (StripGeomDetUnit*)(&det);
 
      if ( (id  > -9999999) && !(stripdet == 0) )
	{
	  // Variables needed by template reco
	  float cotalpha = -9999999.9; 
	  float cotbeta  = -9999999.9;
	  float locBy    = -9999999.9;
	  std::vector<float> vec_cluster_charge; 
	  
	  // Variables returned by template reco 
	  float xrec   = -9999999.9; 
	  float sigmax = -9999999.9; 
	  float probx  = -9999999.9; 
	  int qbin     = -9999999  ;
	  int speed    = template_reco_speed        ; 
	  float probQ  = -9999999.9; 
	  
	  
	  LocalVector lbfield = ( stripdet->surface() ).toLocal( magfield_.inTesla( stripdet->surface().position() ) ); 
	  locBy = lbfield.y();
	  	  
	  
	  LocalVector localDir = ltp.momentum()/ltp.momentum().mag();     
	  float locx = localDir.x();
	  float locy = localDir.y();
	  float locz = localDir.z();
	  cotalpha = locx/locz;
	  cotbeta  = locy/locz;
	  
	  
	  int cluster_size = (int)( (cluster.amplitudes()).size() );
	  for (int i=0; i<cluster_size; ++i)
	    {
	      vec_cluster_charge.push_back( (float)( (cluster.amplitudes())[i] ) );  
	    }
	  
	  
	  float measurement_position_first_strip_center = (float)(cluster.firstStrip()) + 0.5;
	  
	  LocalPoint local_position_first_strip_center 
	    = p.topology->localPosition( measurement_position_first_strip_center, ltp.vector() );
	  

          SiStripTemplate templ(theStripTemp_);
	  ierr = SiStripTemplateReco::StripTempReco1D( id, 
						       cotalpha, 
						       cotbeta, 
						       locBy, 
						       vec_cluster_charge, 
						       templ, 
						       xrec, 
						       sigmax, 
						       probx, 
						       qbin, 
						       speed, 
						       probQ ); 
	  
	  

	  // stripCPEtemplateProbability_ = probQ;
	  // stripCPEtemplateQbin_ = qbin; 

	 
	  template_x_pos = xrec / 10000.0  +  local_position_first_strip_center.x();
     
	  if ( split_cluster_error > 0.0 && use_strip_split_cluster_errors )
	    {
	      template_x_err = split_cluster_error/10000.0; 
	    }
	  else
	    {
	      template_x_err = sigmax/10000.0;
	    }


	} // if ( id  > -9999999 && !stripdet == 0 ) 
      
    } // if ( use_template_reco )


  if ( use_template_reco && ierr == 0 )
    {
      //cout << "Use template reco " << ierr << endl;
      
      LocalPoint template_lp( template_x_pos               , default_lp.y() , default_lp.z()  );
      LocalError template_le( template_x_err*template_x_err, default_le.xy(), default_le.yy() );
      

      final_lv = std::make_pair( template_lp, template_le ); 
    
    }
  else
    {
      //cout << "Use default reco " << ierr << endl;
      
      final_lv = std::make_pair( default_lp, default_le );
    }

  return final_lv; 


}
  


