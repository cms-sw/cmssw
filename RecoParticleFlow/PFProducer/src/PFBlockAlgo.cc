#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/Utils.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFResolutionMap.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include <stdexcept>
#include "TMath.h"

using namespace std;
using namespace reco;

//for debug only 
//#define PFLOW_DEBUG

const PFBlockAlgo::Mask PFBlockAlgo::dummyMask_;

PFBlockAlgo::PFBlockAlgo() : 
  blocks_( new reco::PFBlockCollection ),
  //   tracks_(tracks),
  //   clustersECAL_(clustersECAL),
  //   clustersHCAL_(clustersHCAL),
  resMapEtaECAL_(0),
  resMapPhiECAL_(0),
  resMapEtaHCAL_(0),
  resMapPhiHCAL_(0), 
  DPtovPtCut_(std::vector<double>(4,static_cast<double>(999.))),
  NHitCut_(std::vector<unsigned int>(4,static_cast<unsigned>(0))),
  chi2TrackECAL_(-1),
  chi2GSFECAL_(-1),
  chi2TrackHCAL_(-1), 
  chi2ECALHCAL_ (-1),
  chi2PSECAL_ (-1), 
  chi2PSTrack_ (-1), 
  chi2PSHV_ (-1), 
  resPSpitch_ (0),
  resPSlength_ (0),
  debug_(false) {}



void PFBlockAlgo::setParameters( const char* resMapEtaECAL,
				 const char* resMapPhiECAL,
				 const char* resMapEtaHCAL,
				 const char* resMapPhiHCAL, 
				 std::vector<double>& DPtovPtCut,
				 std::vector<unsigned int>& NHitCut,
				 double chi2TrackECAL,
				 double chi2GSFECAL,
				 double chi2TrackHCAL,
				 double chi2ECALHCAL,
				 double chi2PSECAL,
				 double chi2PSTrack,
				 double chi2PSHV,
				 bool   multiLink ) {
  
  try {
    resMapEtaECAL_ = new PFResolutionMap("resmapEtaECAL",resMapEtaECAL);
    resMapPhiECAL_ = new PFResolutionMap("resmapPhiECAL",resMapPhiECAL);
    resMapEtaHCAL_ = new PFResolutionMap("resmapEtaHCAL",resMapEtaHCAL);
    resMapPhiHCAL_ = new PFResolutionMap("resmapPhiHCAL",resMapPhiHCAL);
  }
  catch(std::exception& err ) {
    // cout<<err.what()<<endl;
    throw;
  }

  DPtovPtCut_    = DPtovPtCut;
  NHitCut_       = NHitCut;
  chi2TrackECAL_ = chi2TrackECAL;
  chi2GSFECAL_   = chi2GSFECAL;
  chi2TrackHCAL_ = chi2TrackHCAL; 
  chi2ECALHCAL_  = chi2ECALHCAL;
  chi2PSECAL_    = chi2PSECAL;
  chi2PSTrack_   = chi2PSTrack;
  chi2PSHV_      = chi2PSHV;
  double strip_pitch = 0.19;
  double strip_length = 6.1;
  resPSpitch_    = strip_pitch/sqrt(12.);
  resPSlength_   = strip_length/sqrt(12.);

  multipleLink_  = multiLink;
}

PFBlockAlgo::~PFBlockAlgo() {

#ifdef PFLOW_DEBUG
  if(debug_)
    cout<<"~PFBlockAlgo - number of remaining elements: "
	<<elements_.size()<<endl;
#endif
  
  if(resMapEtaECAL_) delete resMapEtaECAL_;
  
  if(resMapPhiECAL_) delete resMapPhiECAL_;
  
  if(resMapEtaHCAL_) delete resMapEtaHCAL_;

  if(resMapPhiHCAL_) delete resMapPhiHCAL_;

}

void 
PFBlockAlgo::findBlocks() {

  //  cout<<"findBlocks : "<<blocks_.get()<<endl;
  
  // the blocks have not been passed to the event, and need to be cleared
  if(blocks_.get() )blocks_->clear();
  else 
    blocks_.reset( new reco::PFBlockCollection );

  blocks_->reserve(elements_.size());
  for(IE ie = elements_.begin(); 
      ie != elements_.end();) {
    
#ifdef PFLOW_DEBUG
    if(debug_) {
      cout<<" PFBlockAlgo::findBlocks() ----------------------"<<endl;
      cout<<" element "<<**ie<<endl;
      cout<<" creating new block"<<endl;
    }
#endif
    
    blocks_->push_back( PFBlock() );
    
    vector< PFBlockLink > links;

    //    list< IE > used;
    ie = associate( elements_.end() , ie, links );

    // build remaining links in current block
    packLinks( blocks_->back(), links );
  }       
}




PFBlockAlgo::IE 
PFBlockAlgo::associate( IE last, 
			IE next, 
			vector<PFBlockLink>& links ) {

    
#ifdef PFLOW_DEBUG
  if(debug_ ) cout<<"PFBlockAlgo::associate start ----"<<endl;
#endif

  if( last!= elements_.end() ) {
    PFBlockLink::Type linktype = PFBlockLink::NONE;
    double chi2 = -1; 
    double dist = -1;
    PFBlock::LinkTest linktest = PFBlock::LINKTEST_CHI2;
    link( *last, *next, linktype, linktest, chi2, dist ); 

   
    if(chi2<-0.5) {
#ifdef PFLOW_DEBUG
      if(debug_ ) cout<<"link failed"<<endl;
#endif
      return ++next; // association failed
    }
    else {
      // add next element to the current pflowblock
      blocks_->back().addElement( *next );  

      // (*next)->setIndex( blocks_->back()->indexToLastElement() );
      
      // this is not necessary? 
      // next->setPFBlock(this);
      
      // create a link between next and last
      links.push_back( PFBlockLink(linktype, 
				   linktest,
				   chi2, 
				   dist,
				   (*last)->index(), 
				   (*next)->index() ) );
      // not necessary ?
      //       next->connect( links_.size()-1 );
      //       last->connect( links_.size()-1 );      
    }
  }
  else {
    // add next element to this eflowblock
#ifdef PFLOW_DEBUG
    if(debug_ ) cout<<"adding to block element "<<(**next)<<endl;
#endif
    blocks_->back().addElement( *next );
    // (*next)->setIndex( blocks_->back()->indexToLastElement() );   
    // next->setPFBlock(this);
  }

  // recursive call: associate next and other unused elements 
  
  //   IE afterNext = next;
  //   ++afterNext;
  //  cout<<"last "<<**last<<" next "<<**next<<endl;
  
  for(IE ie = elements_.begin(); 
      ie != elements_.end();) {
    
    if( ie == last || ie == next ) {
      ++ie;
      continue;
    } 
    
    // *ie already included to a block
    if( (*ie)->locked() ) {
#ifdef PFLOW_DEBUG
      if(debug_ ) cout<<"element "<<(**ie)<<"already used"<<endl;
#endif
      ++ie;
      continue;
    }    
    
    
#ifdef PFLOW_DEBUG
    if(debug_ ) cout<<"calling associate "<<(**next)<<" & "<<(**ie)<<endl;
#endif
    ie = associate(next, ie, links);
  }       

#ifdef PFLOW_DEBUG
  if(debug_ ) {
    cout<<"**** deleting element "<<endl;
    cout<<**next<<endl;
  }
#endif
  delete *next;

#ifdef PFLOW_DEBUG
  if(debug_ ) {
    cout<<"**** removing element "<<endl;
  }
#endif

  IE iteratorToNextFreeElement = elements_.erase( next );

#ifdef PFLOW_DEBUG
  if(debug_ ) cout<<"PFBlockAlgo::associate stop ----"<<endl;
#endif

  return iteratorToNextFreeElement;
}



void 
PFBlockAlgo::packLinks( reco::PFBlock& block, 
			const vector<PFBlockLink>& links ) const {
  
  
  const edm::OwnVector< reco::PFBlockElement >& els = block.elements();
  
  block.bookLinkData();

  //First Loop: update all link data
  for( unsigned i1=0; i1<els.size(); i1++ ) {
    for( unsigned i2=0; i2<els.size(); i2++ ) {
      
      // no reflexive link
      if( i1==i2 ) continue;
      
      double chi2 = -1;
      double dist = -1;
      
      bool linked = false;
      PFBlock::LinkTest linktest 
	= PFBlock::LINKTEST_CHI2; 

      // are these elements already linked ?
      // this can be optimized

      for( unsigned il=0; il<links.size(); il++ ) {
	if( (links[il].element1() == i1 && 
	     links[il].element2() == i2) || 
	    (links[il].element1() == i2 && 
	     links[il].element2() == i1) ) { // yes
	  
	  chi2 = links[il].chi2();
	  dist = links[il].dist();
	  linked = true;

	  //modif-beg	  
	  //retrieve type of test used to get chi2
	  linktest = links[il].test();
#ifdef PFLOW_DEBUG
	  if( debug_ )
	    cout << "Reading link vector: linktest used=" 
		 << linktest 
		 << " chi2= " << chi2 
		 << endl; 
#endif
	  //modif-end
	  
	  break;
	} 
      }
      
      if(!linked) {
	PFBlockLink::Type linktype = PFBlockLink::NONE;
	link( & els[i1], & els[i2], linktype, linktest, chi2, dist);
      }

      //loading link data according to link test used: CHI2, RECHIT 
      //block.setLink( i1, i2, chi2, block.linkData() );
#ifdef PFLOW_DEBUG
      if( debug_ )
	cout << "Setting link between elements " << i1 << " and " << i2
	     << " of chi2 =" << chi2 << " computed from link test "
	     << linktest << endl;
#endif
      block.setLink( i1, i2, chi2, dist, block.linkData(), linktest );
    }
  }

  //Second Loop: checking the link by rechit for HCAL 
  //A Hcal-Track link by rechit is preserved  
  //only of the cluster is linked to another track.

  // PJ Keep all link by rechit, even for one track !
  // PJ Indeed, the logic of the link by rechit also applies to the 
  // PJ case where there is only one tracks (and e.g., a neutral 
  // PJ hadron that biasses the cluster position
  /*
  if( multipleLink_ ) {
    for( unsigned i1=0; i1<els.size(); i1++ ) {
      for( unsigned i2=0; i2<els.size(); i2++ ) {
	
	// no reflexive link
	if( i1==i2 ) continue;
	
	//Only checking link by rechit 
	double chi2 = block.chi2( i1, i2, block.linkData(), 
				  PFBlock::LINKTEST_RECHIT );

	double dist = block.dist( i1, i2, block.linkData(), 
				  PFBlock::LINKTEST_RECHIT );

	
	//if( chi2 < chi2TrackHCAL_ || chi2<0 ) continue;
	//if not linked, continue 
	if( chi2<0 ) continue; 
	
 	bool keeplink = false;
#ifdef PFLOW_DEBUG
	if( debug_ )
	  cout << "This is a link by rechit concerning elements: " 
	       << i1  << " and " << i2 << endl;
#endif
	
	unsigned int idCluster = i1;
	unsigned int idTrack   = i2;
	PFBlockElement::Type type2 = els[i2].type();
	if( type2 == PFBlockElement::HCAL ){
	  idCluster = i2; 
	  idTrack = i1;
	}

	//protection: only considering possible HCAL-TRACK
	//link by rechit in what follows
	if( els[idCluster].type() != PFBlockElement::HCAL )
	  continue;
	if( els[idTrack].type()   != PFBlockElement::TRACK )
	  continue;
 
#ifdef PFLOW_DEBUG
	if( debug_ ){
	  cout << "Hcal Cluster is element " << idCluster << endl;
	  cout << "Track is element "        << idTrack << endl;
	  cout << "Checking if cluster "     << idCluster 
	       << " is linked to another track" << endl;
	}
#endif
	
	for( unsigned k1=0; k1<els.size(); k1++ ) {
	  for( unsigned k2=0; k2<els.size(); k2++ ) {
	    
	    // no reflexive link
	    if( k1==k2 ) continue;
	    
	    if( ( k1 != idTrack && 
		  k2 == idCluster ) || 
		( k1 == idCluster && 
		  k2 != idTrack ) ) { // yes
	      
	      //retrieving chi2 values for each possible link tests
	      double chi2loc_chi2   
		= block.chi2( i1, i2, block.linkData(), 
			      PFBlock::LINKTEST_CHI2 ); 
	      double chi2loc_rechit 
		= block.chi2( i1, i2, block.linkData(), 
			      PFBlock::LINKTEST_RECHIT ); 
	      
	      PFBlockElement::Type type1loc = els[k1].type();
	      PFBlockElement::Type type2loc = els[k2].type();

	      //  	      cout << "  elements: " << k1 << " " << k2  << endl;
	      //  	      cout << "  types: "    << type1loc  << " " << type2loc << endl;
	      //  	      cout << "  chi2 from chi2 test="    << chi2loc_chi2    << endl;
	      //  	      cout << "  chi2 from rechit test= " << chi2loc_rechit  << endl;
	      
	      if( type1loc == 1 || type2loc == 1 )
		//if( chi2loc > 0 ){
		//if either track is linked to that cluster by rechit of chi2
		if( chi2loc_chi2 > 0 || chi2loc_rechit > 0 ){
#ifdef PFLOW_DEBUG
		  if( debug_ )
		    cout << "This cluster is linked to another tracks:"
			 << "keep this link" << endl; 
#endif
		  keeplink = true;
		  break;
		}
	    }//finding other track
	    
	  }//loop ele1
	}//loop ele2
	
	if(!keeplink) 
	  { 
#ifdef PFLOW_DEBUG
	    if( debug_ ) 
	      cout << "This cluster is not linked to any other tracks:" 
		   << "link by rechit must be removed" << endl;
#endif
	    //block.setLink( i1, i2, -1, -1, block.linkData() );
	    block.setLink( i1, i2, -1, -1, block.linkData(),
			   PFBlock::LINKTEST_RECHIT );
	  }//destroy link by rechit
      }// loop ele2
    }//loop ele1
  }//mulipleLink
  */

  checkNuclearLinks( block );
}



void 
PFBlockAlgo::buildGraph() {
  // loop on all blocks and create a big graph
}



void 
PFBlockAlgo::link( const reco::PFBlockElement* el1, 
		   const reco::PFBlockElement* el2, 
		   PFBlockLink::Type& linktype, 
		   reco::PFBlock::LinkTest& linktest,
		   double& chi2, double& dist) const {
  


  chi2=-1;
  dist=-1.;
  std::pair<double,double> lnk(chi2,dist);
  linktest = PFBlock::LINKTEST_CHI2; //chi2 by default 

  PFBlockElement::Type type1 = el1->type();
  PFBlockElement::Type type2 = el2->type();

  if( type1==type2 ) {
    // cannot link 2 elements of the same type. 
    // except if the elements are 2 tracks
    if( type1!=PFBlockElement::TRACK ) return;
    // cannot link two primary tracks  (except if they come from a V0)
    else if ( 
	     ((!el1->isSecondary()) && (!el2->isSecondary())) && 
	     ((!el1->trackType(reco::PFBlockElement::T_FROM_V0)) || 
	      (!el2->trackType(reco::PFBlockElement::T_FROM_V0)))
	     ) return;
  }

  linktype = static_cast<PFBlockLink::Type>
    ((1<< (type1-1) ) | (1<< (type2-1) ));

  if(debug_ ) std::cout << " PFBlockAlgo links type1 " << type1 << " type2 " << type2 << std::endl;

  PFBlockElement::Type lowType = type1;
  PFBlockElement::Type highType = type2;
  const PFBlockElement* lowEl = el1;
  const PFBlockElement* highEl = el2;
  
  if(type1>type2) {
    lowType = type2;
    highType = type1;
    lowEl = el2;
    highEl = el1;
  }
  
  switch(linktype) {
  case PFBlockLink::TRACKandPS1:
  case PFBlockLink::TRACKandPS2:
    {
      //       cout<<"TRACKandPS"<<endl;
      PFRecTrackRef trackref = lowEl->trackRefPF();
      PFClusterRef  clusterref = highEl->clusterRef();
      assert( !trackref.isNull() );
      assert( !clusterref.isNull() );
      lnk = testTrackAndPS( *trackref, *clusterref );
      chi2 = lnk.first;
      dist = lnk.second;
      break;
    }

  case PFBlockLink::TRACKandECAL:
    {
      if(debug_ ) cout<<"TRACKandECAL"<<endl;
      PFRecTrackRef trackref = lowEl->trackRefPF();

      if(debug_ ) std::cout << " Track pt " << trackref->trackRef()->pt() << std::endl;

      PFClusterRef  clusterref = highEl->clusterRef();
      assert( !trackref.isNull() );
      assert( !clusterref.isNull() );
      lnk = testTrackAndECAL( *trackref, *clusterref );
      chi2 = lnk.first;
      dist = lnk.second;
      if(debug_ )  std::cout << " chi2 from testTrackAndECAL " << chi2 << std::endl;
      //Link by rechit for ECAL

      /*
      lnk = testTrackAndClusterByRecHit( *trackref, *clusterref );

      if ( chi2 < 100. && chi2 > 0. && lnk.first < 0. ) {
	std::cout << "Warning : ECAL link by chi2 = " << chi2 
		  << ", but no link by RecHit ! " << std::endl;
      }

      chi2 = -1.;
      */


      if( ( chi2 > chi2TrackECAL_ || chi2 < 0 )
	  && multipleLink_ ){	
	//If Chi2 failed checking if Track can be linked by rechit
	//to a ECAL cluster. Definition:
	// A cluster can be linked to a track by rechit if the 
	// extrapolated position of the track to the ECALShowerMax 
	// falls within the boundaries of any cell that belongs 
	// to this cluster.
	if(debug_ ) std::cout << " try  testTrackAndClusterByRecHit " << std::endl;
	lnk = testTrackAndClusterByRecHit( *trackref, *clusterref );
	chi2 = lnk.first;
	dist = lnk.second;
	if(debug_ ) std::cout << " chi2 testTrackAndClusterByRecHit " << chi2 << std::endl;
	linktest = PFBlock::LINKTEST_RECHIT;
      }//link by rechit  

      if ( chi2>0) {
	if(debug_ ) std::cout << " Here a link has been established between a track an Ecal with chi2  " << chi2 <<  std::endl;
      } else {
	if(debug_ ) std::cout << " No link found " << std::endl;
      }



      break;
    }
  case PFBlockLink::TRACKandHCAL:
    {
      //       cout<<"TRACKandHCAL"<<endl;
      PFRecTrackRef trackref = lowEl->trackRefPF();
      PFClusterRef  clusterref = highEl->clusterRef();
      assert( !trackref.isNull() );
      assert( !clusterref.isNull() );
      lnk = testTrackAndHCAL( *trackref, *clusterref );
      chi2 = lnk.first;
      dist = lnk.second;
  
      /*
      lnk = testTrackAndClusterByRecHit( *trackref, *clusterref );

      if ( chi2 < 100. && chi2 > 0. && lnk.first < 0. ) {
	std::cout << "Warning : HCAL link by chi2 = " << chi2 
		  << ", but no link by RecHit ! " << std::endl;
      }

      chi2 = -1.;
      */

      if( ( chi2 > chi2TrackHCAL_ || chi2 < 0 )
	  && multipleLink_ ){	
	//If Chi2 failed checking if Track can be linked by rechit
	//to a HCAL cluster. Definition:
	// A cluster can be linked to a track by rechit if the 
	// extrapolated position of the track to the HCAL entrance 
	// falls within the boundaries of any cell that belongs 
	// to this cluster.
	
	lnk = testTrackAndClusterByRecHit( *trackref, *clusterref );
	chi2 = lnk.first;
	dist = lnk.second;
	linktest = PFBlock::LINKTEST_RECHIT;
      }//link by rechit  
      
      break;
    }
  case PFBlockLink::ECALandHCAL:
    {
      //       cout<<"ECALandHCAL"<<endl;
      PFClusterRef  ecalref = lowEl->clusterRef();
      PFClusterRef  hcalref = highEl->clusterRef();
      assert( !ecalref.isNull() );
      assert( !hcalref.isNull() );
      lnk = testECALAndHCAL( *ecalref, *hcalref );
      chi2 = lnk.first;
      dist = lnk.second;
      break;
    }
  case PFBlockLink::PS1andECAL:
  case PFBlockLink::PS2andECAL:
    {
      //       cout<<"PSandECAL"<<endl;
      PFClusterRef  psref = lowEl->clusterRef();
      PFClusterRef  ecalref = highEl->clusterRef();
      assert( !psref.isNull() );
      assert( !ecalref.isNull() );
      lnk = testPSAndECAL( *psref, *ecalref );
      chi2 = lnk.first;
      dist = lnk.second;

      if ( chi2 > chi2PSECAL_ || chi2 < 0 ) {
	//If Chi2 failed, check link by rechit in this case too. 
	lnk = testECALAndPSByRecHit( *ecalref, *psref );
	dist = lnk.second;
	if ( dist > 0. ) chi2 = dist*dist*1E6;
	else chi2 = -1.;
	linktest = PFBlock::LINKTEST_RECHIT;
      }

      break;
    }
  case PFBlockLink::PS1andPS2:
    {
      PFClusterRef  ps1ref = lowEl->clusterRef();
      PFClusterRef  ps2ref = highEl->clusterRef();
      assert( !ps1ref.isNull() );
      assert( !ps2ref.isNull() );
      lnk = testPS1AndPS2( *ps1ref, *ps2ref );
      chi2 = lnk.first;
      dist = lnk.second;
      break;
    }
  case PFBlockLink::TRACKandTRACK:
    {
      if(debug_ ) cout<<"TRACKandTRACK"<<endl;
      lnk = testLinkByVertex(lowEl, highEl);
      chi2 = lnk.first;
      dist = lnk.second;
      if(debug_ ) std::cout << " PFBlockLink::TRACKandTRACK chi2 " << chi2 << std::endl;
      break;
    }
  case PFBlockLink::ECALandGSF:
    {
      PFClusterRef  clusterref = lowEl->clusterRef();
      assert( !clusterref.isNull() );
      const reco::PFBlockElementGsfTrack *  GsfEl =  dynamic_cast<const reco::PFBlockElementGsfTrack*>(highEl);
      const PFRecTrack * myTrack =  &(GsfEl->GsftrackPF());
      lnk = testTrackAndECAL( *myTrack, *clusterref);
      chi2 = lnk.first;
      dist = lnk.second;
      if( ( chi2 >  chi2GSFECAL_  || chi2 < 0 )
	  && multipleLink_ ){	
	//If Chi2 failed checking if Track can be linked by rechit
	//to a ECAL cluster. Definition:
	// A cluster can be linked to a track by rechit if the 
	// extrapolated position of the track to the ECALShowerMax 
	// falls within the boundaries of any cell that belongs 
	// to this cluster.
	//	std::cout << " try GSF testTrackAndClusterByRecHit " << std::endl;
	lnk = testTrackAndClusterByRecHit( *myTrack, *clusterref );
	chi2 = lnk.first;
	dist = lnk.second;
	if(debug_ ) std::cout << " chi2 testTrackAndClusterByRecHit " << chi2 << std::endl;
	linktest = PFBlock::LINKTEST_RECHIT;
      }//link by rechit  
      
      if ( chi2>0) {
	if(debug_ ) 
	  std::cout << " Here a link has been established between a track an Ecal with chi2  " 
		    << chi2 <<  std::endl;

      } else {
	if(debug_ ) std::cout << " No link found " << std::endl;
      }
      break;
    }
  case PFBlockLink::TRACKandGSF:
    {
      PFRecTrackRef trackref = lowEl->trackRefPF();
      assert( !trackref.isNull() );
      const reco::PFBlockElementGsfTrack *  GsfEl =  dynamic_cast<const reco::PFBlockElementGsfTrack*>(highEl);
      GsfPFRecTrackRef gsfref = GsfEl->GsftrackRefPF();
      reco::TrackRef kftrackref= (*trackref).trackRef();
      assert( !gsfref.isNull() );
      PFRecTrackRef refkf = (*gsfref).kfPFRecTrackRef();
      if(refkf.isNonnull())
	{
	  reco::TrackRef gsftrackref = (*refkf).trackRef();
	  if (gsftrackref.isNonnull()&&kftrackref.isNonnull()) {
	    if (kftrackref == gsftrackref) { 
	      chi2 = 1;
	      dist = 0.001;
	      //	      std::cout <<  " Linked " << std::endl;
	    } else { 
	      chi2 = -1;
	      dist = -1.;
	      //	      std::cout <<  " Not Linked " << std::endl;
	    }
	  }
	  else { 
	    chi2 = -1;
	    dist = -1.;
	    //	    std::cout <<  " Not Linked " << std::endl;
	  }
	}
      else
	{
	  chi2 = -1;
	  dist = -1.;
	  //	  std::cout <<  " Not Linked " << std::endl;
	}
      break;      
    }
	 
  case PFBlockLink::GSFandBREM:
    {
      const reco::PFBlockElementGsfTrack * GsfEl  =  dynamic_cast<const reco::PFBlockElementGsfTrack*>(lowEl);
      const reco::PFBlockElementBrem * BremEl =  dynamic_cast<const reco::PFBlockElementBrem*>(highEl);
      GsfPFRecTrackRef gsfref = GsfEl->GsftrackRefPF();
      GsfPFRecTrackRef bremref = BremEl->GsftrackRefPF();
      assert( !gsfref.isNull() );
      assert( !bremref.isNull() );
      if (gsfref == bremref)  { 
	chi2 = 1;
	dist = 0.001;
      } else { 
	chi2 = -1;
	dist = -1.;
      }
      break;
    }
  case PFBlockLink::ECALandBREM:
    {
      PFClusterRef  clusterref = lowEl->clusterRef();
      assert( !clusterref.isNull() );
      const reco::PFBlockElementBrem * BremEl =  dynamic_cast<const reco::PFBlockElementBrem*>(highEl);
      const PFRecTrack * myTrack = &(BremEl->trackPF());
      double DP = (BremEl->DeltaP())*(-1.);
      double SigmaDP = BremEl->SigmaDeltaP();
      double SignBremDp = DP/SigmaDP;
      lnk = testTrackAndECAL( *myTrack, *clusterref, SignBremDp);
      chi2 = lnk.first;
      dist = lnk.second;

      if( ( chi2 > chi2TrackECAL_ || chi2 < 0 )
	  && multipleLink_ ){	
	//If Chi2 failed checking if Track can be linked by rechit
	//to a ECAL cluster. Definition:
	// A cluster can be linked to a track by rechit if the 
	// extrapolated position of the track to the ECALShowerMax 
	// falls within the boundaries of any cell that belongs 
	// to this cluster.
	if(debug_ ) std::cout << "ECALandBREM: try  testTrackAndClusterByRecHit " << std::endl;
	bool isBrem = true;
	lnk = testTrackAndClusterByRecHit( *myTrack, *clusterref, isBrem);
	chi2 = lnk.first;
	dist = lnk.second;
	if(debug_ ) std::cout << "ECALandBREM: chi2 testTrackAndClusterByRecHit " << chi2 << std::endl;
	linktest = PFBlock::LINKTEST_RECHIT;
      }//link by rechit  

      break;
    }
  case PFBlockLink::PS1andGSF:
  case PFBlockLink::PS2andGSF:
    {
      PFClusterRef  psref = lowEl->clusterRef();
      assert( !psref.isNull() );
      const reco::PFBlockElementGsfTrack *  GsfEl =  dynamic_cast<const reco::PFBlockElementGsfTrack*>(highEl);
      const PFRecTrack * myTrack =  &(GsfEl->GsftrackPF());
      lnk = testTrackAndPS( *myTrack, *psref );
      chi2 = lnk.first;
      dist = lnk.second;
      break;
    }
  case PFBlockLink::PS1andBREM:
  case PFBlockLink::PS2andBREM:
    {
      PFClusterRef  psref = lowEl->clusterRef();
      assert( !psref.isNull() );
      const reco::PFBlockElementBrem * BremEl =  dynamic_cast<const reco::PFBlockElementBrem*>(highEl);
      const PFRecTrack * myTrack = &(BremEl->trackPF());
      lnk = testTrackAndPS( *myTrack, *psref );
      chi2 = lnk.first;
      dist = lnk.second;
      break;
    }
  case PFBlockLink::HCALandGSF:
    {
      PFClusterRef  clusterref = lowEl->clusterRef();
      assert( !clusterref.isNull() );
      const reco::PFBlockElementGsfTrack *  GsfEl =  dynamic_cast<const reco::PFBlockElementGsfTrack*>(highEl);
      const PFRecTrack * myTrack =  &(GsfEl->GsftrackPF());
      lnk = testTrackAndHCAL( *myTrack, *clusterref);
      chi2 = lnk.first;
      dist = lnk.second;
      if( ( chi2 > chi2TrackHCAL_ || chi2 < 0 )
	  && multipleLink_ ){	
	//If Chi2 failed checking if Track can be linked by rechit
	//to a HCAL cluster. Definition:
	// A cluster can be linked to a track by rechit if the 
	// extrapolated position of the track to the HCAL entrance 
	// falls within the boundaries of any cell that belongs 
	// to this cluster.
	
	lnk = testTrackAndClusterByRecHit( *myTrack, *clusterref );
	chi2 = lnk.first;
	dist = lnk.second;
	linktest = PFBlock::LINKTEST_RECHIT;
      }//link by rechit 
      break;
    }
  case PFBlockLink::HCALandBREM:
    {
      PFClusterRef  clusterref = lowEl->clusterRef();
      assert( !clusterref.isNull() );
      const reco::PFBlockElementBrem * BremEl =  dynamic_cast<const reco::PFBlockElementBrem*>(highEl);
      const PFRecTrack * myTrack = &(BremEl->trackPF());
      lnk = testTrackAndHCAL( *myTrack, *clusterref);
      chi2 = lnk.first;
      dist = lnk.second;
      break;
    }
  case PFBlockLink::HFEMandHFHAD:
    {
      // cout<<"HFEMandHFHAD"<<endl;
      PFClusterRef  eref = lowEl->clusterRef();
      PFClusterRef  href = highEl->clusterRef();
      assert( !eref.isNull() );
      assert( !href.isNull() );
      lnk = testHFEMAndHFHADByRecHit( *eref, *href );
      chi2 = lnk.first;
      dist = lnk.second;
      break;
      
      break;
    }
  default:
    chi2 = -1.;
    dist = -1.;
    // cerr<<"link type not implemented yet: 0x"<<hex<<linktype<<dec<<endl;
    // assert(0);
    return;
  }
}

std::pair<double,double>
PFBlockAlgo::testTrackAndPS(const PFRecTrack& track, 
			    const PFCluster& ps)  const {

  //   cout<<"entering testTrackAndPS"<<endl;
  // resolution of PS cluster dxdx and dydy from strip pitch and length
  double dx=0.;
  double dy=0.;
  
  unsigned layerid =0;
  // PS1: vertical strips  PS2: horizontal strips
  switch (ps.layer()) {
  case PFLayer::PS1:
    layerid = reco::PFTrajectoryPoint::PS1;
    
    // vertical strips in PS1, measure x with pitch precision
    dx = resPSpitch_;
    dy = resPSlength_; 
    break;
  case PFLayer::PS2:
    layerid = reco::PFTrajectoryPoint::PS2;
    // horizontal strips in PS2, measure y with pitch precision
    dy = resPSpitch_;
    dx = resPSlength_;
    break;
  default:
    break;
  }
  const reco::PFTrajectoryPoint& atPS
    = track.extrapolatedPoint( layerid );  
  // did not reach PS, cannot be associated with a cluster.
  if( ! atPS.isValid() ) return std::pair<double,double>(-1,-1);   
  
  double trackx = atPS.position().X();
  double tracky = atPS.position().Y();
	double trackz = atPS.position().Z(); // MDN jan 09
  
  // ps position  x, y
  double psx = ps.position().X();
  double psy = ps.position().Y();
	// MDN Jan 09: check that trackz and psz have the same sign
	double psz = ps.position().Z();
	if( trackz*psz < 0.) return std::pair<double,double>(-1,-1); 
  
  // rec track resolution negligible compared to ps resolution?
  // compute chi2 PS_TRACK in x, y
  double trackresolx = 0.;
  double trackresoly = 0.;
  
  double chi2 = (psx-trackx)*(psx-trackx)/(dx*dx + trackresolx*trackresolx)
    + (psy-tracky)*(psy-tracky)/(dy*dy + trackresoly*trackresoly);

  double dist = std::sqrt( (psx-trackx)*(psx-trackx)
			 + (psy-tracky)*(psy-tracky));

  
#ifdef PFLOW_DEBUG
  if(debug_) cout<<"testTrackAndPS "<<chi2<<" "<<endl;
  if(debug_){
    cout<<" trackx " << trackx << " trackresolx " << trackresolx
	<<" tracky " << tracky << " trackresoly " << trackresoly << endl
	<<" psx "    <<  psx   << "  dx "         << dx
	<<" psy "    << psy    << "  dy "         << dy << endl;
  }
#endif
  
  
  if(chi2<chi2PSTrack_ || chi2PSTrack_<0 )
    return std::pair<double,double>(chi2,dist/1000.);
  else 
    return std::pair<double,double>(-1,-1);
}



std::pair<double,double> 
PFBlockAlgo::testTrackAndECAL(const PFRecTrack& track, 
			      const PFCluster& ecal, 
			      double SignBremDp)  const {
  
  //   cout<<"entering testTrackAndECAL"<<endl;
  
  
  double tracketa;
  double trackphi;

  // special chi2 for GSF-ECAL matching
  double chi2cut = (track.algoType()!=PFRecTrack::GSF) ? chi2TrackECAL_ : chi2GSFECAL_;

  //  cout << " SignBremDp " << SignBremDp << endl;
  // The SignBremDp cut has to be optimized 
  if (SignBremDp > 3) {
    const reco::PFTrajectoryPoint& atECAL 
      = track.extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax );
    if( ! atECAL.isValid() ) return std::pair<double,double>(-1,-1);   
    tracketa = atECAL.positionREP().Eta();
    trackphi = atECAL.positionREP().Phi();
    //   cout<<"atECAL "<<atECAL.layer()<<" "
    //       <<atECAL.position().Eta()<<" "
    //       <<atECAL.position().Phi()<<endl;
  }
  else {
    // needed only for the brem when the momentum is bad estimated. 
    // The ECAL cluster energy is taken in these cases

    const reco::PFTrajectoryPoint& atECAL 
      = track.extrapolatedPoint( reco::PFTrajectoryPoint::ECALEntrance );
    if( ! atECAL.isValid() ) return std::pair<double,double>(-1,-1);   
    math::XYZVector posatecal( atECAL.position().x(),
			       atECAL.position().y(),
			       atECAL.position().z());
    
    bool isBelowPS=(fabs(ecal.positionREP().Eta())>1.65) ? true :false;
    double clusenergy = ecal.energy();
    double ecalShowerDepth 
      = reco::PFCluster::getDepthCorrection(clusenergy, isBelowPS,false);
    
    math::XYZVector direction(atECAL.momentum().x(),
			      atECAL.momentum().y(),
			      atECAL.momentum().z() );

    direction=direction.unit();
    posatecal += ecalShowerDepth*direction;
    tracketa = posatecal.eta();
    trackphi = posatecal.phi();
  }


  double ecaleta  = ecal.positionREP().Eta();
  double ecalphi  = ecal.positionREP().Phi();
  

  PFResolutionMap* mapeta = const_cast<PFResolutionMap*>(resMapEtaECAL_);
  PFResolutionMap* mapphi = const_cast<PFResolutionMap*>(resMapPhiECAL_);

  
  double ecaletares 
    = mapeta->GetBinContent(mapeta->FindBin(ecaleta, 
					    ecal.energy() ) );
  double ecalphires 
    = mapphi->GetBinContent(mapphi->FindBin(ecaleta, 
					    ecal.energy() ) );
  
  
  // rec track resolution should be negligible compared to ecal resolution
  double trackres = 0;
  
  std::pair<double,double> lnk = computeChi2( ecaleta, ecaletares, 
					      ecalphi, ecalphires, 
					      tracketa, trackres, 
					      trackphi, trackres);
  double chi2 = lnk.first;

#ifdef PFLOW_DEBUG
  if(debug_) cout<<"testTrackAndECAL "<<chi2<<" "<<endl;
  if(debug_){
    cout<<" ecaleta "  << ecaleta  << "  ecaletares " <<ecaletares
	<<" ecalphi "  << ecalphi  << "  ecalphires " <<ecalphires
	<<" tracketa " << tracketa << "  trackres "   <<trackres
	<<" trackphi " << trackphi << "  trackres "   <<trackres << endl;
  }
#endif
  

  if(chi2<chi2cut || chi2TrackECAL_<0 )
    return lnk;
  else 
    return std::pair<double,double>(-1,-1);
}



std::pair<double,double> 
PFBlockAlgo::testTrackAndHCAL(const PFRecTrack& track, 
			      const PFCluster& hcal)  const {
  
  
  //   cout<<"entering testTrackAndHCAL"<<endl;

  // this is the fake cluster for ps cells
  //   if( ! hcal.type() ) return -1;
  
  
  const reco::PFTrajectoryPoint& atHCAL 
    = track.extrapolatedPoint( reco::PFTrajectoryPoint::HCALEntrance );
  
  //   cout<<"atHCAL "<<atHCAL.layer()<<" "
  //       <<atHCAL.position().Eta()<<" "
  //       <<atHCAL.position().Phi()<<endl;
  
  // did not reach hcal, cannot be associated with a cluster.
  if( ! atHCAL.isValid() ) return std::pair<double,double>(-1,-1);   
  
  double tracketa = atHCAL.positionREP().Eta();
  double trackphi = atHCAL.positionREP().Phi();
  double hcaleta  = hcal.positionREP().Eta();
  double hcalphi  = hcal.positionREP().Phi();
  
  
  PFResolutionMap* mapeta = const_cast<PFResolutionMap*>(resMapEtaHCAL_);
  PFResolutionMap* mapphi = const_cast<PFResolutionMap*>(resMapPhiHCAL_);
  
  double hcaletares 
    = mapeta->GetBinContent(mapeta->FindBin(hcaleta, 
					    hcal.energy() ) );
  double hcalphires 
    = mapphi->GetBinContent(mapphi->FindBin(hcaleta, 
					    hcal.energy() ) );
  
  
  // rec track resolution should be negligible compared to hcal resolution
  double trackres = 0;
  
  std::pair<double,double> lnk = computeChi2( hcaleta, hcaletares, 
					      hcalphi, hcalphires, 
					      tracketa, trackres, 
					      trackphi, trackres);
  double chi2 = lnk.first;
  
#ifdef PFLOW_DEBUG
  if(debug_) cout<<"testTrackAndHCAL "<<chi2<<" "<<endl;
  if(debug_){
    cout<<" hcaleta "  << hcaleta << "  hcaletares "<<hcaletares
	<<" hcalphi "  << hcalphi << "  hcalphires "<<hcalphires
	<<" tracketa " << tracketa<< "  trackres "  <<trackres
	<<" trackphi " << trackphi<< "  trackres "  <<trackres << endl;
  }
#endif
  
  if(chi2<chi2TrackHCAL_ || chi2TrackHCAL_<0 )
    return lnk;
  else 
    return std::pair<double,double>(-1,-1);
}


std::pair<double,double> 
PFBlockAlgo::testECALAndHCAL(const PFCluster& ecal, 
			     const PFCluster& hcal)  const {
  
  //   cout<<"entering testECALAndHCAL"<<endl;
  
  
  PFResolutionMap* mapetaECAL = const_cast<PFResolutionMap*>(resMapEtaECAL_);
  PFResolutionMap* mapphiECAL = const_cast<PFResolutionMap*>(resMapPhiECAL_);
  
  PFResolutionMap* mapetaHCAL = const_cast<PFResolutionMap*>(resMapEtaHCAL_);
  PFResolutionMap* mapphiHCAL = const_cast<PFResolutionMap*>(resMapPhiHCAL_);
  
  // retrieve resolutions from resolution maps
  double ecaletares 
    = mapetaECAL->GetBinContent(mapetaECAL->FindBin(ecal.positionREP().Eta(), 
						    ecal.energy() ) );
  double ecalphires 
    = mapphiECAL->GetBinContent(mapphiECAL->FindBin(ecal.positionREP().Eta(), 
						    ecal.energy() ) );
		      
  double hcaletares 
    = mapetaHCAL->GetBinContent(mapetaHCAL->FindBin(hcal.positionREP().Eta(), 
						    hcal.energy() ) );
  double hcalphires 
    = mapphiHCAL->GetBinContent(mapphiHCAL->FindBin(hcal.positionREP().Eta(), 
						    hcal.energy() ) );
		      
  // compute chi2
  std::pair<double,double> lnk = 
    computeChi2( ecal.positionREP().Eta(), ecaletares, 
		 ecal.positionREP().Phi(), ecalphires, 
		 hcal.positionREP().Eta(), hcaletares, 
		 hcal.positionREP().Phi(), hcalphires );
  double chi2 = lnk.first;
  
#ifdef PFLOW_DEBUG
  if(debug_) cout<<"testECALAndHCAL "<<chi2<<" "<<endl;
  if(debug_){
    cout<<" ecaleta " << ecal.positionREP().Eta()<< "  ecaletares "<<ecaletares
	<<" ecalphi " << ecal.positionREP().Phi()<< "  ecalphires "<<ecalphires
	<<" hcaleta " << hcal.positionREP().Eta()<< "  hcaletares "<<hcaletares
	<<" hcalphi " << hcal.positionREP().Phi()<< "  hcalphires "<<hcalphires<< endl;
  }
#endif


  if(chi2<chi2ECALHCAL_ || chi2ECALHCAL_<0 )
    return lnk;
  else 
    return std::pair<double,double>(-1,-1);
}



std::pair<double,double> 
PFBlockAlgo::testPSAndECAL(const PFCluster& ps, 
			   const PFCluster& ecal)  const {
  
  //   cout<<"entering testPSAndECAL"<<endl;
  
  PFResolutionMap* mapetaECAL = const_cast<PFResolutionMap*>(resMapEtaECAL_);
  PFResolutionMap* mapphiECAL = const_cast<PFResolutionMap*>(resMapPhiECAL_);
  
  // retrieve resolutions from resolution maps
  double ecaletares 
    = mapetaECAL->GetBinContent(mapetaECAL->FindBin(ecal.positionREP().Eta(), 
						    ecal.energy() ) );
  double ecalphires 
    = mapphiECAL->GetBinContent(mapphiECAL->FindBin(ecal.positionREP().Eta(), 
						    ecal.energy() ) );
  // ecal position in eta and phi
  double ecaleta = ecal.positionREP().Eta();
  double ecalphi = ecal.positionREP().Phi();
  
  
  // ps position x, y, z, R and  rho, eta, phi
  double pseta = ps.positionREP().Eta();
  double psphi = ps.positionREP().Phi();
  double psrho = ps.positionREP().Rho();
  double psrho2 = psrho*psrho;
  double psx = ps.position().X();
  double psy = ps.position().Y();
  double psz = ps.position().Z();
  double psR = ps.position().R();
  // resolution of PS cluster dxdx and dydy from strip pitch and length
  double dxdx =0.;
  double dydy =0.;
  switch (ps.layer()) {
  case PFLayer::PS1:
    // vertical strips, measure x with pitch precision
    dxdx = resPSpitch_*resPSpitch_;
    dydy = resPSlength_*resPSlength_;
    break;
  case PFLayer::PS2:
    // horizontal strips, measure y with pitch precision
    dydy = resPSpitch_*resPSpitch_;
    dxdx = resPSlength_*resPSlength_;
    break;
  default:
    break;
  }
  // derivatives deta/dx, deta/dy, dphi/dx, dphi/deta
  double detadx = psx*psz/(psrho2*psR);
  double detady = psy*psz/(psrho2*psR);
  double dphidx = -psy/psrho2;
  double dphidy = psx/psrho2;
  // propagate error matrix  x. y (diagonal) to eta, phi (non diagonal)
  double detadeta = detadx*detadx*dxdx + detady*detady*dydy;
  double dphidphi = dphidx*dphidx*dxdx + dphidy*dphidy*dydy;
  double detadphi = detadx*dphidx*dxdx + detady*dphidy*dydy;
  // add ecal resol in quadrature
  double detadetas = detadeta + ecaletares*ecaletares;
  double dphidphis = dphidphi + ecalphires*ecalphires;
  // compute chi2 in eta, phi with non diagonal error matrix (detadphi non zero)
  double deta = pseta - ecaleta;
  double dphi = Utils::mpi_pi(psphi - ecalphi);
  double det  = detadetas*dphidphis - detadphi*detadphi;
  double chi2 
    = (dphidphis*deta*deta + detadetas*dphi*dphi - 2.*detadphi*deta*dphi)/det;
  double dist = std::sqrt(deta*deta+dphi*dphi);
  
  
  //#ifdef PFLOW_DEBUG
  if(debug_) cout<<"testPSAndECAL "<<chi2<<" "<<endl;
  if(debug_){
    double psetares = sqrt(detadeta);
    double psphires = sqrt (dphidphi);
    cout<< " pseta "  <<pseta   << " psetares "   << psetares
	<< " psphi "  <<psphi   << " psphires "   << psphires << endl
	<< " ecaleta "<<ecaleta << " ecaletares " << ecaletares
	<< " ecalphi "<<ecalphi << " ecalphires " << ecalphires<< endl;
    cout << "deta/dphi/dist = " << deta << " " << dphi << " " << dist << " " << std::endl;
  }
  //#endif
  

  if(chi2<chi2PSECAL_ || chi2PSECAL_<0 )
    return std::pair<double,double>(chi2,dist);
  else 
    return std::pair<double,double>(-1,-1);
}


std::pair<double,double> 
PFBlockAlgo::testPS1AndPS2(const PFCluster& ps1, 
			   const PFCluster& ps2)  const {
  
  //   cout<<"entering testPS1AndPS2"<<endl;
  
  // compute chi2 in y, z using swimming formulae
  // y2 = y1 * z2/z1   and x2 = x1 *z2/z1
  
  // ps position1  x, y, z
  double x1 = ps1.position().X();
  double y1 = ps1.position().Y();
  double z1 = ps1.position().Z();
  double x2 = ps2.position().X();
  double y2 = ps2.position().Y();
  double z2 = ps2.position().Z();
	// MDN Bug correction Jan 09: check that z1 and z2 have the same sign!
	if (z1*z2<0.) return std::pair <double, double> (-1, -1);
  // swim to PS2
  double scale = z2/z1;
  double x1atPS2 = x1*scale;
  double y1atPS2 = y1*scale;
  // resolution of PS cluster dxdx and dydy from strip pitch and length
  // vertical strips in PS1, measure x with pitch precision
  double dx1dx1 = resPSpitch_*resPSpitch_*scale*scale;
  double dy1dy1 = resPSlength_*resPSlength_*scale*scale;
  // horizontal strips in PS2 , measure y with pitch precision
  double dy2dy2 = resPSpitch_*resPSpitch_;
  double dx2dx2 = resPSlength_*resPSlength_;
  
  double chi2 = (x2-x1atPS2)*(x2-x1atPS2)/(dx1dx1 + dx2dx2) 
    + (y2-y1atPS2)*(y2-y1atPS2)/(dy1dy1 + dy2dy2);
  
  double dist = std::sqrt( (x2-x1atPS2)*(x2-x1atPS2)
			 + (y2-y1atPS2)*(y2-y1atPS2));
    
#ifdef PFLOW_DEBUG
  if(debug_) cout<<"testPS1AndPS2 "<<chi2<<" "<<endl;
  if(debug_){
    cout<<" x1atPS2 "<< x1atPS2 << " dx1 "<<resPSpitch_*scale
	<<" y1atPS2 "<< y1atPS2 << " dy1 "<<resPSlength_*scale<< endl
	<<" x2 " <<x2  << " dx2 "<<resPSlength_
	<<" y2 " << y2 << " dy2 "<<resPSpitch_<< endl;
  }
#endif
  if(chi2<chi2PSHV_ || chi2PSHV_<0 )
    return std::pair<double,double>(chi2,dist/1000.);
  else 
    return std::pair<double,double>(-1,-1);
}



std::pair<double,double> 
PFBlockAlgo::testLinkByVertex( const reco::PFBlockElement* elt1, 
			       const reco::PFBlockElement* elt2) const {

  double result=-1.;
  if( (elt1->trackType(reco::PFBlockElement::T_TO_NUCL) &&
       elt2->trackType(reco::PFBlockElement::T_FROM_NUCL)) ||
      (elt1->trackType(reco::PFBlockElement::T_FROM_NUCL) &&
       elt2->trackType(reco::PFBlockElement::T_TO_NUCL)) ||
      (elt1->trackType(reco::PFBlockElement::T_FROM_NUCL) &&
       elt2->trackType(reco::PFBlockElement::T_FROM_NUCL))) {
    
    NuclearInteractionRef ni1_ = elt1->nuclearRef(); 
    NuclearInteractionRef ni2_ = elt2->nuclearRef(); 
    if( ni1_.isNonnull() && ni2_.isNonnull() ) {
      if( ni1_ == ni2_ ) result= 1.0;
    }
  }
  else if (  elt1->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)  &&
	     elt2->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)  ) {
    
    if(debug_ ) std::cout << " testLinkByVertex On Conversions " << std::endl;
    
    if ( elt1->convRef().isNonnull() && elt2->convRef().isNonnull() ) {
      if(debug_ ) std::cout << " PFBlockAlgo.cc testLinkByVertex  Cconversion Refs are non null  " << std::endl;      
      if ( elt1->convRef() ==  elt2->convRef() ) {
	result=1.0;
	if(debug_ ) std::cout << " testLinkByVertex  Cconversion Refs are equal  " << std::endl;           
      }
    } 
    
  }
  else if (  elt1->trackType(reco::PFBlockElement::T_FROM_V0)  &&
             elt2->trackType(reco::PFBlockElement::T_FROM_V0)  ) {
    if(debug_ ) std::cout << " testLinkByVertex On V0 " << std::endl;
    if ( elt1->V0Ref().isNonnull() && elt2->V0Ref().isNonnull() ) {
      if(debug_ ) std::cout << " PFBlockAlgo.cc testLinkByVertex  V0 Refs are non null  " << std::endl;
      if ( elt1->V0Ref() ==  elt2->V0Ref() ) {
	result=1.0;
	if(debug_ ) std::cout << " testLinkByVertex  V0 Refs are equal  " << std::endl;
      }
    }
  }

  return std::pair<double,double>(result,0.);
}

std::pair<double,double> 
PFBlockAlgo::testTrackAndClusterByRecHit( const PFRecTrack& track, 
			       const PFCluster&  cluster,
			       bool isBrem)  const {
  
#ifdef PFLOW_DEBUG
  if( debug_ ) 
    cout<<"entering test link by rechit function"<<endl;
#endif

  //cluster position
  double clustereta  = cluster.positionREP().Eta();
  double clusterphi  = cluster.positionREP().Phi();
  double clusterX    = cluster.position().X();
  double clusterY    = cluster.position().Y();
  double clusterZ    = cluster.position().Z();

  bool barrel = false;
  bool hcal = false;
  double distance = 999999.9;

  //track extrapolation
  const reco::PFTrajectoryPoint& atVertex 
    = track.extrapolatedPoint( reco::PFTrajectoryPoint::ClosestApproach );
  const reco::PFTrajectoryPoint& atECAL 
    = track.extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax );


  //track
  double tracketa = 999999.9;
  double trackphi = 999999.9;
  double track_X  = 999999.9;
  double track_Y  = 999999.9;
  double track_Z  = 999999.9;
  double dHEta = 0.;
  double dHPhi = 0.;

  double trackPt = sqrt(atVertex.momentum().Vect().Perp2());
  if(isBrem == true) 
    trackPt = 2.;


  //retrieving resolution maps
  PFResolutionMap* mapeta;
  PFResolutionMap* mapphi;
  switch (cluster.layer()) {
  case PFLayer::ECAL_BARREL: barrel = true;
  case PFLayer::ECAL_ENDCAP:
#ifdef PFLOW_DEBUG
    if( debug_ )
      cout << "Fetching Ecal Resolution Maps"
	   << endl;
#endif
    mapeta = const_cast<PFResolutionMap*>(resMapEtaECAL_);
    mapphi = const_cast<PFResolutionMap*>(resMapPhiECAL_);

    // did not reach ecal, cannot be associated with a cluster.
    if( ! atECAL.isValid() ) return std::pair<double,double>(-1,-1);   
    
    tracketa = atECAL.positionREP().Eta();
    trackphi = atECAL.positionREP().Phi();
    track_X  = atECAL.position().X();
    track_Y  = atECAL.position().Y();
    track_Z  = atECAL.position().Z();

    distance 
      = std::sqrt( (track_X-clusterX)*(track_X-clusterX)
		  +(track_Y-clusterY)*(track_Y-clusterY)
		  +(track_Z-clusterZ)*(track_Z-clusterZ)
		   );
			           
    break;
   
  case PFLayer::HCAL_BARREL1: barrel = true; 
  case PFLayer::HCAL_ENDCAP:  
#ifdef PFLOW_DEBUG
    if( debug_ )
      cout << "Fetching Hcal Resolution Maps"
	   << endl;
#endif
    mapeta = const_cast<PFResolutionMap*>(resMapEtaHCAL_);
    mapphi = const_cast<PFResolutionMap*>(resMapPhiHCAL_);
    if(isBrem == true) 
      return std::pair<double,double>(-1,-1);
    if(isBrem == false) {
      hcal=true;
      const reco::PFTrajectoryPoint& atHCAL 
	= track.extrapolatedPoint( reco::PFTrajectoryPoint::HCALEntrance );
      const reco::PFTrajectoryPoint& atHCALExit 
	= track.extrapolatedPoint( reco::PFTrajectoryPoint::HCALExit );
      // did not reach hcal, cannot be associated with a cluster.
      if( ! atHCAL.isValid() ) return std::pair<double,double>(-1,-1);   
      
      // The link is computed between 0 and ~1 interaction length in HCAL
      dHEta = atHCALExit.positionREP().Eta()-atHCAL.positionREP().Eta();
      dHPhi = atHCALExit.positionREP().Phi()-atHCAL.positionREP().Phi(); 
      if ( dHPhi > M_PI ) dHPhi = dHPhi - 2.*M_PI;
      else if ( dHPhi < -M_PI ) dHPhi = dHPhi + 2.*M_PI; 
      tracketa = atHCAL.positionREP().Eta() + 0.1*dHEta;
      trackphi = atHCAL.positionREP().Phi() + 0.1*dHPhi;
      track_X  = atHCAL.position().X();
      track_Y  = atHCAL.position().Y();
      track_Z  = atHCAL.position().Z();
      distance 
	= -std::sqrt( (track_X-clusterX)*(track_X-clusterX)
		     +(track_Y-clusterY)*(track_Y-clusterY)
		     +(track_Z-clusterZ)*(track_Z-clusterZ)
		     );
			           
    }
    break;
  case PFLayer::PS1:
  case PFLayer::PS2:
    //Note Alex: Nothing implemented for the
    //PreShower (No resolution maps yet)
#ifdef PFLOW_DEBUG
    if( debug_ )
      cout << "No link by rechit possible for pre-shower yet!"
	   << endl;
#endif
    return std::pair<double,double>(-1,-1);
  default:
    return std::pair<double,double>(-1,-1);
  }


  // Check that, if in the endcap, the track and the cluster are on the same side !
  // PJ - 28-Feb-09
  if ( !barrel && track_Z * clusterZ < 0. ) 
    return std::pair<double,double>(-1,-1);

  double clusteretares 
    = mapeta->GetBinContent(mapeta->FindBin(clustereta, 
					    cluster.energy() ) );
  double clusterphires 
    = mapphi->GetBinContent(mapphi->FindBin(clustereta, 
					    cluster.energy() ) );
  
  
  // rec track resolution should be negligible compared 
  // calo resolution
  double trackres = 0;
  
  std::pair<double,double> lnk = computeChi2( clustereta, clusteretares, 
					      clusterphi, clusterphires, 
					      tracketa, trackres, 
					      trackphi, trackres);
  
#ifdef PFLOW_DEBUG
  double chi2 = lnk.first;
  if(debug_) cout<<"test link by rechit "<<chi2<<" "<<endl;
  if(debug_){
    cout<<" clustereta "  << clustereta << "  clusteretares "<<clusteretares
	<<" clusterphi "  << clusterphi << "  clusterphires "<<clusterphires
	<<" tracketa " << tracketa<< "  trackres "  <<trackres
	<<" trackphi " << trackphi<< "  trackres "  <<trackres << endl;
  }
#endif
  
  //Testing if Track can be linked by rechit to a cluster.
  //A cluster can be linked to a track if the extrapolated position 
  //of the track to the ECAL ShowerMax/HCAL entrance falls within 
  //the boundaries of any cell that belongs to this cluster.

  const std::vector< reco::PFRecHitFraction >& 
    fracs = cluster.recHitFractions();
  
  bool linkedbyrechit = false;
  //loop rechits
  for(unsigned int rhit = 0; rhit < fracs.size(); ++rhit){

    const reco::PFRecHitRef& rh = fracs[rhit].recHitRef();
    double fraction = fracs[rhit].fraction();
    if(fraction < 1E-4) continue;
    if(rh.isNull()) continue;
    
    //getting rechit center position
    const reco::PFRecHit& rechit_cluster = *rh;
    const math::XYZPoint& posxyz 
      = rechit_cluster.position();
    const reco::PFRecHit::REPPoint& posrep 
      = rechit_cluster.positionREP();
    
    //getting rechit corners
    const std::vector< math::XYZPoint >& 
      cornersxyz = rechit_cluster.getCornersXYZ();
    const std::vector<reco::PFRecHit::REPPoint>& corners = 
      rechit_cluster.getCornersREP();
    assert(corners.size() == 4);

    if( barrel || hcal ){ // barrel case matching in eta/phi 
                          // (and HCAL endcap too!)

      //rechit size determination 
      // blown up by 50% (HCAL) to 100% (ECAL) to include cracks & gaps
      // also blown up to account for multiple scattering at low pt.
      double rhsizeEta 
	= fabs(corners[0].Eta() - corners[2].Eta());
      double rhsizePhi 
	= fabs(corners[0].Phi() - corners[2].Phi());
      if ( rhsizePhi > M_PI ) rhsizePhi = 2.*M_PI - rhsizePhi;
      if ( hcal ) { 
	rhsizeEta = rhsizeEta * (1.50 + 0.5/fracs.size()) + 0.2*fabs(dHEta);
	rhsizePhi = rhsizePhi * (1.50 + 0.5/fracs.size()) + 0.2*fabs(dHPhi); 

      } else { 
	rhsizeEta *= 2.00 + 1.0/fracs.size()/min(1.,trackPt/2.);
	rhsizePhi *= 2.00 + 1.0/fracs.size()/min(1.,trackPt/2.); 
      }
      
#ifdef PFLOW_DEBUG
      if( debug_ ) {
	cout << rhit         << " Hcal RecHit=" 
	     << posrep.Eta() << " " 
	     << posrep.Phi() << " "
	     << rechit_cluster.energy() 
	     << endl; 
	for ( unsigned jc=0; jc<4; ++jc ) 
	  cout<<"corners "<<jc<<" "<<corners[jc].Eta()
	      <<" "<<corners[jc].Phi()<<endl;
	
	cout << "RecHit SizeEta=" << rhsizeEta
	     << " SizePhi=" << rhsizePhi << endl;
      }
#endif
    
      //distance track-rechit center
      // const math::XYZPoint& posxyz 
      // = rechit_cluster.position();
      double deta = fabs(posrep.Eta() - tracketa);
      double dphi = fabs(posrep.Phi() - trackphi);
      if ( dphi > M_PI ) dphi = 2.*M_PI - dphi;

#ifdef PFLOW_DEBUG
      if( debug_ ){
	cout << "distance=" 
	     << deta << " " 
	     << dphi << " ";
	if(deta < (rhsizeEta/2.) && dphi < (rhsizePhi/2.))
	  cout << " link here !" << endl;
	else cout << endl;
      }
#endif
      
      if(deta < (rhsizeEta/2.) && dphi < (rhsizePhi/2.)){ 
	linkedbyrechit = true;
	break;
      }
    }
    else { //ECAL & PS endcap case, matching in X,Y

#ifdef PFLOW_DEBUG
      if( debug_ ){
	const math::XYZPoint& posxyz 
	  = rechit_cluster.position();
       
	cout << "RH " << posxyz.X()
	     << " "   << posxyz.Y()
	     << endl;
	
	cout << "TRACK " << track_X
	     << " "      << track_Y
	     << endl;
      }
#endif
      
      double x[5];
      double y[5];

      for ( unsigned jc=0; jc<4; ++jc ) {
	math::XYZPoint cornerposxyz = cornersxyz[jc];
	x[jc] = cornerposxyz.X() + (cornerposxyz.X()-posxyz.X())
	                         * (1.00+0.50/fracs.size()/min(1.,trackPt/2.));
	y[jc] = cornerposxyz.Y() + (cornerposxyz.Y()-posxyz.Y())
                        	 * (1.00+0.50/fracs.size()/min(1.,trackPt/2.));

#ifdef PFLOW_DEBUG
	if( debug_ ){
	  cout<<"corners "<<jc
	      << " " << cornerposxyz.X()
	      << " " << cornerposxyz.Y()
	      << endl;
	}
#endif
      }//loop corners

      //need to close the polygon in order to
      //use the TMath::IsInside fonction from root lib
      x[4] = x[0];
      y[4] = y[0];
  
      //Check if the extrapolation point of the track falls 
      //within the rechit boundaries
      bool isinside = TMath::IsInside(track_X,
				      track_Y,
				      5,x,y);
      
      if( isinside ){
	linkedbyrechit = true;
	break;
      }
    }//
    
  }//loop rechits
  
  if( linkedbyrechit ) {
#ifdef PFLOW_DEBUG
    if( debug_ ) 
      cout << "Track and Cluster LINKED BY RECHIT" << endl;
#endif
    //if ( distance > 40. || distance < -100. ) 
    //  std::cout << "Distance = " << distance 
    //	<< ", Barrel/Hcal/Brem ? " << barrel << " " << hcal << " " << isBrem << std::endl
    //<< " Cluster " << clusterX << " " << clusterY << " " << clusterZ << " " << clustereta << " " << clusterphi << std::endl
    //		<< " Track   " << track_X << " " << track_Y << " " << track_Z << " " << tracketa << " " << trackphi << std::endl;
    
    return lnk;
  } else {
    return std::pair<double,double>(-1,-1);
  }

}

std::pair<double,double> 
PFBlockAlgo::testECALAndPSByRecHit( const PFCluster& clusterECAL, 
				    const PFCluster& clusterPS)  const {

  // Check that clusterECAL is in ECAL endcap and that clusterPS is a preshower cluster
  if ( clusterECAL.layer() != PFLayer::ECAL_ENDCAP ||
       ( clusterPS.layer() != PFLayer::PS1 && 
	 clusterPS.layer() != PFLayer::PS2 ) ) return std::pair<double,double>(-1,-1);

#ifdef PFLOW_DEBUG
  if( debug_ ) 
    cout<<"entering test link by rechit function for ECAL and PS"<<endl;
#endif

  //ECAL cluster position
  double zECAL  = clusterECAL.position().Z();
  double xECAL  = clusterECAL.position().X();
  double yECAL  = clusterECAL.position().Y();

  // PS cluster position, extrapolated to ECAL
  double zPS = clusterPS.position().Z();
  double xPS = clusterPS.position().X(); //* zECAL/zPS;
  double yPS = clusterPS.position().Y(); //* zECAL/zPS;
// MDN jan09 : check that zEcal and zPs have the same sign
	if (zECAL*zPS <0.) return std::pair<double,double>(-1,-1);
  double deltaX = 0.;
  double deltaY = 0.;
  double sqr12 = std::sqrt(12.);
  switch (clusterPS.layer()) {
  case PFLayer::PS1:
    // vertical strips, measure x with pitch precision
    deltaX = resPSpitch_ * sqr12;
    deltaY = resPSlength_ * sqr12;
    break;
  case PFLayer::PS2:
    // horizontal strips, measure y with pitch precision
    deltaY = resPSpitch_ * sqr12;
    deltaX = resPSlength_ * sqr12;
    break;
  default:
    break;
  }

  // Get the rechits
  const std::vector< reco::PFRecHitFraction >&  fracs = clusterECAL.recHitFractions();
  bool linkedbyrechit = false;
  //loop rechits
  for(unsigned int rhit = 0; rhit < fracs.size(); ++rhit){

    const reco::PFRecHitRef& rh = fracs[rhit].recHitRef();
    double fraction = fracs[rhit].fraction();
    if(fraction < 1E-4) continue;
    if(rh.isNull()) continue;

    //getting rechit center position
    const reco::PFRecHit& rechit_cluster = *rh;
    
    //getting rechit corners
    const std::vector< math::XYZPoint >&  corners = rechit_cluster.getCornersXYZ();
    assert(corners.size() == 4);
    
    const math::XYZPoint& posxyz = rechit_cluster.position() * zPS/zECAL;
#ifdef PFLOW_DEBUG
    if( debug_ ){
      cout << "Ecal rechit " << posxyz.X() << " "   << posxyz.Y() << endl;
      cout << "PS cluster  " << xPS << " " << yPS << endl;
    }
#endif
    
    double x[5];
    double y[5];
    for ( unsigned jc=0; jc<4; ++jc ) {
      // corner position projected onto the preshower
      math::XYZPoint cornerpos = corners[jc] * zPS/zECAL;
      // Inflate the size by the size of the PS strips, and by 5% to include ECAL cracks.
      x[jc] = cornerpos.X() + (cornerpos.X()-posxyz.X()) * (0.05 +1.0/fabs((cornerpos.X()-posxyz.X()))*deltaX/2.);
      y[jc] = cornerpos.Y() + (cornerpos.Y()-posxyz.Y()) * (0.05 +1.0/fabs((cornerpos.Y()-posxyz.Y()))*deltaY/2.);
      
#ifdef PFLOW_DEBUG
      if( debug_ ){
	cout<<"corners "<<jc
	    << " " << cornerpos.X() << " " << x[jc] 
	    << " " << cornerpos.Y() << " " << y[jc]
	    << endl;
      }
#endif
    }//loop corners
    
    //need to close the polygon in order to
    //use the TMath::IsInside fonction from root lib
    x[4] = x[0];
    y[4] = y[0];
    
    //Check if the extrapolation point of the track falls 
    //within the rechit boundaries
    bool isinside = TMath::IsInside(xPS,yPS,5,x,y);
      
    if( isinside ){
      linkedbyrechit = true;
      break;
    }

  }//loop rechits
  
  if( linkedbyrechit ) {
    if( debug_ ) cout << "Cluster PS and Cluster ECAL LINKED BY RECHIT" << endl;
    std::pair<double,double> 
      lnk = computeChi2( xECAL/1000.,1.,yECAL/1000.,1.,
			 xPS/1000.  ,1.,yPS/1000.  ,1.);    
    return lnk;
  } else { 
    return std::pair<double,double>(-1,-1);
  }

}

std::pair<double,double> 
PFBlockAlgo::computeChi2( double eta1, double reta1, 
			  double phi1, double rphi1, 
			  double eta2, double reta2, 
			  double phi2, double rphi2 ) const {
  
  double phicor = Utils::mpi_pi(phi1 - phi2);
  
  double chi2 =  
    (eta1 - eta2)*(eta1 - eta2) / ( reta1*reta1+ reta2*reta2 ) +
    phicor*phicor / ( rphi1*rphi1+ rphi2*rphi2 );

  double dist = std::sqrt( (eta1 - eta2)*(eta1 - eta2) 
			  + phicor*phicor);

  return std::pair<double,double>(chi2,dist);

}


std::pair<double,double> 
PFBlockAlgo::testHFEMAndHFHADByRecHit(const reco::PFCluster& clusterHFEM, 
					   const reco::PFCluster& clusterHFHAD) const {
  
  math::XYZPoint posxyzEM = clusterHFEM.position();
  math::XYZPoint posxyzHAD = clusterHFHAD.position();

  double dX = posxyzEM.X()-posxyzHAD.X();
  double dY = posxyzEM.Y()-posxyzHAD.Y();
  double sameZ = posxyzEM.Z()*posxyzHAD.Z();


  std::pair<double,double> badLink(-1,-1);
 
  if(sameZ<0) return badLink;

  double dist2 = dX*dX + dY*dY; 

  if( dist2<0.1 ) {
    // less than one mm
    double dist = sqrt( dist2 );
    return std::pair<double,double>(dist, dist);;
  }
  else 
    return std::pair<double,double>(-1,-1);

}




void 
PFBlockAlgo::checkMaskSize( const reco::PFRecTrackCollection& tracks,
			    const reco::GsfPFRecTrackCollection& gsftracks, 
			    const reco::PFClusterCollection&  ecals,
			    const reco::PFClusterCollection&  hcals,
			    const reco::PFClusterCollection&  hfems,
			    const reco::PFClusterCollection&  hfhads,
			    const reco::PFClusterCollection&  pss, 
			    const Mask& trackMask, 
			    const Mask& gsftrackMask,  
			    const Mask& ecalMask,
			    const Mask& hcalMask, 
			    const Mask& hfemMask,
			    const Mask& hfhadMask,		      
			    const Mask& psMask ) const {

  if( !trackMask.empty() && 
      trackMask.size() != tracks.size() ) {
    string err = "PFBlockAlgo::setInput: ";
    err += "The size of the track mask is different ";
    err += "from the size of the track vector.";
    throw std::length_error( err.c_str() );
  }

  if( !gsftrackMask.empty() && 
      gsftrackMask.size() != gsftracks.size() ) {
    string err = "PFBlockAlgo::setInput: ";
    err += "The size of the gsf track mask is different ";
    err += "from the size of the gsftrack vector.";
    throw std::length_error( err.c_str() );
  }

  if( !ecalMask.empty() && 
      ecalMask.size() != ecals.size() ) {
    string err = "PFBlockAlgo::setInput: ";
    err += "The size of the ecal mask is different ";
    err += "from the size of the ecal clusters vector.";
    throw std::length_error( err.c_str() );
  }
  
  if( !hcalMask.empty() && 
      hcalMask.size() != hcals.size() ) {
    string err = "PFBlockAlgo::setInput: ";
    err += "The size of the hcal mask is different ";
    err += "from the size of the hcal clusters vector.";
    throw std::length_error( err.c_str() );
  }

  if( !hfemMask.empty() && 
      hfemMask.size() != hfems.size() ) {
    string err = "PFBlockAlgo::setInput: ";
    err += "The size of the hfem mask is different ";
    err += "from the size of the hfem clusters vector.";
    throw std::length_error( err.c_str() );
  }

  if( !hfhadMask.empty() && 
      hfhadMask.size() != hfhads.size() ) {
    string err = "PFBlockAlgo::setInput: ";
    err += "The size of the hfhad mask is different ";
    err += "from the size of the hfhad clusters vector.";
    throw std::length_error( err.c_str() );
  }

  if( !psMask.empty() && 
      psMask.size() != pss.size() ) {
    string err = "PFBlockAlgo::setInput: ";
    err += "The size of the ps mask is different ";
    err += "from the size of the ps clusters vector.";
    throw std::length_error( err.c_str() );
  }
  
}


std::ostream& operator<<(std::ostream& out, const PFBlockAlgo& a) {
  if(! out) return out;
  
  out<<"====== Particle Flow Block Algorithm ======= ";
  out<<endl;
  out<<"resMapEtaECAL "<<a.resMapEtaECAL_->GetMapFile()<<endl;
  out<<"resMapPhiECAL "<<a.resMapPhiECAL_->GetMapFile()<<endl;
  out<<"resMapEtaHCAL "<<a.resMapEtaHCAL_->GetMapFile()<<endl;
  out<<"resMapPhiHCAL "<<a.resMapPhiHCAL_->GetMapFile()<<endl;
  out<<"chi2TrackECAL "<<a.chi2TrackECAL_<<endl;
  out<<"chi2TrackHCAL "<<a.chi2TrackHCAL_<<endl;
  out<<"chi2ECALHCAL  "<<a.chi2ECALHCAL_<<endl;
  out<<"chi2PSECAL    "<<a.chi2PSECAL_  <<endl;
  out<<"chi2PSTRACK   "<<a.chi2PSTrack_ <<endl;
  out<<"chi2PSHV      "<<a.chi2PSHV_    <<endl;
  out<<endl;
  out<<"number of unassociated elements : "<<a.elements_.size()<<endl;
  out<<endl;
  
  for(PFBlockAlgo::IEC ie = a.elements_.begin(); 
      ie != a.elements_.end(); ie++) {
    out<<"\t"<<**ie <<endl;
  }

  
  //   const PFBlockCollection& blocks = a.blocks();

  const std::auto_ptr< reco::PFBlockCollection >& blocks
    = a.blocks(); 
    
  if(!blocks.get() ) {
    out<<"blocks already transfered"<<endl;
  }
  else {
    out<<"number of blocks : "<<blocks->size()<<endl;
    out<<endl;
    
    for(PFBlockAlgo::IBC ib=blocks->begin(); 
	ib != blocks->end(); ib++) {
      out<<(*ib)<<endl;
    }
  }

  return out;
}

bool 
PFBlockAlgo::goodPtResolution( const reco::TrackRef& trackref) {

  double P = trackref->p();
  double Pt = trackref->pt();
  double DPt = trackref->ptError();
  unsigned int NHit = trackref->hitPattern().trackerLayersWithMeasurement();
  unsigned int NLostHit = trackref->hitPattern().trackerLayersWithoutMeasurement();
  unsigned int LostHits = trackref->numberOfLostHits();
  double sigmaHad = sqrt(1.20*1.20/P+0.06*0.06) / (1.+LostHits);

  // iteration 1,2,3,4 correspond to algo = 1/5,6,7,8
  unsigned int Algo = trackref->algo() < 5 ? 
    trackref->algo()-1 : trackref->algo()-5;

  // Temporary : Reject all tracking iteration beyond 5th step. 
  if ( Algo > 4 ) return false;
 
  if (debug_) cout << " PFBlockAlgo: PFrecTrack->Track Pt= "
		   << Pt << " DPt = " << DPt << endl;
  if ( DPt/Pt > DPtovPtCut_[Algo]*sigmaHad || 
       NHit < NHitCut_[Algo] || 
       (Algo == 3 && LostHits != 0) ) {
    if (debug_) cout << " PFBlockAlgo: skip badly measured track"
		     << ", P = " << P 
		     << ", Pt = " << Pt 
		     << " DPt = " << DPt 
		     << ", N(hits) = " << NHit << " (Lost : " << LostHits << "/" << NLostHit << ")"
		     << ", Algo = " << Algo
		     << endl;
    if (debug_) cout << " cut is DPt/Pt < " << DPtovPtCut_[Algo] * sigmaHad << endl;
    if (debug_) cout << " cut is NHit >= " << NHitCut_[Algo] << endl;
    /*
    std::cout << "Track REJECTED : ";
    std::cout << ", P = " << P 
	      << ", Pt = " << Pt 
	      << " DPt = " << DPt 
	      << ", N(hits) = " << NHit << " (Lost : " << LostHits << "/" << NLostHit << ")"
	      << ", Algo = " << Algo
	      << std::endl;
    */
    return false;
  }
  /*
  std::cout << "Track Accepted : ";
  std::cout << ", P = " << P 
       << ", Pt = " << Pt 
       << " DPt = " << DPt 
       << ", N(hits) = " << NHit << " (Lost : " << LostHits << "/" << NLostHit << ")"
       << ", Algo = " << Algo
       << std::endl;
  */
  return true;
}

void 
PFBlockAlgo::fillSecondaries( const reco::PFNuclearInteractionRef& nuclref ) {
  // loop on secondaries
  for( reco::PFNuclearInteraction::pfTrackref_iterator
	 pftkref = nuclref->secPFRecTracks_begin();
       pftkref != nuclref->secPFRecTracks_end(); ++pftkref) {
    if( !goodPtResolution( (*pftkref)->trackRef() ) ) continue;
    reco::PFBlockElement *secondaryTrack 
      = new reco::PFBlockElementTrack( *pftkref );
    secondaryTrack->setNuclearRef( nuclref->nuclInterRef(), 
				   reco::PFBlockElement::T_FROM_NUCL );
            
    elements_.push_back( secondaryTrack ); 
  }
}

int 
PFBlockAlgo::niAssocToTrack( const reco::TrackRef& primTkRef,
			     const edm::Handle<reco::PFNuclearInteractionCollection>& nuclh) const {
  if( nuclh.isValid() ) {
    // look for nuclear interaction associated to primTkRef
    for( unsigned int k=0; k<nuclh->size(); ++k) {
      const edm::RefToBase< reco::Track >& trk = nuclh->at(k).primaryTrack();
      if( trk.castTo<reco::TrackRef>() == primTkRef) return k;
    }
    return -1; // not found
  }
  else return -1;
}

// duplication due to limitation of LCG reflex dict from header file
int 
PFBlockAlgo::niAssocToTrack( const reco::TrackRef& primTkRef,
			     const edm::OrphanHandle<reco::PFNuclearInteractionCollection>& nuclh) const {
  if( nuclh.isValid() ) {
    // look for nuclear interaction associated to primTkRef
    for( unsigned int k=0; k<nuclh->size(); ++k) {
      const edm::RefToBase< reco::Track >& trk = nuclh->at(k).primaryTrack();
      if( trk.castTo<reco::TrackRef>() == primTkRef) return k;
    }
    return -1; // not found
  }
  else return -1;
}

int 
PFBlockAlgo::muAssocToTrack( const reco::TrackRef& trackref,
			     const edm::Handle<reco::MuonCollection>& muonh) const {
  if(muonh.isValid() ) {
    for(unsigned j=0;j<muonh->size(); j++) {
      reco::MuonRef muonref( muonh, j );
      if (muonref->track().isNonnull()) 
	if( muonref->track() == trackref ) return j;
    }
  }
  return -1; // not found
}

int 
PFBlockAlgo::v0AssocToTrack( const reco::TrackRef& primTkRef,
			     const edm::Handle<reco::PFV0Collection>& v0) const {
  if( v0.isValid() ) {
    // look for v0 associated to primTkRef
    for( unsigned int k=0; k<v0->size(); ++k) {
      for (uint itk=0;itk<(*v0)[k].Tracks().size();itk++){ 
        if( (*v0)[k].Tracks()[itk] == primTkRef) return k;
      }
    }
    return -1; // not found
  }
  else return -1;
}

// duplication due to limitation of LCG reflex dict from header file
int 
PFBlockAlgo::v0AssocToTrack( const reco::TrackRef& primTkRef,
			     const edm::OrphanHandle<reco::PFV0Collection>& v0) const {
  if( v0.isValid() ) {
    // look for v0 associated to primTkRef
    for( unsigned int k=0; k<v0->size(); ++k) {
      for (uint itk=0;itk<(*v0)[k].Tracks().size();itk++){
        if( (*v0)[k].Tracks()[itk] == primTkRef) return k;
      }
     
    }
    return -1; // not found
  }
  else return -1;
}


int 
PFBlockAlgo::muAssocToTrack( const reco::TrackRef& trackref,
			     const edm::OrphanHandle<reco::MuonCollection>& muonh) const {
  if(muonh.isValid() ) {
    for(unsigned j=0;j<muonh->size(); j++) {
      reco::MuonRef muonref( muonh, j );
      if (muonref->track().isNonnull())
	if( muonref->track() == trackref ) return j;
    }
  }
  return -1; // not found
}


void 
PFBlockAlgo::checkNuclearLinks( reco::PFBlock& block ) const {
  // method which removes link between primary tracks and clusters
  // if at least one of the associated secondary tracks is closer 
  // to these same clusters

  typedef std::multimap<double, unsigned>::iterator IE;

  const edm::OwnVector< reco::PFBlockElement >& els = block.elements();
  // loop on all elements != TRACK
  for( unsigned i1=0; i1 != els.size(); ++i1 ) {
    if( els[i1].type() == PFBlockElement::TRACK ) continue;
    std::multimap<double, unsigned> assocTracks;
    // get associated tracks
    block.associatedElements( i1,  block.linkData(),
			      assocTracks,
			      reco::PFBlockElement::TRACK,
			      reco::PFBlock::LINKTEST_ALL );
    for( IE ie = assocTracks.begin(); ie != assocTracks.end(); ++ie) {
      double   chi2prim  = ie->first;
      unsigned iprim     = ie->second;
      // if this track a primary track (T_To_NUCL)
      if( els[iprim].trackType(PFBlockElement::T_TO_NUCL) )  {
	std::multimap<double, unsigned> secTracks; 
	// get associated secondary tracks
	block.associatedElements( iprim,  block.linkData(),
				  secTracks,
				  reco::PFBlockElement::TRACK,
				  reco::PFBlock::LINKTEST_ALL );
	for( IE ie2 = secTracks.begin(); ie2 != secTracks.end(); ++ie2) { 
	  unsigned isec = ie2->second;
	  double chi2sec_rechit = block.chi2( i1, isec, block.linkData(),
					      PFBlock::LINKTEST_RECHIT );
	  double chi2sec_chi2 = block.chi2( i1, isec, block.linkData(),
					    PFBlock::LINKTEST_CHI2 );
	  double chi2sec;

	  // at present associatedElement return first the chi2 by chi2
	  // maybe in the futur return the min between chi2 and rechit! 
	  if( chi2sec_chi2 > 0) chi2sec = chi2sec_chi2;
	  else chi2sec=chi2sec_rechit;

	  // if one secondary tracks has a chi2 < chi2prim 
	  // remove the link between the element and the primary
	  if( chi2sec < 0 ) continue;
	  else if( chi2sec < chi2prim ) { 
	    block.setLink( i1, iprim, -1, -1, block.linkData(),
			   PFBlock::LINKTEST_CHI2 );
	    block.setLink( i1, iprim, -1, -1, block.linkData(),
			   PFBlock::LINKTEST_RECHIT );
	    continue;
	  }
	} // loop on all associated secondary tracks
      } // test if track is T_TO_NUCL
                             
    } // loop on all associated tracks
  } // loop on all elements
}


