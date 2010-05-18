#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/Utils.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFResolutionMap.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h" // gouzevitch

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
  DPtovPtCut_(std::vector<double>(4,static_cast<double>(999.))),
  NHitCut_(std::vector<unsigned int>(4,static_cast<unsigned>(0))), 
  resPSpitch_ (0),
  resPSlength_ (0),
  debug_(false) {}



void PFBlockAlgo::setParameters( std::vector<double>& DPtovPtCut,
				 std::vector<unsigned int>& NHitCut ) {
  
  DPtovPtCut_    = DPtovPtCut;
  NHitCut_       = NHitCut;
  double strip_pitch = 0.19;
  double strip_length = 6.1;
  resPSpitch_    = strip_pitch/sqrt(12.);
  resPSlength_   = strip_length/sqrt(12.);

}

PFBlockAlgo::~PFBlockAlgo() {

#ifdef PFLOW_DEBUG
  if(debug_)
    cout<<"~PFBlockAlgo - number of remaining elements: "
	<<elements_.size()<<endl;
#endif
  
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
    double dist = -1;
    PFBlock::LinkTest linktest = PFBlock::LINKTEST_RECHIT;
    link( *last, *next, linktype, linktest, dist ); 

   
    if(dist<-0.5) {
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
      
      double dist = -1;
      
      bool linked = false;
      PFBlock::LinkTest linktest 
	= PFBlock::LINKTEST_RECHIT; 

      // are these elements already linked ?
      // this can be optimized

      for( unsigned il=0; il<links.size(); il++ ) {
	if( (links[il].element1() == i1 && 
	     links[il].element2() == i2) || 
	    (links[il].element1() == i2 && 
	     links[il].element2() == i1) ) { // yes
	  
	  dist = links[il].dist();
	  linked = true;

	  //modif-beg	  
	  //retrieve type of test used to get distance
	  linktest = links[il].test();
#ifdef PFLOW_DEBUG
	  if( debug_ )
	    cout << "Reading link vector: linktest used=" 
		 << linktest 
		 << " distance = " << dist 
		 << endl; 
#endif
	  //modif-end
	  
	  break;
	} 
      }
      
      if(!linked) {
	PFBlockLink::Type linktype = PFBlockLink::NONE;
	link( & els[i1], & els[i2], linktype, linktest, dist);
      }

      //loading link data according to link test used: RECHIT 
      //block.setLink( i1, i2, chi2, block.linkData() );
#ifdef PFLOW_DEBUG
      if( debug_ )
	cout << "Setting link between elements " << i1 << " and " << i2
	     << " of dist =" << dist << " computed from link test "
	     << linktest << endl;
#endif
      block.setLink( i1, i2, dist, block.linkData(), linktest );
    }
  }

  checkDisplacedVertexLinks( block );
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
		   double& dist) const {
  


  dist=-1.;
  linktest = PFBlock::LINKTEST_RECHIT; //rechit by default 

  PFBlockElement::Type type1 = el1->type();
  PFBlockElement::Type type2 = el2->type();

  if( type1==type2 ) {
    // cannot link 2 elements of the same type. 
    // except if the elements are 2 tracks
    if( type1!=PFBlockElement::TRACK && type1!=PFBlockElement::GSF) return;
    // cannot link two primary tracks  (except if they come from a V0)
    if( type1 ==PFBlockElement::TRACK) {
      if ( !el1->isLinkedToDisplacedVertex() || !el2->isLinkedToDisplacedVertex()) 
      return;
    }
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
      // PJ - 14-May-09 : A link by rechit is needed here !
      dist = testTrackAndPS( *trackref, *clusterref );
      linktest = PFBlock::LINKTEST_RECHIT;
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
      dist = testTrackAndClusterByRecHit( *trackref, *clusterref );
      linktest = PFBlock::LINKTEST_RECHIT;

      if ( debug_ ) { 
	if( dist > 0. ) { 
	  std::cout << " Here a link has been established"
		    << " between a track an Ecal with dist  " 
		    << dist <<  std::endl;
	} else {
	  std::cout << " No link found " << std::endl;
	}
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
      dist = testTrackAndClusterByRecHit( *trackref, *clusterref );
      linktest = PFBlock::LINKTEST_RECHIT;      
      break;
    }
  case PFBlockLink::ECALandHCAL:
    {
      //       cout<<"ECALandHCAL"<<endl;
      PFClusterRef  ecalref = lowEl->clusterRef();
      PFClusterRef  hcalref = highEl->clusterRef();
      assert( !ecalref.isNull() );
      assert( !hcalref.isNull() );
      // PJ - 14-May-09 : A link by rechit is needed here !
      // dist = testECALAndHCAL( *ecalref, *hcalref );
      dist = -1.;
      linktest = PFBlock::LINKTEST_RECHIT;
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
      dist = testECALAndPSByRecHit( *ecalref, *psref );
      linktest = PFBlock::LINKTEST_RECHIT;      
      break;
    }
  case PFBlockLink::PS1andPS2:
    {
      PFClusterRef  ps1ref = lowEl->clusterRef();
      PFClusterRef  ps2ref = highEl->clusterRef();
      assert( !ps1ref.isNull() );
      assert( !ps2ref.isNull() );
      // PJ - 14-May-09 : A link by rechit is needed here !
      // dist = testPS1AndPS2( *ps1ref, *ps2ref );
      dist = -1.;
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
  case PFBlockLink::TRACKandTRACK:
    {
      if(debug_ ) 
	cout<<"TRACKandTRACK"<<endl;
      dist = testLinkByVertex(lowEl, highEl);
      if(debug_ ) 
	std::cout << " PFBlockLink::TRACKandTRACK dist " << dist << std::endl;
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
  case PFBlockLink::ECALandGSF:
    {
      PFClusterRef  clusterref = lowEl->clusterRef();
      assert( !clusterref.isNull() );
      const reco::PFBlockElementGsfTrack *  GsfEl =  dynamic_cast<const reco::PFBlockElementGsfTrack*>(highEl);
      const PFRecTrack * myTrack =  &(GsfEl->GsftrackPF());
      dist = testTrackAndClusterByRecHit( *myTrack, *clusterref );
      linktest = PFBlock::LINKTEST_RECHIT;
      
      if ( debug_ ) {
	if ( dist > 0. ) {
	  std::cout << " Here a link has been established" 
		    << " between a GSF track an Ecal with dist  " 
		    << dist <<  std::endl;
	} else {
	  if(debug_ ) std::cout << " No link found " << std::endl;
	}
      }
      break;
    }
  case PFBlockLink::TRACKandGSF:
    {
      PFRecTrackRef trackref = lowEl->trackRefPF();
      assert( !trackref.isNull() );
      const reco::PFBlockElementGsfTrack *  GsfEl =  
	dynamic_cast<const reco::PFBlockElementGsfTrack*>(highEl);
      GsfPFRecTrackRef gsfref = GsfEl->GsftrackRefPF();
      reco::TrackRef kftrackref= (*trackref).trackRef();
      assert( !gsfref.isNull() );
      PFRecTrackRef refkf = (*gsfref).kfPFRecTrackRef();
      if(refkf.isNonnull()) {
	reco::TrackRef gsftrackref = (*refkf).trackRef();
	if (gsftrackref.isNonnull()&&kftrackref.isNonnull()) {
	  if (kftrackref == gsftrackref) { 
	    dist = 0.001;
	  } else { 
	    dist = -1.;
	  }
	} else { 
	  dist = -1.;
	}
      } else {
	dist = -1.;
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
	dist = 0.001;
      } else { 
	dist = -1.;
      }
      break;
    }
  case PFBlockLink::GSFandGSF:
    {
      const reco::PFBlockElementGsfTrack * lowGsfEl  =  
	dynamic_cast<const reco::PFBlockElementGsfTrack*>(lowEl);
      const reco::PFBlockElementGsfTrack * highGsfEl  =  
	dynamic_cast<const reco::PFBlockElementGsfTrack*>(highEl);
      
      GsfPFRecTrackRef lowgsfref = lowGsfEl->GsftrackRefPF();
      GsfPFRecTrackRef highgsfref = highGsfEl->GsftrackRefPF();
      assert( !lowgsfref.isNull() );
      assert( !highgsfref.isNull() );
      
      if( (lowGsfEl->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) == false && 
	   highGsfEl->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)) ||
	  (highGsfEl->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) == false && 
	   lowGsfEl->trackType(reco::PFBlockElement::T_FROM_GAMMACONV))) {
	if(lowgsfref->trackId() == highgsfref->trackId()) {
	  dist = 0.001;
	}
	else {
	  dist = -1.;
	}
      }
      break;
    }
  case PFBlockLink::ECALandBREM:
    {
      PFClusterRef  clusterref = lowEl->clusterRef();
      assert( !clusterref.isNull() );
      const reco::PFBlockElementBrem * BremEl =  
	dynamic_cast<const reco::PFBlockElementBrem*>(highEl);
      const PFRecTrack * myTrack = &(BremEl->trackPF());
      /*
      double DP = (BremEl->DeltaP())*(-1.);
      double SigmaDP = BremEl->SigmaDeltaP();
      double SignBremDp = DP/SigmaDP;
      */
      bool isBrem = true;
      dist = testTrackAndClusterByRecHit( *myTrack, *clusterref, isBrem);
      if( debug_ && dist > 0. ) 
	std::cout << "ECALandBREM: dist testTrackAndClusterByRecHit " 
		  << dist << std::endl;
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
  case PFBlockLink::PS1andGSF:
  case PFBlockLink::PS2andGSF:
    {
      PFClusterRef  psref = lowEl->clusterRef();
      assert( !psref.isNull() );
      const reco::PFBlockElementGsfTrack *  GsfEl =  dynamic_cast<const reco::PFBlockElementGsfTrack*>(highEl);
      const PFRecTrack * myTrack =  &(GsfEl->GsftrackPF());
      // PJ - 14-May-09 : A link by rechit is needed here !
      dist = testTrackAndPS( *myTrack, *psref );
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
  case PFBlockLink::PS1andBREM:
  case PFBlockLink::PS2andBREM:
    {
      PFClusterRef  psref = lowEl->clusterRef();
      assert( !psref.isNull() );
      const reco::PFBlockElementBrem * BremEl =  dynamic_cast<const reco::PFBlockElementBrem*>(highEl);
      const PFRecTrack * myTrack = &(BremEl->trackPF());
      // PJ - 14-May-09 : A link by rechit is needed here !
      dist = testTrackAndPS( *myTrack, *psref );
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
  case PFBlockLink::HCALandGSF:
    {
      PFClusterRef  clusterref = lowEl->clusterRef();
      assert( !clusterref.isNull() );
      const reco::PFBlockElementGsfTrack *  GsfEl =  dynamic_cast<const reco::PFBlockElementGsfTrack*>(highEl);
      const PFRecTrack * myTrack =  &(GsfEl->GsftrackPF());
      dist = testTrackAndClusterByRecHit( *myTrack, *clusterref );
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
  case PFBlockLink::HCALandBREM:
    {
      PFClusterRef  clusterref = lowEl->clusterRef();
      assert( !clusterref.isNull() );
      const reco::PFBlockElementBrem * BremEl =  dynamic_cast<const reco::PFBlockElementBrem*>(highEl);
      const PFRecTrack * myTrack = &(BremEl->trackPF());
      bool isBrem = true;
      dist = testTrackAndClusterByRecHit( *myTrack, *clusterref, isBrem);
      break;
    }
  case PFBlockLink::HFEMandHFHAD:
    {
      // cout<<"HFEMandHFHAD"<<endl;
      PFClusterRef  eref = lowEl->clusterRef();
      PFClusterRef  href = highEl->clusterRef();
      assert( !eref.isNull() );
      assert( !href.isNull() );
      dist = testHFEMAndHFHADByRecHit( *eref, *href );
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
      
      break;
    }
  default:
    dist = -1.;
    linktest = PFBlock::LINKTEST_RECHIT;
    // cerr<<"link type not implemented yet: 0x"<<hex<<linktype<<dec<<endl;
    // assert(0);
    return;
  }
}

double
PFBlockAlgo::testTrackAndPS(const PFRecTrack& track, 
			    const PFCluster& ps)  const {

#ifdef PFLOW_DEBUG
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
  if( ! atPS.isValid() ) return -1.;   
  
  double trackx = atPS.position().X();
  double tracky = atPS.position().Y();
  double trackz = atPS.position().Z(); // MDN jan 09
  
  // ps position  x, y
  double psx = ps.position().X();
  double psy = ps.position().Y();
  // MDN Jan 09: check that trackz and psz have the same sign
  double psz = ps.position().Z();
  if( trackz*psz < 0.) return -1.; 
  
  // double chi2 = (psx-trackx)*(psx-trackx)/(dx*dx + trackresolx*trackresolx)
  //  + (psy-tracky)*(psy-tracky)/(dy*dy + trackresoly*trackresoly);

  double dist = std::sqrt( (psx-trackx)*(psx-trackx)
			 + (psy-tracky)*(psy-tracky));  
  if(debug_) cout<<"testTrackAndPS "<< dist <<" "<<endl;
  if(debug_){
    cout<<" trackx " << trackx 
	<<" tracky " << tracky 
	<<" psx "    <<  psx   
	<<" psy "    << psy    
	<< endl;
  }
#endif
  
  // Return -1. as long as no link by rechit is available
  return -1.;
}

double
PFBlockAlgo::testECALAndHCAL(const PFCluster& ecal, 
			     const PFCluster& hcal)  const {
  
  //   cout<<"entering testECALAndHCAL"<<endl;
  
  /*
  double dist = 
    computeDist( ecal.positionREP().Eta(),
		 ecal.positionREP().Phi(), 
		 hcal.positionREP().Eta(), 
		 hcal.positionREP().Phi() );
  */

#ifdef PFLOW_DEBUG
  if(debug_) cout<<"testECALAndHCAL "<< dist <<" "<<endl;
  if(debug_){
    cout<<" ecaleta " << ecal.positionREP().Eta()
	<<" ecalphi " << ecal.positionREP().Phi()
	<<" hcaleta " << hcal.positionREP().Eta()
	<<" hcalphi " << hcal.positionREP().Phi()
  }
#endif

  // Need to implement a link by RecHit
  return -1.;
}



double
PFBlockAlgo::testPS1AndPS2(const PFCluster& ps1, 
			   const PFCluster& ps2)  const {
  
#ifdef PFLOW_DEBUG
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
  if (z1*z2<0.) -1.;
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
  
  // double chi2 = (x2-x1atPS2)*(x2-x1atPS2)/(dx1dx1 + dx2dx2) 
  //  + (y2-y1atPS2)*(y2-y1atPS2)/(dy1dy1 + dy2dy2);
  
  double dist = std::sqrt( (x2-x1atPS2)*(x2-x1atPS2)
			 + (y2-y1atPS2)*(y2-y1atPS2));
    
  if(debug_) cout<<"testPS1AndPS2 "<<dist<<" "<<endl;
  if(debug_){
    cout<<" x1atPS2 "<< x1atPS2 << " dx1 "<<resPSpitch_*scale
	<<" y1atPS2 "<< y1atPS2 << " dy1 "<<resPSlength_*scale<< endl
	<<" x2 " <<x2  << " dx2 "<<resPSlength_
	<<" y2 " << y2 << " dy2 "<<resPSpitch_<< endl;
  }
#endif

  // Need a link by rechit here
  return -1.; 
}



double
PFBlockAlgo::testLinkByVertex( const reco::PFBlockElement* elt1, 
			       const reco::PFBlockElement* elt2) const {

  //  cout << "Test link by vertex between" << endl << *elt1 << endl << " and " << endl << *elt2 << endl;

  double result=-1.;

  reco::PFBlockElement::TrackType T_TO_DISP = reco::PFBlockElement::T_TO_DISP;
  reco::PFBlockElement::TrackType T_FROM_DISP = reco::PFBlockElement::T_FROM_DISP;
  PFDisplacedTrackerVertexRef ni1_TO_DISP = elt1->displacedVertexRef(T_TO_DISP);
  PFDisplacedTrackerVertexRef ni2_TO_DISP = elt2->displacedVertexRef(T_TO_DISP);
  PFDisplacedTrackerVertexRef ni1_FROM_DISP = elt1->displacedVertexRef(T_FROM_DISP);
  PFDisplacedTrackerVertexRef ni2_FROM_DISP = elt2->displacedVertexRef(T_FROM_DISP);
  
  if( ni1_TO_DISP.isNonnull() && ni2_FROM_DISP.isNonnull())
    if( ni1_TO_DISP == ni2_FROM_DISP ) { result = 1.0; return result; }

  if( ni1_FROM_DISP.isNonnull() && ni2_TO_DISP.isNonnull())
    if( ni1_FROM_DISP == ni2_TO_DISP ) { result = 1.0; return result; }

  if( ni1_FROM_DISP.isNonnull() && ni2_FROM_DISP.isNonnull())
    if( ni1_FROM_DISP == ni2_FROM_DISP ) { result = 1.0; return result; }
    
  
  if (  elt1->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)  &&
	     elt2->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)  ) {
    
    if(debug_ ) std::cout << " testLinkByVertex On Conversions " << std::endl;
    
    if ( elt1->convRef().isNonnull() && elt2->convRef().isNonnull() ) {
      if(debug_ ) std::cout << " PFBlockAlgo.cc testLinkByVertex  Cconversion Refs are non null  " << std::endl;      
      if ( elt1->convRef() ==  elt2->convRef() ) {
	result=1.0;
	if(debug_ ) std::cout << " testLinkByVertex  Cconversion Refs are equal  " << std::endl;      
	return result;
      }
    } 
    
  }
  
  if (  elt1->trackType(reco::PFBlockElement::T_FROM_V0)  &&
             elt2->trackType(reco::PFBlockElement::T_FROM_V0)  ) {
    if(debug_ ) std::cout << " testLinkByVertex On V0 " << std::endl;
    if ( elt1->V0Ref().isNonnull() && elt2->V0Ref().isNonnull() ) {
      if(debug_ ) std::cout << " PFBlockAlgo.cc testLinkByVertex  V0 Refs are non null  " << std::endl;
      if ( elt1->V0Ref() ==  elt2->V0Ref() ) {
	result=1.0;
	if(debug_ ) std::cout << " testLinkByVertex  V0 Refs are equal  " << std::endl;
	return result;
      }
    }
  }

  return result;
}

double 
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


  //track at calo's
  double tracketa = 999999.9;
  double trackphi = 999999.9;
  double track_X  = 999999.9;
  double track_Y  = 999999.9;
  double track_Z  = 999999.9;
  double dHEta = 0.;
  double dHPhi = 0.;

  // Quantities at vertex
  double trackPt = isBrem ? 999. : sqrt(atVertex.momentum().Vect().Perp2());
  // double trackEta = isBrem ? 999. : atVertex.momentum().Vect().Eta();


  switch (cluster.layer()) {
  case PFLayer::ECAL_BARREL: barrel = true;
  case PFLayer::ECAL_ENDCAP:
#ifdef PFLOW_DEBUG
    if( debug_ )
      cout << "Fetching Ecal Resolution Maps"
	   << endl;
#endif
    // did not reach ecal, cannot be associated with a cluster.
    if( ! atECAL.isValid() ) return -1.;   
    
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
    if( isBrem ) {  
      return  -1.;
    } else { 
      hcal=true;
      const reco::PFTrajectoryPoint& atHCAL 
	= track.extrapolatedPoint( reco::PFTrajectoryPoint::HCALEntrance );
      const reco::PFTrajectoryPoint& atHCALExit 
	= track.extrapolatedPoint( reco::PFTrajectoryPoint::HCALExit );
      // did not reach hcal, cannot be associated with a cluster.
      if( ! atHCAL.isValid() ) return -1.;   
      
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
    return -1.;
  default:
    return -1.;
  }


  // Check that, if the cluster is in the endcap, 
  // 0) the track indeed points to the endcap at vertex (DISABLED)
  // 1) the track extrapolation is in the endcap too !
  // 2) the track is in the same end-cap !
  // PJ - 10-May-09
  if ( !barrel ) { 
    // if ( fabs(trackEta) < 1.0 ) return -1; 
    if ( !hcal && fabs(track_Z) < 300. ) return -1.;
    if ( track_Z * clusterZ < 0. ) return -1.;
  }
  // Check that, if the cluster is in the barrel, 
  // 1) the track is in the barrel too !
  if ( barrel ) 
    if ( !hcal && fabs(track_Z) > 300. ) return -1.;

  // Finally check that, if the track points to the central barrel (|eta| < 1), 
  // it cannot be linked to a cluster in Endcaps (avoid low pt loopers)


  double dist = computeDist( clustereta, clusterphi, 
			     tracketa, trackphi);
  
#ifdef PFLOW_DEBUG
  if(debug_) cout<<"test link by rechit "<< dist <<" "<<endl;
  if(debug_){
    cout<<" clustereta "  << clustereta 
	<<" clusterphi "  << clusterphi 
	<<" tracketa " << tracketa
	<<" trackphi " << trackphi << endl;
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
    /*    
    //if ( distance > 40. || distance < -100. ) 
    double clusterr = std::sqrt(clusterX*clusterX+clusterY*clusterY);
    double trackr = std::sqrt(track_X*track_X+track_Y*track_Y);
    if ( distance > 40. ) 
    std::cout << "Distance = " << distance 
    << ", Barrel/Hcal/Brem ? " << barrel << " " << hcal << " " << isBrem << std::endl
    << " Cluster " << clusterr << " " << clusterZ << " " << clusterphi << " " << clustereta << std::endl
    << " Track   " << trackr << " " << track_Z << " " << trackphi << " " << tracketa << std::endl;
    if ( !barrel && fabs(trackEta) < 1.0 ) { 
      double clusterr = std::sqrt(clusterX*clusterX+clusterY*clusterY);
      double trackr = std::sqrt(track_X*track_X+track_Y*track_Y);
      std::cout << "TrackEta/Pt = " << trackEta << " " << trackPt << ", distance = " << distance << std::endl 
		<< ", Barrel/Hcal/Brem ? " << barrel << " " << hcal << " " << isBrem << std::endl
		<< " Cluster " << clusterr << " " << clusterZ << " " << clusterphi << " " << clustereta << std::endl
		<< " Track   " << trackr << " " << track_Z << " " << trackphi << " " << tracketa << " " << trackEta << " " << trackPt << std::endl;
    } 
    */
    return dist;
  } else {
    return -1.;
  }

}

double
PFBlockAlgo::testECALAndPSByRecHit( const PFCluster& clusterECAL, 
				    const PFCluster& clusterPS)  const {

  // Check that clusterECAL is in ECAL endcap and that clusterPS is a preshower cluster
  if ( clusterECAL.layer() != PFLayer::ECAL_ENDCAP ||
       ( clusterPS.layer() != PFLayer::PS1 && 
	 clusterPS.layer() != PFLayer::PS2 ) ) return -1.;

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
	if (zECAL*zPS <0.) return -1.;
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
    double dist = computeDist( xECAL/1000.,yECAL/1000.,
			       xPS/1000.  ,yPS/1000);    
    return dist;
  } else { 
    return -1.;
  }

}

double 
PFBlockAlgo::testHFEMAndHFHADByRecHit(const reco::PFCluster& clusterHFEM, 
				      const reco::PFCluster& clusterHFHAD) const {
  
  math::XYZPoint posxyzEM = clusterHFEM.position();
  math::XYZPoint posxyzHAD = clusterHFHAD.position();

  double dX = posxyzEM.X()-posxyzHAD.X();
  double dY = posxyzEM.Y()-posxyzHAD.Y();
  double sameZ = posxyzEM.Z()*posxyzHAD.Z();

  if(sameZ<0) return -1.;

  double dist2 = dX*dX + dY*dY; 

  if( dist2 < 0.1 ) {
    // less than one mm
    double dist = sqrt( dist2 );
    return dist;;
  }
  else 
    return -1.;

}

double
PFBlockAlgo::computeDist( double eta1, double phi1, 
			  double eta2, double phi2 ) const {
  
  double phicor = Utils::mpi_pi(phi1 - phi2);
  
  // double chi2 =  
  //  (eta1 - eta2)*(eta1 - eta2) / ( reta1*reta1+ reta2*reta2 ) +
  //  phicor*phicor / ( rphi1*rphi1+ rphi2*rphi2 );

  double dist = std::sqrt( (eta1 - eta2)*(eta1 - eta2) 
			  + phicor*phicor);

  return dist;

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

  // iteration 1,2,3,4,5 correspond to algo = 1/4,5,6,7,8,9
  unsigned int Algo = 0; 
  switch (trackref->algo()) {
  case TrackBase::ctf:
  case TrackBase::iter0:
  case TrackBase::iter1:
    Algo = 0;
    break;
  case TrackBase::iter2:
    Algo = 1;
    break;
  case TrackBase::iter3:
    Algo = 2;
    break;
  case TrackBase::iter4:
    Algo = 3;
    break;
  case TrackBase::iter5:
    Algo = 4;
    break;
  default:
    Algo = 5;
    break;
  }

  // Protection against 0 momentum tracks
  if ( P < 0.05 ) return false;

  // Temporary : Reject all tracking iteration beyond 5th step. 
  if ( Algo > 4 ) return false;
 
  if (debug_) cout << " PFBlockAlgo: PFrecTrack->Track Pt= "
		   << Pt << " DPt = " << DPt << endl;
  if ( ( DPtovPtCut_[Algo] > 0. && 
	 DPt/Pt > DPtovPtCut_[Algo]*sigmaHad ) || 
       NHit < NHitCut_[Algo] ) { 
    // (Algo >= 3 && LostHits != 0) ) {
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
PFBlockAlgo::checkDisplacedVertexLinks( reco::PFBlock& block ) const {
  // method which removes link between primary tracks and clusters
  // if at least one of the associated secondary tracks is closer 
  // to these same clusters
  /*
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
      double   distprim  = ie->first;
      unsigned iprim     = ie->second;
      // if this track a primary track (T_TO_DISP)
      // the new strategy gouzevitch: remove all the links from primary track
      if( els[iprim].trackType(PFBlockElement::T_TO_DISP) )  {
	    block.setLink( i1, iprim, -1, block.linkData(),
			   PFBlock::LINKTEST_RECHIT );
	    
	      // here the old startegy
	std::multimap<double, unsigned> secTracks; 
	// get associated secondary tracks
	block.associatedElements( iprim,  block.linkData(),
				  secTracks,
				  reco::PFBlockElement::TRACK,
				  reco::PFBlock::LINKTEST_ALL );
	for( IE ie2 = secTracks.begin(); ie2 != secTracks.end(); ++ie2) { 
	  unsigned isec = ie2->second;
	  double distsec = block.dist( i1, isec, block.linkData(),
				       PFBlock::LINKTEST_RECHIT );

	  // at present associatedElement return first the chi2 by chi2
	  // maybe in the futur return the min between chi2 and rechit! 
	  // if one secondary tracks has a chi2 < chi2prim 
	  // remove the link between the element and the primary
	  if( distsec < 0 ) continue;
	  else if( distsec < distprim ) { 
	    block.setLink( i1, iprim, -1, block.linkData(),
			   PFBlock::LINKTEST_RECHIT );
	    continue;
	  }
	} // loop on all associated secondary tracks
	    
      } // test if track is T_TO_DISP
                             
    } // loop on all associated tracks
  } // loop on all elements
  */
}


