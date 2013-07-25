// \class JetTracksAssociationXtrpCalo
// Associate jets with tracks by extrapolation to calo face
// $Id: JetTracksAssociationXtrpCalo.h,v 1.2 2011/02/16 18:25:33 stadie Exp $

#ifndef RecoJets_JetAssociationAlgorithms_JetTracksAssociationXtrpCalo_h
#define RecoJets_JetAssociationAlgorithms_JetTracksAssociationXtrpCalo_h


#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/Point3D.h"
#include <vector>
#include "DataFormats/JetReco/interface/TrackExtrapolation.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class MagneticField;
class Propagator;

class JetTracksAssociationXtrpCalo { 
 public:
  typedef reco::JetTracksAssociation::Container Association;
  typedef edm::RefToBase<reco::Jet> JetRef;
  typedef std::vector<JetRef> JetRefs;
  typedef std::vector<reco::TrackRef> TrackRefs;
  /// Constructor
  JetTracksAssociationXtrpCalo();
  
  /// Destructor
  ~JetTracksAssociationXtrpCalo();

  /// Associates tracks to jets
  void produce( Association*,
		JetRefs const &,
		std::vector<reco::TrackExtrapolation> const &,
		CaloGeometry const &,
		double dR );

  void associateInputTracksToJet( reco::TrackRefVector& associated,
				  const reco::Jet& fJet,
				  std::vector<reco::TrackExtrapolation> const & fExtrapolations,
				  double dR ) ;
  

  

 private:

  /// Unused
  virtual void associateTracksToJet( reco::TrackRefVector&,
				     const reco::Jet&,
				     const TrackRefs& ) {}
};

#endif // RecoJets_JetAssociationAlgorithms_JetTracksAssociationXtrpCalo_h
