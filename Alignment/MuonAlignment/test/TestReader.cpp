// -*- C++ -*-
//
//
// Description: Module to test the Alignment software
//
//


// system include files
#include <string>
#include <TTree.h>
#include <TRotMatrix.h>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorRcd.h"

#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Alignment/MuonAlignment/interface/MuonAlignment.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterBuilder.h" 

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

//
//
// class declaration
//

class TestMuonReader : public edm::EDAnalyzer {
public:
  explicit TestMuonReader( const edm::ParameterSet& );
  ~TestMuonReader();

  void recursiveGetMuComponents(std::vector<Alignable*> &composite, std::vector<Alignable*> &chambers, int kind);
  align::EulerAngles toPhiXYZ(const align::RotationType &);
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  TTree* theTree;
  TFile* theFile;
  float x,y,z,phi,theta,length,thick,width;
  TRotMatrix* rot;

};

//
// constructors and destructor
//
TestMuonReader::TestMuonReader( const edm::ParameterSet& iConfig ) :
  theTree(0), theFile(0),
  x(0.), y(0.), z(0.), phi(0.), theta(0.), length(0.), thick(0.), width(0.),
  rot(0)
{ 
}


TestMuonReader::~TestMuonReader()
{ 
}

void TestMuonReader::recursiveGetMuComponents(std::vector<Alignable*> &composites, std::vector<Alignable*> &chambers, int kind)
{
  for (std::vector<Alignable*>::const_iterator cit = composites.begin(); cit != composites.end(); cit++)
  {
    if ((*cit)->alignableObjectId() == kind)
    {
      chambers.push_back(*cit);
      continue;
    }
    else 
    {
      std::vector<Alignable*> components = (*cit)->components();
      recursiveGetMuComponents(components, chambers, kind);
    }
  }
}

align::EulerAngles TestMuonReader::toPhiXYZ(const align::RotationType& rot)
{
  align::EulerAngles angles(3);
  angles(1) = reco::deltaPhi(std::atan2(rot.yz(), rot.zz()),0.);
  angles(2) = reco::deltaPhi(std::asin(-rot.xz()),0.);
  angles(3) = reco::deltaPhi(std::atan2(rot.xy(), rot.xx()),0.);
  return angles;
}


void
TestMuonReader::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  using namespace std;

  // get chamber alignables from ideal geometry:
  edm::ESTransientHandle<DDCompactView> cpv;
  iSetup.get<IdealGeometryRecord>().get(cpv);
  edm::ESHandle<MuonDDDConstants> mdc;
  iSetup.get<MuonNumberingRecord>().get(mdc);
  DTGeometryBuilderFromDDD DTGeometryBuilder;
  CSCGeometryBuilderFromDDD CSCGeometryBuilder;
  boost::shared_ptr<DTGeometry> ideal_dtGeometry(new DTGeometry );
  DTGeometryBuilder.build(ideal_dtGeometry, &(*cpv), *mdc);
  boost::shared_ptr<CSCGeometry> ideal_cscGeometry(new CSCGeometry);
  CSCGeometryBuilder.build(ideal_cscGeometry, &(*cpv), *mdc);
  AlignableMuon ideal_alignableMuon(&(*ideal_dtGeometry), &(*ideal_cscGeometry));
  AlignableNavigator ideal_navigator(ideal_alignableMuon.components());

  // read in Alignables through MuonAlignment
  MuonAlignment real_align( iSetup );
  AlignableMuon* real_alignableMuon = real_align.getAlignableMuon();
  AlignableNavigator real_navigator(real_alignableMuon->components());

  //edm::LogInfo("MuonAlignment") << "Starting!";

  // Read in alignments constants from DBase
  edm::ESHandle<Alignments> dtAlignments;
  iSetup.get<DTAlignmentRcd>().get(dtAlignments);
  edm::ESHandle<Alignments> cscAlignments;
  iSetup.get<CSCAlignmentRcd>().get( cscAlignments );
  //edm::ESHandle<AlignmentErrors> dtAlignmentErrors;
  //iSetup.get<DTAlignmentErrorRcd>().get(dtAlignmentErrors);
  //edm::ESHandle<AlignmentErrors> cscAlignmentErrors;
  //iSetup.get<CSCAlignmentErrorRcd>().get( cscAlignmentErrors );

  // read in GlobalTrackingGeometry, which is supposed to have the alignments applied
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  // read muon GPR
  edm::ESHandle<Alignments> globalPositionRcd;
  iSetup.get<GlobalPositionRcd>().get(globalPositionRcd);
  const AlignTransform & globalCoordinates = align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon));
  //const AlignTransform::Translation &globalShift = globalCoordinates.translation();
  const AlignTransform::Rotation globalRotation = globalCoordinates.rotation(); // by value!
  const AlignTransform::Rotation inverseGlobalRotation = globalRotation.inverse();

  /*
  cout << "label id wheel station sector SL layer r_x r_y r_z r_phix r_phiy r_phiz c_x c_y c_z c_phix c_phiy c_phiz g_x g_y g_z g_phix g_phiy g_phiz"<<endl;
  for (vector<AlignTransform>::const_iterator it = dtAlignments->m_align.begin(); it != dtAlignments->m_align.end(); it++)
  {
    DTLayerId id((*it).rawId());

    //find this in ideal alignment:
    AlignableDetOrUnitPtr ideal_al = ideal_navigator.alignableFromDetId((*it).rawId());
    if (ideal_al.isNull()) { cout<<" no ideal alignable for "<<id<<endl; continue;}

    if (ideal_al->alignableObjectId() != align::AlignableDTSuperLayer) continue; // only SL

    //find this in real alignment:
    AlignableDetOrUnitPtr real_al = real_navigator.alignableFromDetId((*it).rawId());
    if (real_al.isNull()) { cout<<" no real alignable for "<<id<<endl; continue;}

    // find this in geometry
    DetId did((*it).rawId());
    const GeomDet* geomDet = theTrackingGeometry->idToDet(did);


    char nme[100];
    sprintf(nme,"mbb%+d_%d_%02d_%d",id.station(),id.station(),id.sector(),id.superlayer());


    // ideal alignment transformation
    align::PositionType ideal_position = ideal_al->globalPosition();
    align::RotationType ideal_rotation = ideal_al->globalRotation();
    //align::rectify(ideal_rotation);
    //cout << nme <<" "<< (*it).rawId()<<"  "<<ideal_position.basicVector()<<" "<<align::toAngles(ideal_rotation)<<endl;continue;

    // "real" alignment transformation
    align::PositionType real_position = real_al->globalPosition();
    align::RotationType real_rotation = real_al->globalRotation();
    //cout << nme <<" "<< (*it).rawId()<<"  "<<ideal_position.basicVector()<<" "<<align::toAngles(real_rotation)<<endl;continue;
    //WRT ideal:
    align::PositionType i_real_position = align::PositionType( ideal_rotation * (real_position.basicVector() - ideal_position.basicVector()) );
    align::RotationType i_real_rotation = real_rotation * ideal_rotation.transposed();
    //align::rectify(i_real_rotation);
    align::EulerAngles i_real_abg = align::toAngles(i_real_rotation);
    align::EulerAngles i_real_xyz = toPhiXYZ(i_real_rotation);

    // "real" alignment transformations that are read directly from DB
    // (first need to apply global correction)
    CLHEP::Hep3Vector pos = globalRotation * CLHEP::Hep3Vector( (*it).translation() ) + globalShift;
    CLHEP::HepRotation rot = CLHEP::HepRotation( (*it).rotation() )  * inverseGlobalRotation;
    align::PositionType position( pos.x(), pos.y(), pos.z());
    align::RotationType rotation( rot.xx(), rot.xy(), rot.xz(),
                                  rot.yx(), rot.yy(), rot.yz(),
                                  rot.zx(), rot.zy(), rot.zz() );
    // WRT ideal
    align::PositionType i_position = align::PositionType( ideal_rotation * (position.basicVector() - ideal_position.basicVector()) );
    align::RotationType i_rotation = rotation * ideal_rotation.transposed();
    //align::rectify(i_rotation);
    align::EulerAngles i_abg = align::toAngles(i_rotation);
    align::EulerAngles i_xyz = toPhiXYZ(i_rotation);

    // alignment transformation from geometry:
    const Surface::PositionType& gpos = geomDet->position();
    const Surface::RotationType& grot = geomDet->rotation();
    align::PositionType gposition( gpos.x(), gpos.y(), gpos.z());
    align::RotationType grotation( grot.xx(), grot.xy(), grot.xz(),
                                   grot.yx(), grot.yy(), grot.yz(),
                                   grot.zx(), grot.zy(), grot.zz() );
    // WRT ideal
    align::PositionType i_gposition = align::PositionType( ideal_rotation * (gposition.basicVector() - ideal_position.basicVector()) );
    align::RotationType i_grotation = grotation * ideal_rotation.transposed();
    //align::rectify(i_grotation);
    align::EulerAngles i_gabg = align::toAngles(i_grotation);
    align::EulerAngles i_gxyz = toPhiXYZ(i_grotation);

    cout<<setprecision(8)<<fixed;
    cout <<nme<<" "<<(*it).rawId()<<" "<<id.wheel()<<" "<<id.station()<<" "<<id.sector()<<" "<<id.superlayer()<<" "<<id.layer()
        <<"  "<<i_real_position.x()*10.<<" "<<i_real_position.y()*10.<<" "<<i_real_position.z()*10.
        <<"  "<<i_real_xyz[0]*1000.<<" "<<i_real_xyz[1]*1000.<<" "<<i_real_xyz[2]*1000.
        <<"  "<<i_position.x()*10.<<" "<<i_position.y()*10.<<" "<<i_position.z()*10.
        <<"  "<<i_xyz[0]*1000.<<" "<<i_xyz[1]*1000.<<" "<<i_xyz[2]*1000.
        <<"  "<<i_gposition.x()*10.<<" "<<i_gposition.y()*10.<<" "<<i_gposition.z()*10.
        <<"  "<<i_gxyz[0]*1000.<<" "<<i_gxyz[1]*1000.<<" "<<i_gxyz[2]*1000. << endl;
  }
  //cout << endl << "----------------------" << endl;
  */

  cout << "label id wheel station sector SL layer gl_x gl_y gl_z ch_x ch_y ch_z r_x r_y r_z r_ch_x r_ch_y r_ch_z g_x g_y g_z g_ch_x g_ch_y g_ch_z"<<endl;
  for (vector<AlignTransform>::const_iterator it = dtAlignments->m_align.begin(); it != dtAlignments->m_align.end(); it++)
  {
    DTLayerId id((*it).rawId());
    DTChamberId cid(id.wheel(), id.station(), id.sector());

    //find this in ideal alignment:
    AlignableDetOrUnitPtr ideal_al = ideal_navigator.alignableFromDetId((*it).rawId());
    if (ideal_al.isNull()) { cout<<" no ideal alignable for "<<id<<endl; continue;}

    if (ideal_al->alignableObjectId() != align::AlignableDetUnit) continue; // only layers #2
    if (id.layer() != 2) continue;

    AlignableDetOrUnitPtr ideal_al_ch = ideal_navigator.alignableFromDetId(cid.rawId());
    if (ideal_al_ch.isNull()) { cout<<" no ideal alignable for "<<cid<<endl; continue;}

    //find them in real alignment:
    AlignableDetOrUnitPtr real_al = real_navigator.alignableFromDetId((*it).rawId());
    if (real_al.isNull()) { cout<<" no real alignable for "<<id<<endl; continue;}
    AlignableDetOrUnitPtr real_al_ch = real_navigator.alignableFromDetId(cid.rawId());
    if (real_al_ch.isNull()) { cout<<" no real alignable for "<<cid<<endl; continue;}

    // find them in geometry
    DetId did((*it).rawId());
    const GeomDet* geom_det = theTrackingGeometry->idToDet(did);
    DetId cdid(cid.rawId());
    const GeomDet* geom_det_ch = theTrackingGeometry->idToDet(cdid);


    char nme[100];
    sprintf(nme,"mbb%+d_%d_%02d_%d",id.station(),id.station(),id.sector(),id.superlayer());

    // get the global position of the ideal local point (0,100,0):
    align::LocalPoint lp(0,100.,0);
    align::GlobalPoint gp = ideal_al->surface().toGlobal(lp);
    // local in chamber's frame
    align::LocalPoint lp_ch = ideal_al_ch->surface().toLocal(gp);

    // find local position of this gp in "real" alignment in layer and chamber frame of ref.
    align::LocalPoint real_lp = real_al->surface().toLocal(gp);
    align::LocalPoint real_lp_ch = real_al_ch->surface().toLocal(gp);

    // alignment transformation from geometry:
    LocalPoint geo_lp = geom_det->toLocal(gp);
    LocalPoint geo_lp_ch = geom_det_ch->toLocal(gp);

    cout<<setprecision(8)<<fixed;
    cout <<nme<<" "<<(*it).rawId()<<" "<<id.wheel()<<" "<<id.station()<<" "<<id.sector()<<" "<<id.superlayer()<<" "<<id.layer()
        <<"  "<<gp.x()*10.<<" "<<gp.y()*10.<<" "<<gp.z()*10.
        <<"  "<<lp_ch.x()*10.<<" "<<lp_ch.y()*10.<<" "<<lp_ch.z()*10.
        <<"  "<<real_lp.x()*10.<<" "<<real_lp.y()*10.<<" "<<real_lp.z()*10.
        <<"  "<<real_lp_ch.x()*10.<<" "<<real_lp_ch.y()*10.<<" "<<real_lp_ch.z()*10.
        <<"  "<<geo_lp.x()*10.<<" "<<geo_lp.y()*10.<<" "<<geo_lp.z()*10.
        <<"  "<<geo_lp_ch.x()*10.<<" "<<geo_lp_ch.y()*10.<<" "<<geo_lp_ch.z()*10.<< endl;
  }


  /*
  for ( vector<AlignTransformError>::const_iterator it = dtAlignmentErrors->m_alignError.begin();
		it != dtAlignmentErrors->m_alignError.end(); it++ )
	{
	  CLHEP::HepSymMatrix error = (*it).matrix();
	  cout << (*it).rawId() << " ";
	  for ( int i=0; i<error.num_row(); i++ )
		for ( int j=0; j<=i; j++ ) 
		  cout << " " << error[i][j];
	  cout << endl;
	}
  */

/*
  //vector<Alignable*>::const_iterator csc_ideal = ideal_endcaps.begin();
  cout<<setprecision(3)<<fixed;
  //cout<<" lens : "<<ideal_me_chambers.size()<<" "<<cscAlignments->m_align.size()<<endl;

  for ( vector<AlignTransform>::const_iterator it = cscAlignments->m_align.begin(); it != cscAlignments->m_align.end(); it++ )
  {
    CSCDetId id((*it).rawId());
    if (id.layer()>0) continue; // look at chambers only, skip layers

    if (id.station()==1 && id.ring()==4) continue; // not interested in duplicated ME1/4 

    char nme[100];
    sprintf(nme,"%d/%d/%02d",id.station(),id.ring(),id.chamber());
    string me = "ME+";
    if (id.endcap()==2) me = "ME-";
    me += nme;
    
    // find this chamber in ideal geometry
    const Alignable* ideal=0;
    for (vector<Alignable*>::const_iterator cideal = ideal_me_chambers.begin(); cideal != ideal_me_chambers.end(); cideal++)
      if ((*cideal)->geomDetId().rawId() == (*it).rawId()) { ideal = *cideal; break; }
    if (ideal==0) {
      cout<<" no ideal chamber for "<<id<<endl;
      continue;
    }

    //if (ideal->geomDetId().rawId() != (*it).rawId()) cout<<" badid : "<<(*csc_ideal)->geomDetId().rawId()<<" "<<(*it).rawId()<<endl;

    align::PositionType position((*it).translation().x(), (*it).translation().y(), (*it).translation().z());

    CLHEP::HepRotation rot( (*it).rotation() );
    align::RotationType rotation( rot.xx(), rot.xy(), rot.xz(),
                                  rot.yx(), rot.yy(), rot.yz(),
                                  rot.zx(), rot.zy(), rot.zz() );
    //align::EulerAngles abg = align::toAngles(rotation);

    align::PositionType idealPosition = ideal->globalPosition();
    align::RotationType idealRotation = ideal->globalRotation();
    //align::EulerAngles abg = align::toAngles(idealRotation);
    //cout << me <<" "<< (*it).rawId()<<"  "<<idealPosition.basicVector()<<" "<<abg<<endl;continue;

    // compute transformations relative to ideal
    align::PositionType rposition = align::PositionType( idealRotation * (position.basicVector() - idealPosition.basicVector()) );
    align::RotationType rrotation = rotation * idealRotation.transposed();
    align::EulerAngles rabg = align::toAngles(rrotation);
    
    //align::EulerAngles rxyz = toPhiXYZ(rrotation);
    //if (fabs(rabg[0]-rxyz[0])>0.00001 || fabs(rabg[1]-rxyz[1])>0.00001 ||fabs(rabg[1]-rxyz[1])>0.00001)
    //  cout << me <<" large angle diff = "<<fabs(rabg[0]-rxyz[0])*1000.<<" "<<fabs(rabg[1]-rxyz[1])*1000.<<" "<<fabs(rabg[2]-rxyz[2])*1000.<<" = "
    //        << 1000.*rabg[0] <<" "<< 1000.*rabg[1] <<" "<< 1000.*rabg[2] <<" - " << 1000.*rxyz[0] <<" "<< 1000.*rxyz[1] <<" "<< 1000.*rxyz[2]<<endl;

    cout << me <<" "<< (*it).rawId()
      //<< "  " << (*it).translation().x() << " " << (*it).translation().y() << " " << (*it).translation().z()
      //<< "  " << abg[0] << " " << abg[1] << " " << abg[2] 
      //<< "  " << rotation.xx() << " " << rotation.xy() << " " << rotation.xz()
      //<< " " << rotation.yx() << " " << rotation.yy() << " " << rotation.yz()
      //<< " " << rotation.zx() << " " << rotation.zy() << " " << rotation.zz()

      <<"  "<< 10.*rposition.x() <<" "<< 10.*rposition.y() <<" "<< 10.*rposition.z()
      <<"  "<< 1000.*rabg[0] <<" "<< 1000.*rabg[1] <<" "<< 1000.*rabg[2]
      << endl;
  }
*/

/*
  cout << endl << "----------------------" << endl;

  for ( vector<AlignTransformError>::const_iterator it = cscAlignmentErrors->m_alignError.begin();
		it != cscAlignmentErrors->m_alignError.end(); it++ )
	{
	  CLHEP::HepSymMatrix error = (*it).matrix();
	  cout << (*it).rawId() << " ";
	  for ( int i=0; i<error.num_row(); i++ )
		for ( int j=0; j<=i; j++ ) 
		  cout << " " << error[i][j];
	  cout << endl;
	}


  edm::LogInfo("MuonAlignment") << "Done!";
*/
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestMuonReader);
