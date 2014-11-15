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


#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignTransformErrorExtended.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"

#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterBuilder.h" 

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

//
//
// class declaration
//

class TestMuonReader : public edm::EDAnalyzer {
public:
  explicit TestMuonReader( const edm::ParameterSet& );
  ~TestMuonReader();

  void recursiveGetMuChambers(std::vector<Alignable*> &composite, std::vector<Alignable*> &chambers, int kind);
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

void TestMuonReader::recursiveGetMuChambers(std::vector<Alignable*> &composites, std::vector<Alignable*> &chambers, int kind)
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
      recursiveGetMuChambers(components, chambers, kind);
    }
  }
}

align::EulerAngles TestMuonReader::toPhiXYZ(const align::RotationType& rot)
{
  align::EulerAngles angles(3);
  angles(1) = std::atan2(rot.yz(), rot.zz());
  angles(2) = std::asin(-rot.xz());
  angles(3) = std::atan2(rot.xy(), rot.xx());
  return angles;
}


void
TestMuonReader::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  // first, get chamber alignables from ideal geometry:

  edm::ESTransientHandle<DDCompactView> cpv;
  iSetup.get<IdealGeometryRecord>().get(cpv);

  edm::ESHandle<MuonDDDConstants> mdc;
  iSetup.get<MuonNumberingRecord>().get(mdc);

  DTGeometryBuilderFromDDD DTGeometryBuilder;
  CSCGeometryBuilderFromDDD CSCGeometryBuilder;

  boost::shared_ptr<DTGeometry> dtGeometry(new DTGeometry );
  DTGeometryBuilder.build(dtGeometry, &(*cpv), *mdc);
  boost::shared_ptr<CSCGeometry> cscGeometry(new CSCGeometry);
  CSCGeometryBuilder.build(cscGeometry, &(*cpv), *mdc);

  AlignableMuon ideal_alignableMuon(&(*dtGeometry), &(*cscGeometry));

  std::vector<Alignable*> ideal_barrels = ideal_alignableMuon.DTBarrel();
  std::vector<Alignable*> ideal_endcaps = ideal_alignableMuon.CSCEndcaps();

  std::vector<Alignable*> ideal_mb_chambers, ideal_me_chambers;
  recursiveGetMuChambers(ideal_barrels, ideal_mb_chambers, align::AlignableDTChamber);
  recursiveGetMuChambers(ideal_endcaps, ideal_me_chambers, align::AlignableCSCChamber);
  //std::cout<<" #ideals dts="<<ideal_mb_chambers.size()<<" cscs="<<ideal_me_chambers.size()<<std::endl;


/*   
  edm::LogInfo("MuonAlignment") << "Starting!";

  // Retrieve DT alignment[Error]s from DBase
  edm::ESHandle<Alignments> dtAlignments;
  iSetup.get<DTAlignmentRcd>().get( dtAlignments );
  edm::ESHandle<AlignmentErrorsExtended> dtAlignmentErrorsExtended;
  iSetup.get<DTAlignmentErrorExtendedRcd>().get( dtAlignmentErrorsExtended );

  for ( std::vector<AlignTransform>::const_iterator it = dtAlignments->m_align.begin();
		it != dtAlignments->m_align.end(); it++ )
	{
	  CLHEP::HepRotation rot( (*it).rotation() );
	  align::RotationType rotation( rot.xx(), rot.xy(), rot.xz(),
					rot.yx(), rot.yy(), rot.yz(),
					rot.zx(), rot.zy(), rot.zz() );

	  std::cout << (*it).rawId()
				<< "  " << (*it).translation().x()
				<< " " << (*it).translation().y()
				<< " " << (*it).translation().z()
				<< "  " << rotation.xx() << " " << rotation.xy() << " " << rotation.xz()
				<< " " << rotation.yx() << " " << rotation.yy() << " " << rotation.yz()
				<< " " << rotation.zx() << " " << rotation.zy() << " " << rotation.zz()
				<< std::endl;

	}
  std::cout << std::endl << "----------------------" << std::endl;

  for ( std::vector<AlignTransformErrorExtended>::const_iterator it = dtAlignmentErrorsExtended->m_alignError.begin();
		it != dtAlignmentErrorsExtended->m_alignError.end(); it++ )
	{
	  CLHEP::HepSymMatrix error = (*it).matrix();
	  std::cout << (*it).rawId() << " ";
	  for ( int i=0; i<error.num_row(); i++ )
		for ( int j=0; j<=i; j++ ) 
		  std::cout << " " << error[i][j];
	  std::cout << std::endl;
	}
*/


  // Retrieve CSC alignment[Error]s from DBase
  edm::ESHandle<Alignments> cscAlignments;
  iSetup.get<CSCAlignmentRcd>().get( cscAlignments );
  //edm::ESHandle<AlignmentErrorsExtended> cscAlignmentErrorsExtended;
  //iSetup.get<CSCAlignmentErrorExtendedRcd>().get( cscAlignmentErrorsExtended );

  //std::vector<Alignable*>::const_iterator csc_ideal = ideal_endcaps.begin();
  std::cout<<std::setprecision(3)<<std::fixed;
  //std::cout<<" lens : "<<ideal_me_chambers.size()<<" "<<cscAlignments->m_align.size()<<std::endl;

  for ( std::vector<AlignTransform>::const_iterator it = cscAlignments->m_align.begin(); it != cscAlignments->m_align.end(); it++ )
  {
    CSCDetId id((*it).rawId());
    if (id.layer()>0) continue; // look at chambers only, skip layers

    if (id.station()==1 && id.ring()==4) continue; // not interested in duplicated ME1/4 

    char nme[100];
    sprintf(nme,"%d/%d/%02d",id.station(),id.ring(),id.chamber());
    std::string me = "ME+";
    if (id.endcap()==2) me = "ME-";
    me += nme;
    
    // find this chamber in ideal geometry
    const Alignable* ideal=0;
    for (std::vector<Alignable*>::const_iterator cideal = ideal_me_chambers.begin(); cideal != ideal_me_chambers.end(); cideal++)
      if ((*cideal)->geomDetId().rawId() == (*it).rawId()) { ideal = *cideal; break; }
    if (ideal==0) {
      std::cout<<" no ideal chamber for "<<id<<std::endl;
      continue;
    }

    //if (ideal->geomDetId().rawId() != (*it).rawId()) std::cout<<" badid : "<<(*csc_ideal)->geomDetId().rawId()<<" "<<(*it).rawId()<<std::endl;

    align::PositionType position((*it).translation().x(), (*it).translation().y(), (*it).translation().z());

    CLHEP::HepRotation rot( (*it).rotation() );
    align::RotationType rotation( rot.xx(), rot.xy(), rot.xz(),
                                  rot.yx(), rot.yy(), rot.yz(),
                                  rot.zx(), rot.zy(), rot.zz() );
    //align::EulerAngles abg = align::toAngles(rotation);

    align::PositionType idealPosition = ideal->globalPosition();
    align::RotationType idealRotation = ideal->globalRotation();
    //align::EulerAngles abg = align::toAngles(idealRotation);
    //std::cout << me <<" "<< (*it).rawId()<<"  "<<idealPosition.basicVector()<<" "<<abg<<std::endl;continue;

    // compute transformations relative to ideal
    align::PositionType rposition = align::PositionType( idealRotation * (position.basicVector() - idealPosition.basicVector()) );
    align::RotationType rrotation = rotation * idealRotation.transposed();
    align::EulerAngles rabg = align::toAngles(rrotation);
    
    //align::EulerAngles rxyz = toPhiXYZ(rrotation);
    //if (fabs(rabg[0]-rxyz[0])>0.00001 || fabs(rabg[1]-rxyz[1])>0.00001 ||fabs(rabg[1]-rxyz[1])>0.00001)
    //  std::cout << me <<" large angle diff = "<<fabs(rabg[0]-rxyz[0])*1000.<<" "<<fabs(rabg[1]-rxyz[1])*1000.<<" "<<fabs(rabg[2]-rxyz[2])*1000.<<" = "
    //        << 1000.*rabg[0] <<" "<< 1000.*rabg[1] <<" "<< 1000.*rabg[2] <<" - " << 1000.*rxyz[0] <<" "<< 1000.*rxyz[1] <<" "<< 1000.*rxyz[2]<<std::endl;

    std::cout << me <<" "<< (*it).rawId()
      //<< "  " << (*it).translation().x() << " " << (*it).translation().y() << " " << (*it).translation().z()
      //<< "  " << abg[0] << " " << abg[1] << " " << abg[2] 
      //<< "  " << rotation.xx() << " " << rotation.xy() << " " << rotation.xz()
      //<< " " << rotation.yx() << " " << rotation.yy() << " " << rotation.yz()
      //<< " " << rotation.zx() << " " << rotation.zy() << " " << rotation.zz()

      <<"  "<< 10.*rposition.x() <<" "<< 10.*rposition.y() <<" "<< 10.*rposition.z()
      <<"  "<< 1000.*rabg[0] <<" "<< 1000.*rabg[1] <<" "<< 1000.*rabg[2]
      << std::endl;
  }
/*
  std::cout << std::endl << "----------------------" << std::endl;

  for ( std::vector<AlignTransformErrorExtended>::const_iterator it = cscAlignmentErrorsExtended->m_alignError.begin();
		it != cscAlignmentErrorsExtended->m_alignError.end(); it++ )
	{
	  CLHEP::HepSymMatrix error = (*it).matrix();
	  std::cout << (*it).rawId() << " ";
	  for ( int i=0; i<error.num_row(); i++ )
		for ( int j=0; j<=i; j++ ) 
		  std::cout << " " << error[i][j];
	  std::cout << std::endl;
	}


  edm::LogInfo("MuonAlignment") << "Done!";
*/
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestMuonReader);
