#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <iostream>
#include <fstream>
#include "TFile.h"
#include "TH1F.h"
#include "TGraph.h"
#include "TTree.h"


#include "SurveyInputCSCfromPins.h"

#define SQR(x) ((x)*(x))

SurveyInputCSCfromPins::SurveyInputCSCfromPins(const edm::ParameterSet& cfg)
  : m_pinPositions(cfg.getParameter<std::string>("pinPositions"))
  , m_rootFile(cfg.getParameter<std::string>("rootFile"))
  , m_verbose(cfg.getParameter<bool>("verbose"))
{}


void SurveyInputCSCfromPins::beginJob(const edm::EventSetup& iSetup)
{
	edm::LogInfo("SurveyInputCSCfromPins") << "***************ENTERING BEGIN JOB******************" << "  \n";
	
	std::ifstream in;
  	in.open(m_pinPositions.c_str());
  
	Double_t x1, y1, z1, x2, y2, z2, a, b, tot=0.0, maxErr=0.0, h, s1, dx, dy, dz, PhX, PhZ, T, cosPhX, sinPhX, cosPhZ, sinPhZ;
   
	int ID1, ID2, ID3, ID4, ID5, i=1, ii=0;
	
	TFile *file1 = new TFile(m_rootFile.c_str(),"recreate");
	TTree *tree1 = new TTree("tree1","alignment pins");
      
   	if (m_verbose) {
   
   		tree1->Branch("displacement_x_pin1_cm", &x1, "x1/D");
   		tree1->Branch("displacement_y_pin1_cm", &y1, "y1/D");
	   	tree1->Branch("displacement_z_pin1_cm", &z1, "z1/D");
	   	tree1->Branch("displacement_x_pin2_cm", &x2, "x2/D");
   		tree1->Branch("displacement_y_pin2_cm", &y2, "y2/D");
   		tree1->Branch("displacement_z_pin2_cm", &z2, "z2/D");     
	   	tree1->Branch("error_vector_length_cm",&h, "h/D"); 
   		tree1->Branch("stretch_diff_cm",&s1, "s1/D");
	   	tree1->Branch("stretch_factor",&T, "T/D");
   		tree1->Branch("chamber_displacement_x_cm",&dx, "dx/D");
	   	tree1->Branch("chamber_displacement_y_cm",&dy, "dy/D");
   		tree1->Branch("chamber_displacement_z_cm",&dz, "dz/D");
	   	tree1->Branch("chamber_rotation_x_rad",&PhX, "PhX/D");
   		tree1->Branch("chamber_rotation_z_rad",&PhZ, "PhZ/D");
  	}
  
	edm::ESHandle<DTGeometry> dtGeometry;
	edm::ESHandle<CSCGeometry> cscGeometry;
 	iSetup.get<MuonGeometryRecord>().get( dtGeometry );     
 	iSetup.get<MuonGeometryRecord>().get( cscGeometry );
 
 	AlignableMuon* theAlignableMuon = new AlignableMuon( &(*dtGeometry) , &(*cscGeometry) );
 	AlignableNavigator* theAlignableNavigator = new AlignableNavigator( theAlignableMuon );
 
 
 
 	std::vector<Alignable*> theEndcaps = theAlignableMuon->CSCEndcaps();
 
	for (std::vector<Alignable*>::const_iterator aliiter = theEndcaps.begin();  aliiter != theEndcaps.end();  ++aliiter) {
     
 		addComponent(*aliiter);
    	}
    
	
	while (in.good())
  	{
    
    		in >> ID1 >> ID2 >> ID3 >> ID4 >> ID5 >> x1 >> y1 >> z1 >> x2 >> y2 >> z2 >> a >> b;
		 
		x1=x1/10.0;
		y1=y1/10.0;
		z1=z1/10.0;
		x2=x2/10.0;
		y2=y2/10.0;
		z2=z2/10.0;
   
 		CSCDetId layerID(ID1, ID2, ID3, ID4, 1);
		
		// We cannot use chamber ID (when ID5=0), because AlignableNavigator gives the error (aliDet and aliDetUnit are undefined for chambers)
		
		
		Alignable* theAlignable1 = theAlignableNavigator->alignableFromDetId( layerID ); 
 		Alignable* chamberAli=theAlignable1->mother();

  		LocalVector LC1 = chamberAli->surface().toLocal(GlobalVector(x1,y1,z1));
  		LocalVector LC2 = chamberAli->surface().toLocal(GlobalVector(x2,y2,z2));
  
		LocalPoint LP1(LC1.x(), LC1.y() + a, LC1.z() + b);
  		LocalPoint LP2(LC2.x(), LC2.y() - a, LC2.z() + b);
  
  		LocalPoint P((LP1.x() - LP2.x())/(2.*a), (LP1.y() - LP2.y())/(2.*a), (LP1.z() - LP2.z())/(2.*a));
  		LocalPoint Pp((LP1.x() + LP2.x())/(2.), (LP1.y() + LP2.y())/(2.), (LP1.z() + LP2.z())/(2.));
    
  		T=P.mag();
	
		sinPhX=P.z()/T;
		cosPhX=sqrt(1-SQR(sinPhX));
		cosPhZ=P.y()/(T*cosPhX);
		sinPhZ=-P.x()/(T*cosPhZ);
	 	
		PhX=atan2(sinPhX,cosPhX);
	
		PhZ=atan2(sinPhZ,cosPhZ);
	
		dx=Pp.x()-sinPhZ*sinPhX*b;
		dy=Pp.y()+cosPhZ*sinPhX*b;
		dz=Pp.z()-cosPhX*b;
	

 		GlobalPoint PG1 = chamberAli->surface().toGlobal(LP1);
 		GlobalPoint PG2 = chamberAli->surface().toGlobal(LP2);
 
	
 		LocalVector lvector( dx, dy, dz);
 		GlobalVector gvector = ( chamberAli->surface()).toGlobal( lvector );
 		chamberAli->move( gvector );
  
  		chamberAli->rotateAroundLocalX( PhX );
  		chamberAli->rotateAroundLocalZ( PhZ );
 
  		align::ErrorMatrix error = ROOT::Math::SMatrixIdentity();
  		chamberAli->setSurvey( new SurveyDet(chamberAli->surface(), error*(0.03)) );
    
    		if (m_verbose) {
    			
			edm::LogInfo("SurveyInputCSCfromPins") << " survey information = " << chamberAli->survey() << "  \n";
			
  			LocalPoint LP1n = chamberAli->surface().toLocal(PG1);
	
			LocalPoint hiP(LP1n.x(), LP1n.y() - a*T, LP1n.z() - b);

			h=hiP.mag();
			s1=LP1n.y() - a;

			if (h>maxErr) { maxErr=h; 
			
				ii=i;
	 		}
	
			edm::LogInfo("SurveyInputCSCfromPins") << "  \n" 
			 << "i " << i++ << " " << ID1 << " " << ID2 << " " << ID3 << " " << ID4 << " " << ID5 << " error  " << h  << " \n"	
			 <<" x1 " << x1 << " y1 " << y1 << " z1 " << z1 << " x2 " << x2 << " y2 " << y2 << " z2 " << z2  << " \n"
			 << " error " << h <<  " S1 " << s1 << " \n"
			 <<" dx " << dx << " dy " << dy << " dz " << dz << " PhX " << PhX << " PhZ " << PhZ  << " \n";

			tot+=h;
	
			tree1->Fill();
			}
   	
   	} 
 
	in.close();

	if (m_verbose) {
	
   		file1->Write();
		edm::LogInfo("SurveyInputCSCfromPins") << " Total error  " << tot << "  Max Error " << maxErr << " N " << ii << "  \n";
   	}

	file1->Close();
   
	for (std::vector<Alignable*>::const_iterator aliiter = theEndcaps.begin();  aliiter != theEndcaps.end();  ++aliiter) {
     
 		fillAllRecords(*aliiter);
    	} 

	edm::LogInfo("SurveyInputCSCfromPins") << "*************END BEGIN JOB***************" << "  \n";
}


void SurveyInputCSCfromPins::fillAllRecords(Alignable *ali) {
   	if (ali->survey() == 0) {
	
      		align::ErrorMatrix error = ROOT::Math::SMatrixIdentity();
      		ali->setSurvey(new SurveyDet(ali->surface(), error*(1e-8)));
   	}

   	std::vector<Alignable*> components = ali->components();
   	for (std::vector<Alignable*>::const_iterator iter = components.begin();  iter != components.end();  ++iter) {
	
      		fillAllRecords(*iter);
   	}
}


// Plug in to framework
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SurveyInputCSCfromPins);
