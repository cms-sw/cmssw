// -*- C++ -*-
//
// Package:    TrackerOfflineValidation
// Class:      TrackerOfflineValidation
// 
/**\class TrackerOfflineValidation TrackerOfflineValidation.cc Alignment/Validator/src/TrackerOfflineValidation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Erik Butz
//         Created:  Tue Dec 11 14:03:05 CET 2007
// $Id: TrackerOfflineValidation.cc,v 1.1 2008/02/27 17:34:40 ebutz Exp $
//
//


// system include files
#include <memory>
#include <map>
#include <sstream>
#include <math.h>

// ROOT includes
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TFile.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/UtilAlgos/interface/TFileDirectory.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

//
// class decleration
//

//typedef std::vector<DetId>          DetIdContainer;


class TrackerOfflineValidation : public edm::EDAnalyzer {
public:
  explicit TrackerOfflineValidation(const edm::ParameterSet&);
  ~TrackerOfflineValidation();
 
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  const edm::ParameterSet parset_;
  edm::ESHandle<TrackerGeometry> tkgeom;
  edm::ParameterSet Parameters;
  
  std::vector<TH1*> v_trackhistos;
  std::vector<TProfile*> v_trackprofiles;
  std::vector<TH2*> v_track2Dhistos;
  
  std::vector<TH1*> v_subdetres;
  std::vector<TH1*> v_subdetnormres;

  std::vector<TProfile*> v_tobrodphi;
  std::vector<TProfile*> v_tobrodphi_rms;
  std::vector<TProfile*> v_toblayerphi;
  std::vector<TProfile*> v_toblayerphi_rms;
  
  std::vector<TProfile*> v_tibstringphi;
  std::vector<TProfile*> v_tibstringphi_rms;
  
  std::vector<TProfile*> v_tiblayerphi;
  std::vector<TProfile*> v_tiblayerphi_rms;

  std::vector<TProfile*> v_tiddiscphi;
  std::vector<TProfile*> v_tiddiscphi_rms;

  std::vector<TProfile*> v_teclayerphi;
  std::vector<TProfile*> v_teclayerphi_rms;


  std::map<align::StructureType, std::vector<TProfile*> > m_hierarchy_profiles;

  std::map<int,TH1*> m_pxbresiduals;
  std::map<int,TH1*> m_pxeresiduals;
  std::map<int,TH1*> m_tibresiduals;
  std::map<int,TH1*> m_tidresiduals;
  std::map<int,TH1*> m_tobresiduals;
  std::map<int,TH1*> m_tecresiduals;

  std::map<int,TH1*> m_pxbnormresiduals;
  std::map<int,TH1*> m_pxenormresiduals;
  std::map<int,TH1*> m_tibnormresiduals;
  std::map<int,TH1*> m_tidnormresiduals;
  std::map<int,TH1*> m_tobnormresiduals;
  std::map<int,TH1*> m_tecnormresiduals;

  AlignableTracker *alitracker;
  
  void bookDirHists(TFileDirectory &tfd, const Alignable& ali, const AlignableObjectId &aliobjid);
  void bookHists(TFileDirectory &tfd, const Alignable& ali, align::StructureType type, int i, const AlignableObjectId &aliobjid);
 
  void collateSummaryHists( TFileDirectory &tfd, const Alignable& ali, int i, const AlignableObjectId &aliobjid, std::vector<TProfile*> &v_levelprofiles);
  
  TProfile* bookSummaryHists(TFileDirectory &tfd, const Alignable& ali, align::StructureType type, int i, const AlignableObjectId &aliobjid); 
 
  // From MillePedeAlignmentMonitor: Get Index for Arbitary vector<class> by name
  template <class OBJECT_TYPE>  
  int GetIndex(const std::vector<OBJECT_TYPE*> &vec, const TString &name);
  

  
  
  // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//
typedef std::vector<DetId>                 DetIdContainer;
//
// static data member definitions
//
template <class OBJECT_TYPE>  
int TrackerOfflineValidation::GetIndex(const std::vector<OBJECT_TYPE*> &vec, const TString &name)
{
  int result = 0;
  for (typename std::vector<OBJECT_TYPE*>::const_iterator iter = vec.begin(), iterEnd = vec.end();
       iter != iterEnd; ++iter, ++result) {
    if (*iter && (*iter)->GetName() == name) return result;
  }
  edm::LogError("Alignment") << "@SUB=MillePedeMonitor::GetIndex" << " could not find " << name;
  return -1;
}
//
// constructors and destructor
//
TrackerOfflineValidation::TrackerOfflineValidation(const edm::ParameterSet& iConfig)
  :   parset_(iConfig)
{
   //now do what ever initialization is needed
 

}


TrackerOfflineValidation::~TrackerOfflineValidation()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//


// ------------ method called once each job just before starting event loop  ------------
void 
TrackerOfflineValidation::beginJob(const edm::EventSetup& es)
{
  es.get<TrackerDigiGeometryRecord>().get( tkgeom );
  edm::Service<TFileService> fs;    
  AlignableObjectId aliobjid;

  // construct alignable tracker to get access to alignable hierarchy 
  // -> no backward compatibility below 1_8_X

  alitracker = new AlignableTracker(&(*tkgeom));
  
  
  v_trackhistos.push_back(fs->make<TH1F>("h_tracketa","Track #eta;#eta_{Track};Number of Tracks",90,-3.,3.));	       
  v_trackhistos.push_back(fs->make<TH1F>("h_curvature","Curvature #kappa;#kappa_{Track};Number of Tracks",100,-.05,.05));
  v_trackhistos.push_back(fs->make<TH1F>("h_curvature_pos","Curvature |#kappa| Positive Tracks;|#kappa_{pos Track}|;Number of Tracks",100,.0,.05));
  v_trackhistos.push_back(fs->make<TH1F>("h_curvature_neg","Curvature |#kappa| Negative Tracks;|#kappa_{neg Track}|;Number of Tracks",100,.0,.05));
  v_trackhistos.push_back(fs->make<TH1F>("h_diff_curvature","Curvature |#kappa| Tracks Difference;|#kappa_{Track}|;# Pos Tracks - # Neg Tracks",100,.0,.05));
  v_trackhistos.push_back(fs->make<TH1F>("h_chi2","#chi^{2};#chi^{2}_{Track};Number of Tracks",500,-0.01,500.));	       
  v_trackhistos.push_back(fs->make<TH1F>("h_normchi2","#chi^{2}/ndof;#chi^{2}/ndof;Number of Tracks",100,-0.01,10.));     
  v_trackhistos.push_back(fs->make<TH1F>("h_pt","p_{T};p_{T}^{track};Number of Tracks",100,0.,2500));                     

  v_trackprofiles.push_back(fs->make<TProfile>("h_d0_vs_phi","Transverse Impact Parameter vs. #phi;#phi_{Track};#LT d_{0} #GT",100,-3.15,3.15));  
  v_trackprofiles.push_back(fs->make<TProfile>("h_dz_vs_phi","Longitudinal Impact Parameter vs. #phi;#phi_{Track};#LT d_{z} #GT",100,-3.15,3.15));
  v_trackprofiles.push_back(fs->make<TProfile>("h_d0_vs_eta","Transverse Impact Parameter vs. #eta;#eta_{Track};#LT d_{0} #GT",100,-3.15,3.15));  
  v_trackprofiles.push_back(fs->make<TProfile>("h_dz_vs_eta","Longitudinal Impact Parameter vs. #eta;#eta_{Track};#LT d_{z} #GT",100,-3.15,3.15));
  v_trackprofiles.push_back(fs->make<TProfile>("h_chi2_vs_phi","#chi^{2} vs. #phi;#phi_{Track};#LT #chi^{2} #GT",100,-3.15,3.15));		  
  v_trackprofiles.push_back(fs->make<TProfile>("h_normchi2_vs_phi","#chi^{2}/ndof vs. #phi;#phi_{Track};#LT #chi^{2}/ndof #GT",100,-3.15,3.15));  
  v_trackprofiles.push_back(fs->make<TProfile>("h_chi2_vs_eta","#chi^{2} vs. #eta;#eta_{Track};#LT #chi^{2} #GT",100,-3.15,3.15));		  
  v_trackprofiles.push_back(fs->make<TProfile>("h_normchi2_vs_eta","#chi^{2}/ndof vs. #eta;#eta_{Track};#LT #chi^{2}/ndof #GT",100,-3.15,3.15));  

  v_track2Dhistos.push_back(fs->make<TH2F>("h2_d0_vs_phi","Transverse Impact Parameter vs. #phi;#phi_{Track};d_{0}",100, -3.15, 3.15, 100,-1.,1.) );  
  v_track2Dhistos.push_back(fs->make<TH2F>("h2_dz_vs_phi","Longitudinal Impact Parameter vs. #phi;#phi_{Track};d_{z}",100, -3.15, 3.15, 100,-100.,100.));
  v_track2Dhistos.push_back(fs->make<TH2F>("h2_d0_vs_eta","Transverse Impact Parameter vs. #eta;#eta_{Track};d_{0}",100, -3.15, 3.15, 100,-1.,1.));  
  v_track2Dhistos.push_back(fs->make<TH2F>("h2_dz_vs_eta","Longitudinal Impact Parameter vs. #eta;#eta_{Track};d_{z}",100, -3.15, 3.15, 100,-100.,100.));
  v_track2Dhistos.push_back(fs->make<TH2F>("h2_chi2_vs_phi","#chi^{2} vs. #phi;#phi_{Track};#chi^{2}",100, -3.15, 3.15, 500, 0., 500.));		  
  v_track2Dhistos.push_back(fs->make<TH2F>("h2_normchi2_vs_phi","#chi^{2}/ndof vs. #phi;#phi_{Track};#chi^{2}/ndof",100, -3.15, 3.15, 100, 0., 10.));  
  v_track2Dhistos.push_back(fs->make<TH2F>("h2_chi2_vs_eta","#chi^{2} vs. #eta;#eta_{Track};#chi^{2}",100, -3.15, 3.15, 500, 0., 500.));		  
  v_track2Dhistos.push_back(fs->make<TH2F>("h2_normchi2_vs_eta","#chi^{2}/ndof vs. #eta;#eta_{Track};#chi^{2}/ndof",100,-3.15,3.15, 100, 0., 10.));  

  Parameters = parset_.getParameter<edm::ParameterSet>("TH1ResModules");
  int32_t i_residuals_Nbins =  Parameters.getParameter<int32_t>("Nbinx");
  double d_residual_xmin = Parameters.getParameter<double>("xmin");
  double d_residual_xmax = Parameters.getParameter<double>("xmax");

  Parameters = parset_.getParameter<edm::ParameterSet>("TH1NormResModules");
  int32_t i_normres_Nbins =  Parameters.getParameter<int32_t>("Nbinx");
  double d_normres_xmin = Parameters.getParameter<double>("xmin");
  double d_normres_xmax = Parameters.getParameter<double>("xmax");
  
  v_subdetres.push_back(fs->make<TH1F>("h_residuals_pxb","Residuals in PXB;x_{pred} - x_{rec} [cm]", i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetres.push_back(fs->make<TH1F>("h_residuals_pxe","Residuals in PXE;x_{pred} - x_{rec} [cm]", i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetres.push_back(fs->make<TH1F>("h_residuals_tib","Residuals in TIB;x_{pred} - x_{rec} [cm]", i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetres.push_back(fs->make<TH1F>("h_residuals_tid","Residuals in TID;x_{pred} - x_{rec} [cm]", i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetres.push_back(fs->make<TH1F>("h_residuals_tob","Residuals in TOB;x_{pred} - x_{rec} [cm]", i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetres.push_back(fs->make<TH1F>("h_residuals_tec","Residuals in TEC;x_{pred} - x_{rec} [cm]", i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;

  v_subdetnormres.push_back(fs->make<TH1F>("h_normresiduals_pxb","Normalized Residuals in PXB;(x_{pred} - x_{rec})/#sqrt{V}", 
					   i_normres_Nbins,d_normres_xmin,d_normres_xmax));   
  v_subdetnormres.push_back(fs->make<TH1F>("h_normresiduals_pxe","Normalized Residuals in PXE;(x_{pred} - x_{rec})/#sqrt{V}", 
					   i_normres_Nbins,d_normres_xmin,d_normres_xmax));   
  v_subdetnormres.push_back(fs->make<TH1F>("h_normresiduals_tib","Normalized Residuals in TIB;(x_{pred} - x_{rec})/#sqrt{V}", 
					   i_normres_Nbins,d_normres_xmin,d_normres_xmax));   
  v_subdetnormres.push_back(fs->make<TH1F>("h_normresiduals_tid","Normalized Residuals in TID;(x_{pred} - x_{rec})/#sqrt{V}", 
					   i_normres_Nbins,d_normres_xmin,d_normres_xmax));   
  v_subdetnormres.push_back(fs->make<TH1F>("h_normresiduals_tob","Normalized Residuals in TOB;(x_{pred} - x_{rec})/#sqrt{V}", 
					   i_normres_Nbins,d_normres_xmin,d_normres_xmax));   
  v_subdetnormres.push_back(fs->make<TH1F>("h_normresiduals_tec","Normalized Residuals in TEC;(x_{pred} - x_{rec})/#sqrt{V}", 
					   i_normres_Nbins,d_normres_xmin,d_normres_xmax));

  
  edm::LogInfo("TrackerOfflineValidation") << "There are " << (*tkgeom).detIds().size() << " detUnits in the Geometry record" << std::endl;

  // recursively book histogramms on lowest level
  bookDirHists(static_cast<TFileDirectory&>(*fs), *alitracker, aliobjid);
  
 
}


void
TrackerOfflineValidation::bookDirHists( TFileDirectory &tfd, const Alignable& ali, const AlignableObjectId &aliobjid)
{
  std::vector<Alignable*> alivec(ali.components());
  for(int i=0, iEnd = ali.components().size();i < iEnd; ++i) {
    std::string structurename  = aliobjid.typeToName((alivec)[i]->alignableObjectId());
    edm::LogVerbatim("TrackerOfflineValidation") << "StructureName = " << structurename;
    std::stringstream dirname;
    
    // add no suffix counter to Strip and Pixel
    // just aesthetics
    if(structurename != "Strip" && structurename != "Pixel") {
      dirname << structurename << "_" << i;
    } else {
      dirname << structurename;
    }
    if (structurename.find("Endcap",0) != std::string::npos )   {
      TFileDirectory f = tfd.mkdir((dirname.str()).c_str());
      bookHists(f, *(alivec)[i], ali.alignableObjectId() , i, aliobjid);
      bookDirHists( f, *(alivec)[i], aliobjid);

    } else if((structurename != "Det" && structurename != "DetUnit" ) || alivec[i]->components().size() > 1) {
      
      // next line by Gero, try to include splitted rechit case
      //} else if((structurename != "Det" && structurename != "DetUnit" )) {
      TFileDirectory f = tfd.mkdir((dirname.str()).c_str());
      bookHists(tfd, *(alivec)[i], ali.alignableObjectId() , i, aliobjid);
      bookDirHists( f, *(alivec)[i], aliobjid);
   
      //} else if(structurename == "Det" && alivec[i]->components().size() > 1) {
      //bookDirHists( tfd, *(alivec)[i], aliobjid);
    } else {
      //std::cout << structurename << ' ' << alivec[i]->components().size() << ' ' << alivec[i]->id() << std::endl;

      bookHists(tfd, *(alivec)[i], ali.alignableObjectId() , i, aliobjid);
    }
    
  }


  
}





void 
TrackerOfflineValidation::bookHists(TFileDirectory &tfd, const Alignable& ali, align::StructureType type, int i, const AlignableObjectId &aliobjid)
{

  // binnings for module level histogramms are steerable via cfg file
  Parameters = parset_.getParameter<edm::ParameterSet>("TH1ResModules");
  int32_t i_residuals_Nbins =  Parameters.getParameter<int32_t>("Nbinx");
  double d_residual_xmin = Parameters.getParameter<double>("xmin");
  double d_residual_xmax = Parameters.getParameter<double>("xmax");
  Parameters = parset_.getParameter<edm::ParameterSet>("TH1NormResModules");
  int32_t i_normres_Nbins =  Parameters.getParameter<int32_t>("Nbinx");
  double d_normres_xmin = Parameters.getParameter<double>("xmin");
  double d_normres_xmax = Parameters.getParameter<double>("xmax");

  const DetId id = ali.id();
  uint subdetid = id.subdetId();
  TIBDetId tibid(id.rawId());
  TOBDetId tobid(id.rawId());
 
  // book residual and normalized residual histogramms for 
  // the lowest level in each subdetector
  switch (type) 
    {
      // PXB Histogramms
    case (align::TPBLadder) :
      m_pxbresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_residual_subdet%d_module%d", id.subdetId(), id.rawId()),
						  Form("Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
						  i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
      m_pxbnormresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_normresidual_subdet%d_module%d", id.subdetId(), id.rawId()),
						  Form("Normalized Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
						  i_normres_Nbins,d_normres_xmin,d_normres_xmax);
      break;
      // PXE Histogramms
    case (align::TPEPanel) :
      m_pxeresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_residual_subdet%d_module%d", id.subdetId(), id.rawId()),
						  Form("Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
						  i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
      m_pxenormresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_normresidual_subdet%d_module%d", id.subdetId(), id.rawId()),
						  Form("Normalized Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
						  i_normres_Nbins,d_normres_xmin,d_normres_xmax);
      break;
      // TIB Histogramms
    case (align::TIBString) : 
      m_tibresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_residual_subdet%d_side%d_module%d", 
						       id.subdetId() ,tibid.string()[0], id.rawId()),
						  Form("Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
						  i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
      m_tibnormresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_normresidual_subdet%d_side%d_module%d", 
							   id.subdetId() ,tibid.string()[0], id.rawId()),
						      Form("Normalized Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
						      i_normres_Nbins,d_normres_xmin,d_normres_xmax);
      break;

      // TID Histogramms
    case (align::TIDSide) :
      m_tidresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_residual_subdet%d_module%d", id.subdetId(), id.rawId()),
						  Form("Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()),
						  i_residuals_Nbins,d_residual_xmin,d_residual_xmax);  
      m_tidnormresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_normresidual_subdet%d_module%d", id.subdetId(), id.rawId()),
						  Form("Normalized Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()),
						  i_normres_Nbins,d_normres_xmin,d_normres_xmax);  
      break;

      // TOB Histogramms
    case (align::TOBRod) :
      m_tobresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_residual_subdet%d_side%d_layer%d__rod%d_module%d",id.subdetId() ,tobid.rod()[0],
						       tobid.layer(), tobid.rod()[1], id.rawId()),
						  Form("Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
      m_tobnormresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_normresidual_subdet%d_side%d_layer%d__rod%d_module%d",id.subdetId() ,tobid.rod()[0],
							   tobid.layer(), tobid.rod()[1], id.rawId()),
						      Form("Normalized Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()),
						      i_normres_Nbins,d_normres_xmin,d_normres_xmax);
      break;
      // TEC Histogramms
    case (align::TECRing) :
      m_tecresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_residual_subdet%d_module%d", id.subdetId(), id.rawId()),
						  Form("Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
						  i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
      m_tecnormresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_normresidual_subdet%d_module%d", id.subdetId(), id.rawId()),
						  Form("Normalized Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
						  i_normres_Nbins,d_normres_xmin,d_normres_xmax);
      break;
      
    case (align::AlignableDet) :
      if(subdetid == PixelSubdetector::PixelBarrel ) {
	m_pxbresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_residual_subdet%d_module%d", id.subdetId(), id.rawId()),
						    Form("Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
						    i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
	m_pxbnormresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_normresidual_subdet%d_module%d", id.subdetId(), id.rawId()),
							Form("Normalized Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
							i_normres_Nbins,d_normres_xmin,d_normres_xmax);
      } else if (subdetid == PixelSubdetector::PixelEndcap) {      
	m_pxeresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_residual_subdet%d_module%d", id.subdetId(), id.rawId()),
						    Form("Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
						    i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
	m_pxenormresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_normresidual_subdet%d_module%d", id.subdetId(), id.rawId()),
							Form("Normalized Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
							i_normres_Nbins,d_normres_xmin,d_normres_xmax);
      } else if(subdetid  == StripSubdetector::TIB) {
	  m_tibresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_residual_subdet%d_side%d_module%d", 
							   id.subdetId() ,tibid.string()[0], id.rawId()),
						      Form("Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
						      i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
	  m_tibnormresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_normresidual_subdet%d_side%d_module%d", 
							       id.subdetId() ,tibid.string()[0], id.rawId()),
							  Form("Normalized Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
							  i_normres_Nbins,d_normres_xmin,d_normres_xmax);
      } else if(subdetid  == StripSubdetector::TID) {
	m_tidresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_residual_subdet%d_module%d", id.subdetId(), id.rawId()),
						    Form("Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()),
						    i_residuals_Nbins,d_residual_xmin,d_residual_xmax);  
	m_tidnormresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_normresidual_subdet%d_module%d", id.subdetId(), id.rawId()),
							Form("Normalized Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()),
							i_normres_Nbins,d_normres_xmin,d_normres_xmax);  
      } else if(subdetid  == StripSubdetector::TOB) {
	m_tobresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_residual_subdet%d_side%d_layer%d__rod%d_module%d",id.subdetId() ,tobid.rod()[0],
							 tobid.layer(), tobid.rod()[1], id.rawId()),
						    Form("Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
	m_tobnormresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_normresidual_subdet%d_side%d_layer%d__rod%d_module%d",id.subdetId() ,tobid.rod()[0],
							     tobid.layer(), tobid.rod()[1], id.rawId()),
							Form("Normalized Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()),
							i_normres_Nbins,d_normres_xmin,d_normres_xmax);
      } else if(subdetid  == StripSubdetector::TEC) {
	m_tecresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_residual_subdet%d_module%d", id.subdetId(), id.rawId()),
						    Form("Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
						    i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
	m_tecnormresiduals[id.rawId()] = tfd.make<TH1F>(Form("h_normresidual_subdet%d_module%d", id.subdetId(), id.rawId()),
							Form("Normalized Residual for module %d;x_{pred} - x_{rec} [cm]",id.rawId()), 
							i_normres_Nbins,d_normres_xmin,d_normres_xmax);
      } else {
	edm::LogWarning("Residuals") << "No such subdetector: " << subdetid;      
      }
      
      break;


    default :
      edm::LogVerbatim("TrackerOfflineValidation") << "nothing to be done for Structure " << aliobjid.typeToName(type)<< std::endl;
      //std::cout << "nothing to be done for Structure " << aliobjid.typeToName(type)<< std::endl;
      break;
      
    }
  
  
  
}


// ------------ method called to for each event  ------------
void
TrackerOfflineValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  using namespace edm;
  TrackerValidationVariables avalidator_(iSetup,parset_);
  
    
  std::vector<TrackerValidationVariables::AVTrackStruct> v_trackstruct;
  avalidator_.fillTrackQuantities(iEvent,v_trackstruct);
  std::vector<TrackerValidationVariables::AVHitStruct> v_hitstruct;
  avalidator_.fillHitQuantities(iEvent,v_hitstruct);
  
  
  for (std::vector<TrackerValidationVariables::AVTrackStruct>::const_iterator it = v_trackstruct.begin(),
	 itEnd = v_trackstruct.end(); it != itEnd; ++it) {
    
    // Fill 1D track histos
    static const int etaindex = this->GetIndex(v_trackhistos,"h_tracketa");
    v_trackhistos[etaindex]->Fill(it->eta);
    static const int kappaindex = this->GetIndex(v_trackhistos,"h_curvature");
    v_trackhistos[kappaindex]->Fill(it->kappa);
    static const int kappaposindex = this->GetIndex(v_trackhistos,"h_curvature_pos");
    if(it->charge > 0)
      v_trackhistos[kappaposindex]->Fill(fabs(it->kappa));
    static const int kappanegindex = this->GetIndex(v_trackhistos,"h_curvature_neg");
    if(it->charge < 0)
      v_trackhistos[kappanegindex]->Fill(fabs(it->kappa));
    static const int normchi2index = this->GetIndex(v_trackhistos,"h_normchi2");
    v_trackhistos[normchi2index]->Fill(it->normchi2);
    static const int chi2index = this->GetIndex(v_trackhistos,"h_chi2");
    v_trackhistos[chi2index]->Fill(it->chi2);
    static const int ptindex = this->GetIndex(v_trackhistos,"h_pt");
    v_trackhistos[ptindex]->Fill(it->pt);
    
    // Fill track profiles
    static const int d0phiindex = this->GetIndex(v_trackprofiles,"h_d0_vs_phi");
    v_trackprofiles[d0phiindex]->Fill(it->phi,it->d0);
    static const int dzphiindex = this->GetIndex(v_trackprofiles,"h_dz_vs_phi");
    v_trackprofiles[dzphiindex]->Fill(it->phi,it->dz);
    static const int d0etaindex = this->GetIndex(v_trackprofiles,"h_d0_vs_eta");
    v_trackprofiles[d0etaindex]->Fill(it->eta,it->d0);
    static const int dzetaindex = this->GetIndex(v_trackprofiles,"h_dz_vs_eta");
    v_trackprofiles[dzetaindex]->Fill(it->eta,it->dz);
    static const int chiphiindex = this->GetIndex(v_trackprofiles,"h_chi2_vs_phi");
    v_trackprofiles[chiphiindex]->Fill(it->phi,it->chi2);
    static const int normchiphiindex = this->GetIndex(v_trackprofiles,"h_normchi2_vs_phi");
    v_trackprofiles[normchiphiindex]->Fill(it->phi,it->normchi2);
    static const int chietaindex = this->GetIndex(v_trackprofiles,"h_chi2_vs_eta");
    v_trackprofiles[chietaindex]->Fill(it->eta,it->chi2);
    static const int normchietaindex = this->GetIndex(v_trackprofiles,"h_normchi2_vs_eta");
    v_trackprofiles[normchietaindex]->Fill(it->eta,it->normchi2);
    
    // Fill 2D track histos
    static const int d0phiindex_2d = this->GetIndex(v_track2Dhistos,"h2_d0_vs_phi");
    v_track2Dhistos[d0phiindex_2d]->Fill(it->phi,it->d0);
    static const int dzphiindex_2d = this->GetIndex(v_track2Dhistos,"h2_dz_vs_phi");
    v_track2Dhistos[dzphiindex_2d]->Fill(it->phi,it->dz);
    static const int d0etaindex_2d = this->GetIndex(v_track2Dhistos,"h2_d0_vs_eta");
    v_track2Dhistos[d0etaindex_2d]->Fill(it->eta,it->d0);
    static const int dzetaindex_2d = this->GetIndex(v_track2Dhistos,"h2_dz_vs_eta");
    v_track2Dhistos[dzetaindex_2d]->Fill(it->eta,it->dz);
    static const int chiphiindex_2d = this->GetIndex(v_track2Dhistos,"h2_chi2_vs_phi");
    v_track2Dhistos[chiphiindex_2d]->Fill(it->phi,it->chi2);
    static const int normchiphiindex_2d = this->GetIndex(v_track2Dhistos,"h2_normchi2_vs_phi");
    v_track2Dhistos[normchiphiindex_2d]->Fill(it->phi,it->normchi2);
    static const int chietaindex_2d = this->GetIndex(v_track2Dhistos,"h2_chi2_vs_eta");
    v_track2Dhistos[chietaindex_2d]->Fill(it->eta,it->chi2);
    static const int normchietaindex_2d = this->GetIndex(v_track2Dhistos,"h2_normchi2_vs_eta");
    v_track2Dhistos[normchietaindex_2d]->Fill(it->eta,it->normchi2);
     
  } // finish loop over track quantities


  // hit quantities: residuals, normalized residuals
  for (std::vector<TrackerValidationVariables::AVHitStruct>::const_iterator it = v_hitstruct.begin(),
  	 itEnd = v_hitstruct.end(); it != itEnd; ++it) {
    uint subdetid = DetId::DetId(it->rawDetId).subdetId();
    
    if(subdetid == PixelSubdetector::PixelBarrel ) {
      m_pxbresiduals[it->rawDetId]->Fill(it->resX);
      if(it->resErrX != 0)
  	m_pxbnormresiduals[it->rawDetId]->Fill(it->resX/it->resErrX);
    } else if (subdetid == PixelSubdetector::PixelEndcap) {      
      m_pxeresiduals[it->rawDetId]->Fill(it->resX);
      if(it->resErrX != 0)
  	m_pxenormresiduals[it->rawDetId]->Fill(it->resX/it->resErrX);
    } else if(subdetid  == StripSubdetector::TIB) {
      m_tibresiduals[it->rawDetId]->Fill(it->resX);
      if(it->resErrX != 0)
  	m_tibnormresiduals[it->rawDetId]->Fill(it->resX/it->resErrX);
    } else if(subdetid  == StripSubdetector::TID) {
      m_tidresiduals[it->rawDetId]->Fill(it->resX);
      if(it->resErrX != 0)
  	m_tidnormresiduals[it->rawDetId]->Fill(it->resX/it->resErrX);
    } else if(subdetid  == StripSubdetector::TOB) {
      m_tobresiduals[it->rawDetId]->Fill(it->resX);
      if(it->resErrX != 0)
  	m_tobnormresiduals[it->rawDetId]->Fill(it->resX/it->resErrX);
    } else if(subdetid  == StripSubdetector::TEC) {
      m_tecresiduals[it->rawDetId]->Fill(it->resX);
      if(it->resErrX != 0)
  	m_tecnormresiduals[it->rawDetId]->Fill(it->resX/it->resErrX);
    } else {
      edm::LogWarning("Residuals") << "No such subdetector: " << subdetid;      
    }
  }
  
}



// ------------ method called once each job just after ending the event loop  ------------
void 
TrackerOfflineValidation::endJob() {
  
  edm::Service<TFileService> fs;   
  AlignableObjectId aliobjid;


  static const int kappadiffindex = this->GetIndex(v_trackhistos,"h_diff_curvature");
  v_trackhistos[kappadiffindex]->Add(v_trackhistos[this->GetIndex(v_trackhistos,"h_curvature_neg")],v_trackhistos[this->GetIndex(v_trackhistos,"h_curvature_pos")],-1,1);
  // Collate Information for Subdetectors
  // So far done by default, should be steerable
  for(std::map<int, TH1*>::const_iterator itPxb = m_pxbresiduals.begin(), itEnd = m_pxbresiduals.end(); itPxb != itEnd;++itPxb ) {
    v_subdetres[PixelSubdetector::PixelBarrel - 1]->Add(itPxb->second); 
    v_subdetnormres[PixelSubdetector::PixelBarrel - 1]->Add(static_cast<TH1F*>(m_pxbnormresiduals[itPxb->first]));
  }
  for(std::map<int, TH1*>::const_iterator itPxe = m_pxeresiduals.begin(), itEnd = m_pxeresiduals.end(); itPxe != itEnd;++itPxe ) {
    v_subdetres[PixelSubdetector::PixelEndcap - 1]->Add(itPxe->second);
    v_subdetnormres[PixelSubdetector::PixelEndcap - 1]->Add(static_cast<TH1F*>(m_pxenormresiduals[itPxe->first]));
  }
  for(std::map<int, TH1*>::const_iterator itTib = m_tibresiduals.begin(), itEnd = m_tibresiduals.end(); itTib != itEnd;++itTib ) {
    TIBDetId tibid(itTib->first);
    v_subdetres[StripSubdetector::TIB - 1]->Add(itTib->second);
    v_subdetnormres[StripSubdetector::TIB -1]->Add(static_cast<TH1F*>(m_tibnormresiduals[itTib->first]));
  }
  for(std::map<int, TH1*>::const_iterator itTid = m_tidresiduals.begin(), itEnd = m_tidresiduals.end(); itTid != itEnd;++itTid ) {
    TIDDetId tidid(itTid->first);
    v_subdetres[StripSubdetector::TID - 1]->Add(itTid->second);
    v_subdetnormres[StripSubdetector::TID - 1]->Add(static_cast<TH1F*>(m_tidnormresiduals[itTid->first]));
  }
  for(std::map<int, TH1*>::const_iterator itTob = m_tobresiduals.begin(), itEnd = m_tobresiduals.end(); itTob != itEnd;++itTob ) {
    TOBDetId tobid(itTob->first);  
    v_subdetres[StripSubdetector::TOB - 1]->Add(itTob->second);
    v_subdetnormres[StripSubdetector::TOB - 1]->Add(static_cast<TH1F*>(m_tobnormresiduals[itTob->first]));
  }
  for(std::map<int, TH1*>::const_iterator itTec = m_tecresiduals.begin(), itEnd = m_tecresiduals.end(); itTec != itEnd;++itTec ) {
    v_subdetres[StripSubdetector::TEC - 1]->Add(itTec->second);
    v_subdetnormres[StripSubdetector::TEC - 1]->Add(static_cast<TH1F*>(m_tecnormresiduals[itTec->first]));
  }
  
  // create summary histogramms recursively
  std::vector<TProfile*> v_trackerprofiles;
  collateSummaryHists((*fs),(*alitracker), 0, aliobjid, v_trackerprofiles);

  
}


void
TrackerOfflineValidation::collateSummaryHists( TFileDirectory &tfd, const Alignable& ali, int i, const AlignableObjectId &aliobjid, std::vector<TProfile*> &v_levelprofiles)
{
 
  std::vector<Alignable*> alivec(ali.components());
   
  if(aliobjid.typeToName((alivec)[0]->alignableObjectId()) == "Det" || aliobjid.typeToName((alivec)[0]->alignableObjectId()) == "DetUnit") {
     return;
  }

  for(int iComp=0, iCompEnd = ali.components().size();iComp < iCompEnd; ++iComp) {

    std::vector<TProfile*> v_profiles;        
    std::string structurename  = aliobjid.typeToName((alivec)[iComp]->alignableObjectId());
 
    edm::LogVerbatim("TrackerOfflineValidation") << "StructureName = " << structurename;
    std::stringstream dirname;
    
    // add no suffix counter to strip and pixel
    // just aesthetics
    if(structurename != "Strip" && structurename != "Pixel") {
      dirname << structurename << "_" << iComp;
    } else {
      dirname << structurename;
    }
    
    if((structurename != "Det" && structurename != "DetUnit" )) {
      TFileDirectory f = tfd.mkdir((dirname.str()).c_str());
      collateSummaryHists( f, *(alivec)[iComp], i, aliobjid, v_profiles);
      
      v_levelprofiles.push_back(bookSummaryHists(tfd, *(alivec[iComp]), ali.alignableObjectId(), iComp, aliobjid));
      
      for(uint n = 0;n<v_profiles.size();++n) 
	v_levelprofiles[iComp]->Fill(n+1,v_profiles[n]->GetMean(2));
    } else {
      // nothing to be done for det or detunits
      continue;
    }

  }

}

TProfile*
TrackerOfflineValidation::bookSummaryHists(TFileDirectory &tfd, const Alignable& ali, align::StructureType type, int i, const AlignableObjectId &aliobjid)
{
  
  
  int subsize = ali.components().size();
  align::StructureType alitype = ali.alignableObjectId();
  align::StructureType subtype = ali.components()[0]->alignableObjectId();
  TProfile * p_this = tfd.make<TProfile>(Form("h_%s_%d_phi",aliobjid.typeToName(alitype).c_str(),i), 
					 Form("Mean for %s in %s %d;%s;#LT #Delta x #GT",aliobjid.typeToName(alitype).c_str(),
					      aliobjid.typeToName(type).c_str(),i,aliobjid.typeToName(subtype).c_str()),
					 subsize,0.5,subsize+0.5)  ;
  
  if(aliobjid.typeToName(ali.components()[0]->alignableObjectId()) == "Det") {
    for(int k=0;k<subsize;++k) {
      DetId detid = ali.components()[k]->id();
      uint subdetid = detid.subdetId();
      if(subdetid == PixelSubdetector::PixelBarrel) {
	p_this->Fill(k+1,m_pxbresiduals[detid.rawId()]->GetMean());
      } else if(subdetid == PixelSubdetector::PixelEndcap) {
	p_this->Fill(k+1,m_pxeresiduals[detid.rawId()]->GetMean());
      } else if(subdetid == StripSubdetector::TIB) {
	p_this->Fill(k+1,m_tibresiduals[detid.rawId()]->GetMean());
      } else if(subdetid == StripSubdetector::TID) {
	p_this->Fill(k+1,m_tidresiduals[detid.rawId()]->GetMean());
      } else if(subdetid == StripSubdetector::TOB) {
	p_this->Fill(k+1,m_tobresiduals[detid.rawId()]->GetMean());
      } else if(subdetid == StripSubdetector::TEC) {
	p_this->Fill(k+1,m_tecresiduals[detid.rawId()]->GetMean());
      } else {
	edm::LogError("TrackerOfflineValidation") << "No valid Tracker Subdetector: " << subdetid;
      } 
      
    }
  }
  return p_this;
  
  
}




//define this as a plug-in
DEFINE_FWK_MODULE(TrackerOfflineValidation);
