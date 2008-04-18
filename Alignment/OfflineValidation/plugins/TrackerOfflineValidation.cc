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
// $Id: TrackerOfflineValidation.cc,v 1.2 2008/03/26 16:20:26 ebutz Exp $
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
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

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
  
  struct ModuleHistos{
    ModuleHistos() :  ResHisto(), NormResHisto() {} 
    TH1* ResHisto;
    TH1* NormResHisto;
    TH1* ResXprimeHisto;
    TH1* NormResXprimeHisto;
  };
  
  ModuleHistos& GetHistStructFromMap(const DetId& detid);


  std::vector<TH1*> v_trackhistos;
  std::vector<TProfile*> v_trackprofiles;
  std::vector<TH2*> v_track2Dhistos;
  
  std::vector<TH1*> v_subdetres;
  std::vector<TH1*> v_subdetnormres;
  std::vector<TH1*> v_subdetXprimeRes;

  //__gnu_cxx::hash_map<int,TrackerOfflineValidation::ModuleHistos> m_modulehistos;
  
  std::map<int,TrackerOfflineValidation::ModuleHistos> m_pxbResiduals;
  std::map<int,TrackerOfflineValidation::ModuleHistos> m_pxeResiduals;
  std::map<int,TrackerOfflineValidation::ModuleHistos> m_tibResiduals;
  std::map<int,TrackerOfflineValidation::ModuleHistos> m_tidResiduals;
  std::map<int,TrackerOfflineValidation::ModuleHistos> m_tobResiduals;
  std::map<int,TrackerOfflineValidation::ModuleHistos> m_tecResiduals;

  AlignableTracker *alitracker;
  void bookGlobalHists(TFileDirectory &tfd);
  void bookDirHists(TFileDirectory &tfd, const Alignable& ali, const AlignableObjectId &aliobjid);
  void bookHists(TFileDirectory &tfd, const Alignable& ali, align::StructureType type, int i, const AlignableObjectId &aliobjid);
 
  void collateSummaryHists( TFileDirectory &tfd, const Alignable& ali, int i, const AlignableObjectId &aliobjid, std::vector<std::pair<TH1*,TH1*> > &v_levelprofiles);
  
  std::pair<TH1*,TH1*> bookSummaryHists(TFileDirectory &tfd, const Alignable& ali, align::StructureType type, int i, const AlignableObjectId &aliobjid); 
 
 
  float Fwhm(const TH1* hist);

  // From MillePedeAlignmentMonitor: Get Index for Arbitary vector<class> by name
  template <class OBJECT_TYPE>  
  int GetIndex(const std::vector<OBJECT_TYPE*> &vec, const TString &name);
  
 
  
  // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//
typedef std::vector<DetId>  DetIdContainer;
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
  edm::LogError("Alignment") << "@SUB=TrackerOfflineValidation::GetIndex" << " could not find " << name;
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
  

  //
  // Book Histogramms for global track quantities
  TFileDirectory trackglobal = fs->mkdir("GlobalTrackVariables");
  
  this->bookGlobalHists(trackglobal);
  
  Parameters = parset_.getParameter<edm::ParameterSet>("TH1ResModules");
  int32_t i_residuals_Nbins =  Parameters.getParameter<int32_t>("Nbinx");
  double d_residual_xmin = Parameters.getParameter<double>("xmin");
  double d_residual_xmax = Parameters.getParameter<double>("xmax");

  Parameters = parset_.getParameter<edm::ParameterSet>("TH1NormResModules");
  int32_t i_normres_Nbins =  Parameters.getParameter<int32_t>("Nbinx");
  double d_normres_xmin = Parameters.getParameter<double>("xmin");
  double d_normres_xmax = Parameters.getParameter<double>("xmax");
  

  v_subdetres.push_back(fs->make<TH1F>("h_Residuals_pxb","Residuals in PXB;x_{pred} - x_{rec} [cm]", 
				       i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetres.push_back(fs->make<TH1F>("h_Residuals_pxe","Residuals in PXE;x_{pred} - x_{rec} [cm]", 
				       i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetres.push_back(fs->make<TH1F>("h_Residuals_tib","Residuals in TIB;x_{pred} - x_{rec} [cm]", 
				       i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetres.push_back(fs->make<TH1F>("h_Residuals_tid","Residuals in TID;x_{pred} - x_{rec} [cm]", 
				       i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetres.push_back(fs->make<TH1F>("h_Residuals_tob","Residuals in TOB;x_{pred} - x_{rec} [cm]", 
				       i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetres.push_back(fs->make<TH1F>("h_Residuals_tec","Residuals in TEC;x_{pred} - x_{rec} [cm]", 
				       i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;

  v_subdetXprimeRes.push_back(fs->make<TH1F>("h_XprimeResiduals_pxb","Residuals in PXB;x_{pred} - x_{rec} [cm]", 
				       i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetXprimeRes.push_back(fs->make<TH1F>("h_XprimeResiduals_pxe","Residuals in PXE;x_{pred} - x_{rec} [cm]", 
				       i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetXprimeRes.push_back(fs->make<TH1F>("h_XprimeResiduals_tib","Residuals in TIB;x_{pred} - x_{rec} [cm]", 
				       i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetXprimeRes.push_back(fs->make<TH1F>("h_XprimeResiduals_tid","Residuals in TID;x_{pred} - x_{rec} [cm]", 
				       i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetXprimeRes.push_back(fs->make<TH1F>("h_XprimeResiduals_tob","Residuals in TOB;x_{pred} - x_{rec} [cm]", 
				       i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;
  v_subdetXprimeRes.push_back(fs->make<TH1F>("h_XprimeResiduals_tec","Residuals in TEC;x_{pred} - x_{rec} [cm]", 
				       i_residuals_Nbins,d_residual_xmin,d_residual_xmax)) ;


  v_subdetnormres.push_back(fs->make<TH1F>("h_normResiduals_pxb","Normalized Residuals in PXB;(x_{pred} - x_{rec})/#sqrt{V}", 
					   i_normres_Nbins,d_normres_xmin,d_normres_xmax));   
  v_subdetnormres.push_back(fs->make<TH1F>("h_normResiduals_pxe","Normalized Residuals in PXE;(x_{pred} - x_{rec})/#sqrt{V}", 
					   i_normres_Nbins,d_normres_xmin,d_normres_xmax));   
  v_subdetnormres.push_back(fs->make<TH1F>("h_normResiduals_tib","Normalized Residuals in TIB;(x_{pred} - x_{rec})/#sqrt{V}", 
					   i_normres_Nbins,d_normres_xmin,d_normres_xmax));   
  v_subdetnormres.push_back(fs->make<TH1F>("h_normResiduals_tid","Normalized Residuals in TID;(x_{pred} - x_{rec})/#sqrt{V}", 
					   i_normres_Nbins,d_normres_xmin,d_normres_xmax));   
  v_subdetnormres.push_back(fs->make<TH1F>("h_normResiduals_tob","Normalized Residuals in TOB;(x_{pred} - x_{rec})/#sqrt{V}", 
					   i_normres_Nbins,d_normres_xmin,d_normres_xmax));   
  v_subdetnormres.push_back(fs->make<TH1F>("h_normResiduals_tec","Normalized Residuals in TEC;(x_{pred} - x_{rec})/#sqrt{V}", 
					   i_normres_Nbins,d_normres_xmin,d_normres_xmax));


  

  
  edm::LogInfo("TrackerOfflineValidation") << "There are " << (*tkgeom).detIds().size() << " detUnits in the Geometry record" << std::endl;

  // recursively book histogramms on lowest level
  this->bookDirHists(static_cast<TFileDirectory&>(*fs), *alitracker, aliobjid);
  
 
}

void 
TrackerOfflineValidation::bookGlobalHists(TFileDirectory &tfd )
{

  v_trackhistos.push_back(tfd.make<TH1F>("h_tracketa","Track #eta;#eta_{Track};Number of Tracks",90,-3.,3.));	       
  v_trackhistos.push_back(tfd.make<TH1F>("h_curvature","Curvature #kappa;#kappa_{Track};Number of Tracks",100,-.05,.05));
  v_trackhistos.push_back(tfd.make<TH1F>("h_curvature_pos","Curvature |#kappa| Positive Tracks;|#kappa_{pos Track}|;Number of Tracks",100,.0,.05));
  v_trackhistos.push_back(tfd.make<TH1F>("h_curvature_neg","Curvature |#kappa| Negative Tracks;|#kappa_{neg Track}|;Number of Tracks",100,.0,.05));
  v_trackhistos.push_back(tfd.make<TH1F>("h_diff_curvature","Curvature |#kappa| Tracks Difference;|#kappa_{Track}|;# Pos Tracks - # Neg Tracks",100,.0,.05));
  v_trackhistos.push_back(tfd.make<TH1F>("h_chi2","#chi^{2};#chi^{2}_{Track};Number of Tracks",500,-0.01,500.));	       
  v_trackhistos.push_back(tfd.make<TH1F>("h_normchi2","#chi^{2}/ndof;#chi^{2}/ndof;Number of Tracks",100,-0.01,10.));     
  v_trackhistos.push_back(tfd.make<TH1F>("h_pt","p_{T};p_{T}^{track};Number of Tracks",100,0.,2500));                     

  v_trackprofiles.push_back(tfd.make<TProfile>("h_d0_vs_phi","Transverse Impact Parameter vs. #phi;#phi_{Track};#LT d_{0} #GT",100,-3.15,3.15));  
  v_trackprofiles.push_back(tfd.make<TProfile>("h_dz_vs_phi","Longitudinal Impact Parameter vs. #phi;#phi_{Track};#LT d_{z} #GT",100,-3.15,3.15));
  v_trackprofiles.push_back(tfd.make<TProfile>("h_d0_vs_eta","Transverse Impact Parameter vs. #eta;#eta_{Track};#LT d_{0} #GT",100,-3.15,3.15));  
  v_trackprofiles.push_back(tfd.make<TProfile>("h_dz_vs_eta","Longitudinal Impact Parameter vs. #eta;#eta_{Track};#LT d_{z} #GT",100,-3.15,3.15));
  v_trackprofiles.push_back(tfd.make<TProfile>("h_chi2_vs_phi","#chi^{2} vs. #phi;#phi_{Track};#LT #chi^{2} #GT",100,-3.15,3.15));		  
  v_trackprofiles.push_back(tfd.make<TProfile>("h_normchi2_vs_phi","#chi^{2}/ndof vs. #phi;#phi_{Track};#LT #chi^{2}/ndof #GT",100,-3.15,3.15));  
  v_trackprofiles.push_back(tfd.make<TProfile>("h_chi2_vs_eta","#chi^{2} vs. #eta;#eta_{Track};#LT #chi^{2} #GT",100,-3.15,3.15));		  
  v_trackprofiles.push_back(tfd.make<TProfile>("h_normchi2_vs_eta","#chi^{2}/ndof vs. #eta;#eta_{Track};#LT #chi^{2}/ndof #GT",100,-3.15,3.15));  

  v_track2Dhistos.push_back(tfd.make<TH2F>("h2_d0_vs_phi","Transverse Impact Parameter vs. #phi;#phi_{Track};d_{0}",100, -3.15, 3.15, 100,-1.,1.) );  
  v_track2Dhistos.push_back(tfd.make<TH2F>("h2_dz_vs_phi","Longitudinal Impact Parameter vs. #phi;#phi_{Track};d_{z}",100, -3.15, 3.15, 100,-100.,100.));
  v_track2Dhistos.push_back(tfd.make<TH2F>("h2_d0_vs_eta","Transverse Impact Parameter vs. #eta;#eta_{Track};d_{0}",100, -3.15, 3.15, 100,-1.,1.));  
  v_track2Dhistos.push_back(tfd.make<TH2F>("h2_dz_vs_eta","Longitudinal Impact Parameter vs. #eta;#eta_{Track};d_{z}",100, -3.15, 3.15, 100,-100.,100.));
  v_track2Dhistos.push_back(tfd.make<TH2F>("h2_chi2_vs_phi","#chi^{2} vs. #phi;#phi_{Track};#chi^{2}",100, -3.15, 3.15, 500, 0., 500.));		  
  v_track2Dhistos.push_back(tfd.make<TH2F>("h2_normchi2_vs_phi","#chi^{2}/ndof vs. #phi;#phi_{Track};#chi^{2}/ndof",100, -3.15, 3.15, 100, 0., 10.));  
  v_track2Dhistos.push_back(tfd.make<TH2F>("h2_chi2_vs_eta","#chi^{2} vs. #eta;#eta_{Track};#chi^{2}",100, -3.15, 3.15, 500, 0., 500.));		  
  v_track2Dhistos.push_back(tfd.make<TH2F>("h2_normchi2_vs_eta","#chi^{2}/ndof vs. #eta;#eta_{Track};#chi^{2}/ndof",100,-3.15,3.15, 100, 0., 10.));  
  v_track2Dhistos.push_back(tfd.make<TH2F>("h2_kappa_vs_phi","#kappa vs. #phi;#phi_{Track};#kappa",100,-3.15,3.15, 100, .0,.05));  
  



}


void
TrackerOfflineValidation::bookDirHists( TFileDirectory &tfd, const Alignable& ali, const AlignableObjectId &aliobjid)
{
  std::vector<Alignable*> alivec(ali.components());
  for(int i=0, iEnd = ali.components().size();i < iEnd; ++i) {
    std::string structurename  = aliobjid.typeToName((alivec)[i]->alignableObjectId());
    LogDebug("TrackerOfflineValidation") << "StructureName = " << structurename;
    std::stringstream dirname;
    
    // add no suffix counter to Strip and Pixel
    // just aesthetics
    if(structurename != "Strip" && structurename != "Pixel") {
      dirname << structurename << "_" << i+1;
    } else {
      dirname << structurename;
    }
    if (structurename.find("Endcap",0) != std::string::npos )   {
      TFileDirectory f = tfd.mkdir((dirname.str()).c_str());
      bookHists(f, *(alivec)[i], ali.alignableObjectId() , i, aliobjid);
      bookDirHists( f, *(alivec)[i], aliobjid);
    } else if((structurename != "Det" && structurename != "DetUnit" ) || alivec[i]->components().size() > 1) {      
      TFileDirectory f = tfd.mkdir((dirname.str()).c_str());
      bookHists(tfd, *(alivec)[i], ali.alignableObjectId() , i, aliobjid);
      bookDirHists( f, *(alivec)[i], aliobjid);
    } else {
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

  TrackerAlignableId aliid;
  const DetId id = ali.id();
  
  std::pair<int,int> subdetandlayer = aliid.typeAndLayerFromDetId(id);

  align::StructureType subtype = align::invalid;
  
  // are we on or just above det, detunit level respectively?
  if (type == align::AlignableDetUnit )
    subtype = type;
  else if(      ali.alignableObjectId() == align::AlignableDet || 
		ali.alignableObjectId() == align::AlignableDetUnit) 
    subtype = ali.alignableObjectId();
  
	
  std::stringstream histoname, histotitle, normhistoname, normhistotitle, xprimehistoname, xprimehistotitle;
  if( subdetandlayer.first == StripSubdetector::TID || subdetandlayer.first == StripSubdetector::TEC ||
      subdetandlayer.first == PixelSubdetector::PixelEndcap ) {
    histoname << "h_residuals_subdet_" << subdetandlayer.first 
	      << "_wheel_" << subdetandlayer.second << "_module_" << id.rawId();
    xprimehistoname << "h_xprime_residuals_subdet_" << subdetandlayer.first 
	      << "_wheel_" << subdetandlayer.second << "_module_" << id.rawId();
    normhistoname << "h_normresiduals_subdet_" << subdetandlayer.first 
		  << "_wheel_" << subdetandlayer.second << "_module_" << id.rawId();
    histotitle << "Residual for module " << id.rawId() << ";x_{pred} - x_{rec} [cm]";
    normhistotitle << "Normalized Residual for module " << id.rawId() << ";x_{pred} - x_{rec}/#sigma";
    xprimehistotitle << "X' Residual for module " << id.rawId() << ";x_{pred} - x_{rec} [cm]";
  } else if (subdetandlayer.first == StripSubdetector::TIB || subdetandlayer.first == StripSubdetector::TOB ||
	     subdetandlayer.first == PixelSubdetector::PixelBarrel ) {
    histoname << "h_residuals_subdet_" << subdetandlayer.first 
	      << "_layer_" << subdetandlayer.second << "_module_" << id.rawId();
    xprimehistoname << "h_xprime_residuals_subdet_" << subdetandlayer.first 
		    << "_layer_" << subdetandlayer.second << "_module_" << id.rawId();
    normhistoname << "h_normresiduals_subdet_" << subdetandlayer.first 
		  << "_layer_" << subdetandlayer.second << "_module_" << id.rawId();
    histotitle << "Residual for module " << id.rawId() << ";x_{pred} - x_{rec} [cm]";
    normhistotitle << "Normalized Residual for module " << id.rawId() << ";x_{pred} - x_{rec}/#sigma";
    xprimehistotitle << "X' Residual for module " << id.rawId() << ";x_{pred} - x_{rec} [cm]";
  } else {
    edm::LogWarning("TrackerOfflineValidation") << "@SUB=TrackerOfflineValidation::bookHists" 
						<< "Unknown subdetid: " <<  subdetandlayer.first; 
    
  }
  
  
  if(subtype == align::AlignableDet || subtype == align::AlignableDetUnit) {
    ModuleHistos &histStruct = this->GetHistStructFromMap(id);
    histStruct.ResHisto       =  tfd.make<TH1F>(histoname.str().c_str(),histotitle.str().c_str(),		     
					 i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
    histStruct.NormResHisto   =  tfd.make<TH1F>(normhistoname.str().c_str(),normhistotitle.str().c_str(),
						i_normres_Nbins,d_normres_xmin,d_normres_xmax);
    histStruct.ResXprimeHisto =  tfd.make<TH1F>(xprimehistoname.str().c_str(),xprimehistotitle.str().c_str(),
						i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
  }
  
}


TrackerOfflineValidation::ModuleHistos& TrackerOfflineValidation::GetHistStructFromMap(const DetId& detid)
{

  uint subdetid = detid.subdetId();
  if(subdetid == PixelSubdetector::PixelBarrel ) {
    return m_pxbResiduals[detid.rawId()];
  } else if (subdetid == PixelSubdetector::PixelEndcap) {      
    return m_pxeResiduals[detid.rawId()];
  } else if(subdetid  == StripSubdetector::TIB) {
    return m_tibResiduals[detid.rawId()];
  } else if(subdetid  == StripSubdetector::TID) {
    return m_tidResiduals[detid.rawId()];
  } else if(subdetid  == StripSubdetector::TOB) {
    return m_tobResiduals[detid.rawId()];
  } else if(subdetid  == StripSubdetector::TEC) {
    return m_tecResiduals[detid.rawId()];
  } else {
    throw cms::Exception("Geometry Error") 
      << "[TrackerOfflineValidation] Error, tried to get reference for non-tracker subdet " << subdetid 
      << " from detector " << detid.det();
    return m_pxbResiduals[0];
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
    static const int kappaphiindex_2d = this->GetIndex(v_track2Dhistos,"h2_kappa_vs_phi");
    v_track2Dhistos[kappaphiindex_2d]->Fill(it->phi,it->kappa);
     
  } // finish loop over track quantities


  // hit quantities: residuals, normalized residuals
  for (std::vector<TrackerValidationVariables::AVHitStruct>::const_iterator it = v_hitstruct.begin(),
  	 itEnd = v_hitstruct.end(); it != itEnd; ++it) {
    DetId detid(it->rawDetId);
    ModuleHistos &histStruct = this->GetHistStructFromMap(detid);
    histStruct.ResHisto->Fill(it->resX);
    if(it->resXprime != -999)  histStruct.ResXprimeHisto->Fill(it->resXprime);
    if(it->resErrX != 0)       histStruct.NormResHisto->Fill(it->resX/it->resErrX);
    
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
  for(std::map<int, TrackerOfflineValidation::ModuleHistos>::const_iterator itPxb = m_pxbResiduals.begin(), 
	itEnd = m_pxbResiduals.end(); itPxb != itEnd;++itPxb ) {
    v_subdetres[PixelSubdetector::PixelBarrel - 1]->Add(itPxb->second.ResHisto); 
    v_subdetXprimeRes[PixelSubdetector::PixelBarrel - 1]->Add(itPxb->second.ResXprimeHisto); 
    v_subdetnormres[PixelSubdetector::PixelBarrel - 1]->Add(itPxb->second.NormResHisto);
  }
  for(std::map<int, TrackerOfflineValidation::ModuleHistos>::const_iterator itPxe = m_pxeResiduals.begin(), 
	itEnd = m_pxeResiduals.end(); itPxe != itEnd;++itPxe ) {
    v_subdetres[PixelSubdetector::PixelEndcap - 1]->Add(itPxe->second.ResHisto);
    v_subdetXprimeRes[PixelSubdetector::PixelEndcap - 1]->Add(itPxe->second.ResXprimeHisto);
    v_subdetnormres[PixelSubdetector::PixelEndcap - 1]->Add(itPxe->second.NormResHisto);
  }
  for(std::map<int, TrackerOfflineValidation::ModuleHistos>::const_iterator itTib = m_tibResiduals.begin(), 
	itEnd = m_tibResiduals.end(); itTib != itEnd;++itTib ) {
    v_subdetres[StripSubdetector::TIB - 1]->Add(itTib->second.ResHisto);
    v_subdetXprimeRes[StripSubdetector::TIB - 1]->Add(itTib->second.ResXprimeHisto);
    v_subdetnormres[StripSubdetector::TIB -1]->Add(itTib->second.NormResHisto);
  }
  for(std::map<int, TrackerOfflineValidation::ModuleHistos>::const_iterator itTid = m_tidResiduals.begin(), 
	itEnd = m_tidResiduals.end(); itTid != itEnd;++itTid ) {
    v_subdetres[StripSubdetector::TID - 1]->Add(itTid->second.ResHisto);
    v_subdetXprimeRes[StripSubdetector::TID - 1]->Add(itTid->second.ResXprimeHisto);
    v_subdetnormres[StripSubdetector::TID - 1]->Add(itTid->second.NormResHisto);
  }
  for(std::map<int, TrackerOfflineValidation::ModuleHistos>::const_iterator itTob = m_tobResiduals.begin(), 
	itEnd = m_tobResiduals.end(); itTob != itEnd;++itTob ) {
    v_subdetres[StripSubdetector::TOB - 1]->Add(itTob->second.ResHisto);
    v_subdetXprimeRes[StripSubdetector::TOB - 1]->Add(itTob->second.ResXprimeHisto);
    v_subdetnormres[StripSubdetector::TOB - 1]->Add(itTob->second.NormResHisto);
  }
  for(std::map<int, TrackerOfflineValidation::ModuleHistos>::const_iterator itTec = m_tecResiduals.begin(), 
	itEnd = m_tecResiduals.end(); itTec != itEnd;++itTec ) {
    v_subdetres[StripSubdetector::TEC - 1]->Add(itTec->second.ResHisto);
    v_subdetXprimeRes[StripSubdetector::TEC - 1]->Add(itTec->second.ResXprimeHisto);
    v_subdetnormres[StripSubdetector::TEC - 1]->Add(itTec->second.NormResHisto);
  }
  
  // create summary histogramms recursively
  std::vector<std::pair<TH1*,TH1*> > v_trackerprofiles;
  collateSummaryHists((*fs),(*alitracker), 0, aliobjid, v_trackerprofiles);

  
}


void
TrackerOfflineValidation::collateSummaryHists( TFileDirectory &tfd, const Alignable& ali, int i, const AlignableObjectId &aliobjid, std::vector<std::pair<TH1*,TH1*> > &v_levelprofiles)
{
 
  std::vector<Alignable*> alivec(ali.components());
   
  if((aliobjid.typeToName((alivec)[0]->alignableObjectId()) == "Det" || 
      aliobjid.typeToName((alivec)[0]->alignableObjectId()) == "DetUnit") 
     )  {
     return;
  }

  for(int iComp=0, iCompEnd = ali.components().size();iComp < iCompEnd; ++iComp) {

    std::vector<std::pair<TH1*,TH1*> > v_profiles;        
    std::string structurename  = aliobjid.typeToName((alivec)[iComp]->alignableObjectId());
 
    LogDebug("TrackerOfflineValidation") << "StructureName = " << structurename;
    std::stringstream dirname;
    
    // add no suffix counter to strip and pixel
    // just aesthetics
    if(structurename != "Strip" && structurename != "Pixel") {
      dirname << structurename << "_" << iComp+1;
    } else {
      dirname << structurename;
    }
    
    if( (structurename != "Det" && structurename != "DetUnit" )  || (alivec)[0]->components().size() > 1
       ) {
      TFileDirectory f = tfd.mkdir((dirname.str()).c_str());
      collateSummaryHists( f, *(alivec)[iComp], i, aliobjid, v_profiles);
    
      v_levelprofiles.push_back(bookSummaryHists(tfd, *(alivec[iComp]), ali.alignableObjectId(), iComp, aliobjid));
      
      for(uint n = 0; n < v_profiles.size(); ++n) {
	v_levelprofiles[iComp].first->SetBinContent(n+1,v_profiles[n].second->GetMean(1));
	v_levelprofiles[iComp].first->SetBinError(n+1,Fwhm(v_profiles[n].second)/2.);
	v_levelprofiles[iComp].second->Add(v_profiles[n].second);
      }
    } else {
      // nothing to be done for det or detunits
      continue;
    }

  }

}

std::pair<TH1*,TH1*> 
TrackerOfflineValidation::bookSummaryHists(TFileDirectory &tfd, const Alignable& ali, align::StructureType type, int i, const AlignableObjectId &aliobjid)
{
  Parameters = parset_.getParameter<edm::ParameterSet>("TH1ResModules");
  int32_t i_residuals_Nbins =  Parameters.getParameter<int32_t>("Nbinx");
  double d_residual_xmin = Parameters.getParameter<double>("xmin");
  double d_residual_xmax = Parameters.getParameter<double>("xmax");
  uint subsize = ali.components().size();
  align::StructureType alitype = ali.alignableObjectId();
  align::StructureType subtype = ali.components()[0]->alignableObjectId();
  TH1 *h_thissummary = 0;
  
  if( subtype  != align::AlignableDet && subtype  != align::AlignableDetUnit ) {
    h_thissummary = tfd.make<TH1F>(Form("h_summary%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
				Form("Summary for substructures in %s %d;%s;#LT #Delta x #GT",
				     aliobjid.typeToName(alitype).c_str(),i,aliobjid.typeToName(subtype).c_str()),
				subsize,0.5,subsize+0.5)  ;
   
  } else if( subtype == align::AlignableDet && subsize > 1) {
    h_thissummary = tfd.make<TH1F>(Form("h_summary%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
				Form("Summary for substructures in %s %d;%s;#LT #Delta x #GT",
				     aliobjid.typeToName(alitype).c_str(),i,aliobjid.typeToName(subtype).c_str()),
				(2*subsize),0.5,2*subsize+0.5)  ;  
  } else {
    edm::LogWarning("TrackerOfflineValidation") << "@SUB=TrackerOfflineValidation::bookSummaryHists" 
						<< "No summary histogramm for hierarchy level" << aliobjid.typeToName(subtype);      
  }

  TH1* h_this = tfd.make<TH1F>(Form("h_%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
				Form("Residual for %s %d in %s ",aliobjid.typeToName(alitype).c_str(),i,
				     aliobjid.typeToName(type).c_str(),aliobjid.typeToName(subtype).c_str()),
				i_residuals_Nbins,d_residual_xmin,d_residual_xmax);
  
  // special case I: For DetUnits and Detwith  only one subcomponent start filling summary histos
  if( (  subtype == align::AlignableDet && ali.components()[0]->components().size() == 1) || 
      subtype  == align::AlignableDetUnit  
      ) {
    for(uint k=0;k<subsize;++k) {
      DetId detid = ali.components()[k]->id();
      ModuleHistos &histStruct = this->GetHistStructFromMap(detid);
      h_thissummary->SetBinContent(k+1, histStruct.ResHisto->GetMean());
      h_thissummary->SetBinError(k+1, histStruct.ResHisto->GetRMS());
      h_this->Add(histStruct.ResHisto);
    }
    
  }
  // special case II: Fill summary histos for dets with two detunits 
  else if( subtype == align::AlignableDet && subsize > 1) {
    for(uint k = 0; k < subsize; ++k) { 
      int jEnd = ali.components()[0]->components().size();
      for(uint j = 0; j <  jEnd; ++j) {
	DetId detid = ali.components()[k]->components()[j]->id();
	ModuleHistos &histStruct = this->GetHistStructFromMap(detid);	
	h_thissummary->SetBinContent(2*k+j+1,histStruct.ResHisto->GetMean());
	h_thissummary->SetBinError(2*k+j+1,histStruct.ResHisto->GetRMS());
	h_this->Add( histStruct.ResHisto);

      }
    }
  }
  

  return std::make_pair(h_thissummary,h_this);
  
  
}


float 
TrackerOfflineValidation::Fwhm (const TH1* hist) 
{
  float fwhm = 0.;
  float max = hist->GetMaximum();
  int left = -1, right = -1;
  for(unsigned int i = 1, iEnd = hist->GetNbinsX(); i <= iEnd; ++i) {
    if(hist->GetBinContent(i) < max/2. && hist->GetBinContent(i+1) > max/2. && left == -1) {
      if(max/2. - hist->GetBinContent(i) < hist->GetBinContent(i+1) - max/2.) {
	left = i;
	++i;
      } else {
	left = i+1;
	++i;
      }
    }
    if(left != -1 && right == -1) {
      if(hist->GetBinContent(i) > max/2. && hist->GetBinContent(i+1) < max/2.) {
	if( hist->GetBinContent(i) - max/2. < max/2. - hist->GetBinContent(i+1)) {
	  right = i;
	} else {
	  right = i+1;
	}
	
      }
    }
  }
  fwhm = hist->GetXaxis()->GetBinCenter(right) - hist->GetXaxis()->GetBinCenter(left);
  return fwhm;
}


//define this as a plug-in
DEFINE_FWK_MODULE(TrackerOfflineValidation);
