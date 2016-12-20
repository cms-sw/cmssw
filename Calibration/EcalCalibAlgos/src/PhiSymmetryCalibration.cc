#include "Calibration/EcalCalibAlgos/interface/PhiSymmetryCalibration.h"

// System include files
#include <memory>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsXMLTranslator.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

// Geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

//Channel status
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"

#include "FWCore/Framework/interface/Run.h"






#include "boost/filesystem/operations.hpp"

using namespace std;
#include <fstream>
#include <iostream>
#include "TH2F.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TF1.h"
#include "TGraph.h"
#include "TCanvas.h"

const float PhiSymmetryCalibration::kMiscalRangeEB = .05;
const float PhiSymmetryCalibration::kMiscalRangeEE = .10;




//_____________________________________________________________________________
// Class constructor

PhiSymmetryCalibration::PhiSymmetryCalibration(const edm::ParameterSet& iConfig) :

  ecalHitsProducer_(iConfig.getParameter<std::string>("ecalRecHitsProducer")),
  barrelHits_( iConfig.getParameter< std::string > ("barrelHitCollection")),
  endcapHits_( iConfig.getParameter< std::string > ("endcapHitCollection")),
  eCut_barl_( iConfig.getParameter< double > ("eCut_barrel") ),
  ap_( iConfig.getParameter<double> ("ap") ),
  b_( iConfig.getParameter<double> ("b") ), 
  eventSet_( iConfig.getParameter< int > ("eventSet") ),
  statusThreshold_(iConfig.getUntrackedParameter<int>("statusThreshold",3)),
  reiteration_(iConfig.getUntrackedParameter< bool > ("reiteration",false)),
  oldcalibfile_(iConfig.getUntrackedParameter<std::string>("oldcalibfile",
                                            "EcalintercalibConstants.xml"))
{


  isfirstpass_=true;

  et_spectrum_b_histos.resize(kBarlRings);
  e_spectrum_b_histos.resize(kBarlRings);
  et_spectrum_e_histos.resize(kEndcEtaRings);
  e_spectrum_e_histos.resize(kEndcEtaRings); 
  
  spectra=true;

  nevents_=0;
  eventsinrun_=0;
  eventsinlb_=0;
}


//_____________________________________________________________________________
// Close files, etc.

PhiSymmetryCalibration::~PhiSymmetryCalibration()
{


  for(Int_t i=0;i<kBarlRings;i++){
    delete et_spectrum_b_histos[i];
    delete e_spectrum_b_histos[i];
    
  }
  for(Int_t i=0;i<kEndcEtaRings;i++){
    delete et_spectrum_e_histos[i];
    delete e_spectrum_e_histos[i];    
  }

  
}


//_____________________________________________________________________________
// Initialize algorithm

void PhiSymmetryCalibration::beginJob( )
{


  // initialize arrays
  for (int sign=0; sign<kSides; sign++) {
    for (int ieta=0; ieta<kBarlRings; ieta++) {
      for (int iphi=0; iphi<kBarlWedges; iphi++) {
	etsum_barl_[ieta][iphi][sign]=0.;
	nhits_barl_[ieta][iphi][sign]=0;

      }
    }
    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	etsum_endc_[ix][iy][sign]=0.;
	nhits_endc_[ix][iy][sign]=0;
      }
    }
  }



  for (int imiscal=0; imiscal<kNMiscalBinsEB; imiscal++) {
    miscalEB_[imiscal]= (1-kMiscalRangeEB) + float(imiscal)* (2*kMiscalRangeEB/(kNMiscalBinsEB-1));
    for (int ieta=0; ieta<kBarlRings; ieta++) etsum_barl_miscal_[imiscal][ieta]=0.;
  }

  for (int imiscal=0; imiscal<kNMiscalBinsEE; imiscal++) {
    miscalEE_[imiscal]= (1-kMiscalRangeEE) + float(imiscal)* (2*kMiscalRangeEE/(kNMiscalBinsEE-1));
    for (int ring=0; ring<kEndcEtaRings; ring++) etsum_endc_miscal_[imiscal][ring]=0.;
  }



  // start spectra stuff
  if (eventSet_!=1) spectra = false;
  
  if(spectra)
    {
      ostringstream t;
      for(Int_t i=0;i<kBarlRings;i++)
	{
	  t << "et_spectrum_b_" << i+1;
	  et_spectrum_b_histos[i]=new TH1F(t.str().c_str(),";E_{T} [MeV]",50,0.,500.);
	  t.str("");
	  
	  t << "e_spectrum_b_" << i+1;
	  e_spectrum_b_histos[i]=new TH1F(t.str().c_str(),";E [MeV]",50,0.,500.);
	  t.str("");
	
	}
      for(Int_t i=0;i<kEndcEtaRings;i++)
	{
	  t << "et_spectrum_e_" << i+1;
	  et_spectrum_e_histos[i]=new TH1F(t.str().c_str(),";E_{T} [MeV]",75,0.,1500.);
	  t.str("");
	  
	  t << "e_spectrum_e_" << i+1;
	  e_spectrum_e_histos[i]=new TH1F(t.str().c_str(),";E [MeV]",75,0.,1500.);
	  t.str("");
	
	}
    }
  // end spectra stuff
}


//_____________________________________________________________________________
// Terminate algorithm

void PhiSymmetryCalibration::endJob()
{

  edm::LogInfo("Calibration") << "[PhiSymmetryCalibration] At end of job";

  // start spectra stuff
  if(spectra)
    {
      TFile f("Espectra_plus.root","recreate");

      for(int i=0;i<kBarlRings;i++){
	et_spectrum_b_histos[i]->Write();
	e_spectrum_b_histos[i]->Write();
      }

      for(int i=0;i<kEndcEtaRings;i++){
	et_spectrum_e_histos[i]->Write();
	e_spectrum_e_histos[i]->Write();
      }

      f.Close();
    }
  



  if (eventSet_==1) {
    // calculate factors to convert from fractional deviation of ET sum from 
    // the mean to the estimate of the miscalibration factor
    getKfactors();

    std::ofstream k_barl_out("k_barl.dat", ios::out);
    for (int ieta=0; ieta<kBarlRings; ieta++)
      k_barl_out << ieta << " " << k_barl_[ieta] << endl;
    k_barl_out.close();

    std::ofstream k_endc_out("k_endc.dat", ios::out);
    for (int ring=0; ring<kEndcEtaRings; ring++)
      k_endc_out << ring << " " << k_endc_[ring] << endl;
    k_endc_out.close();
  }


  if (eventSet_!=0) {
    // output ET sums

    stringstream etsum_file_barl;
    etsum_file_barl << "etsum_barl_"<<eventSet_<<".dat";

    std::ofstream etsum_barl_out(etsum_file_barl.str().c_str(),ios::out);

    for (int ieta=0; ieta<kBarlRings; ieta++) {
      for (int iphi=0; iphi<kBarlWedges; iphi++) {
	for (int sign=0; sign<kSides; sign++) {
	  etsum_barl_out << eventSet_ << " " << ieta << " " << iphi << " " << sign 
		         << " " << etsum_barl_[ieta][iphi][sign] << " " 
                         <<  nhits_barl_[ieta][iphi][sign] << endl;
	}
      }
    }
    etsum_barl_out.close();

    stringstream etsum_file_endc;
    etsum_file_endc << "etsum_endc_"<<eventSet_<<".dat";

    std::ofstream etsum_endc_out(etsum_file_endc.str().c_str(),ios::out);
    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	int ring = e_.endcapRing_[ix][iy];
	if (ring!=-1) {
	  for (int sign=0; sign<kSides; sign++) {
	    etsum_endc_out << eventSet_ << " " << ix << " " << iy << " " << sign 
			   << " " << etsum_endc_[ix][iy][sign] << " " 
			   << nhits_endc_[ix][iy][sign]<<" " 
                           << e_.endcapRing_[ix][iy]<<endl;
	  }
	}
      }
    }
    etsum_endc_out.close();
  } 
  cout<<"Events processed " << nevents_<< endl;
}


//_____________________________________________________________________________
// Called at each event

void PhiSymmetryCalibration::analyze( const edm::Event& event, const edm::EventSetup& setup )
{
  using namespace edm;
  using namespace std;

  if (isfirstpass_) {
    setUp(setup);
    isfirstpass_=false;
  }

  
  Handle<EBRecHitCollection> barrelRecHitsHandle;
  Handle<EERecHitCollection> endcapRecHitsHandle;
  
  event.getByLabel(ecalHitsProducer_,barrelHits_,barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    LogError("") << "[PhiSymmetryCalibration] Error! Can't get product!" << std::endl;
  }
  
  event.getByLabel(ecalHitsProducer_,endcapHits_,endcapRecHitsHandle);
  if (!endcapRecHitsHandle.isValid()) {
    LogError("") << "[PhiSymmetryCalibration] Error! Can't get product!" << std::endl;
  }
  
 
  // get the ecal geometry
  edm::ESHandle<CaloGeometry> geoHandle;
  setup.get<CaloGeometryRecord>().get(geoHandle);
  const CaloSubdetectorGeometry *barrelGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  const CaloSubdetectorGeometry *endcapGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
 
  bool pass=false;
  // select interesting EcalRecHits (barrel)
  EBRecHitCollection::const_iterator itb;
  for (itb=barrelRecHitsHandle->begin(); itb!=barrelRecHitsHandle->end(); itb++) {
    EBDetId hit = EBDetId(itb->id());
    float eta = barrelGeometry->getGeometry(hit)->getPosition().eta();
    float et = itb->energy()/cosh(eta);
    float e  = itb->energy();
    
    

    // if iterating, correct by the previous calib constants found,
    // which are supplied in the form of correction 
    if (reiteration_) {
      et= et  * oldCalibs_[hit];
      e = e  * oldCalibs_[hit];
    }

    float et_thr = eCut_barl_/cosh(eta) + 1.;

    int sign = hit.ieta()>0 ? 1 : 0;

    if (e >  eCut_barl_ && et < et_thr && e_.goodCell_barl[abs(hit.ieta())-1][hit.iphi()-1][sign]) {
      etsum_barl_[abs(hit.ieta())-1][hit.iphi()-1][sign] += et;
      nhits_barl_[abs(hit.ieta())-1][hit.iphi()-1][sign] ++;
      pass =true;
    }//if energy

    if (eventSet_==1) {
      // apply a miscalibration to all crystals and increment the 
      // ET sum, combined for all crystals
      for (int imiscal=0; imiscal<kNMiscalBinsEB; imiscal++) {
	if (miscalEB_[imiscal]*e >  eCut_barl_&& miscalEB_[imiscal]*et < et_thr && e_.goodCell_barl[abs(hit.ieta())-1][hit.iphi()-1][sign]) {
	  etsum_barl_miscal_[imiscal][abs(hit.ieta())-1] += miscalEB_[imiscal]*et;
	}
      }

      // spectra stuff
      if(spectra && hit.ieta()>0) //POSITIVE!!!
	//      if(spectra && hit.ieta()<0) //NEGATIVE!!!
	{
	  et_spectrum_b_histos[abs(hit.ieta())-1]->Fill(et*1000.);
	  e_spectrum_b_histos[abs(hit.ieta())-1]->Fill(e*1000.);
	}//if spectra
      
    }//if eventSet_==1
  }//for barl


  // select interesting EcalRecHits (endcaps)
  EERecHitCollection::const_iterator ite;
  for (ite=endcapRecHitsHandle->begin(); ite!=endcapRecHitsHandle->end(); ite++) {
    EEDetId hit = EEDetId(ite->id());
    float eta = abs(endcapGeometry->getGeometry(hit)->getPosition().eta());
    //float phi = endcapGeometry->getGeometry(hit)->getPosition().phi();

    float et = ite->energy()/cosh(eta);
    float e  = ite->energy();

    // if iterating, multiply by the previous correction factor
    if (reiteration_) {
      et= et * oldCalibs_[hit];
      e = e * oldCalibs_[hit];
    }

    int sign = hit.zside()>0 ? 1 : 0;


    // changes of eCut_endc_ -> variable linearthr 
    // e_cut = ap + eta_ring*b

    double eCut_endc=0;
    for (int ring=0; ring<kEndcEtaRings; ring++) {

      if(eta>e_.etaBoundary_[ring] && eta<e_.etaBoundary_[ring+1])
	{  
	  float eta_ring= abs(e_.cellPos_[ring][50].eta())  ;
	  eCut_endc = ap_ + eta_ring*b_;

	}
    }


    float et_thr = eCut_endc/cosh(eta) + 1.;
   
    if (e > eCut_endc && et < et_thr && e_.goodCell_endc[hit.ix()-1][hit.iy()-1][sign]){
      etsum_endc_[hit.ix()-1][hit.iy()-1][sign] += et;
      nhits_endc_[hit.ix()-1][hit.iy()-1][sign] ++;
      pass=true;
    }
 
   

    if (eventSet_==1) {
      // apply a miscalibration to all crystals and increment the 
      // ET sum, combined for all crystals
      for (int imiscal=0; imiscal<kNMiscalBinsEE; imiscal++) {
	if (miscalEE_[imiscal]*e> eCut_endc && et*miscalEE_[imiscal] < et_thr && e_.goodCell_endc[hit.ix()-1][hit.iy()-1][sign]){
	  int ring = e_.endcapRing_[hit.ix()-1][hit.iy()-1];
	  etsum_endc_miscal_[imiscal][ring] += miscalEE_[imiscal]*et;
	}
      }

      // spectra stuff
      if(spectra && hit.zside()>0) //POSITIVE!!!

	{
	  int ring = e_.endcapRing_[hit.ix()-1][hit.iy()-1];

	  et_spectrum_e_histos[ring]->Fill(et*1000.);
	  e_spectrum_e_histos[ring]->Fill(e*1000.);

	  if(ring==16)
	    {
	      //int iphi_endc = 0;
	      for (int ip=0; ip<e_.nRing_[ring]; ip++) {
		//if (phi==e_.phi_endc_[ip][ring]) iphi_endc=ip;
	      }

	    }
	}//if spectra

    }//if eventSet_==1
  }//for endc

  if (pass) {
    nevents_++;
    eventsinrun_++;
    eventsinlb_++;
  }
}

void PhiSymmetryCalibration::endRun(edm::Run& run, const edm::EventSetup&){
 
  
  std::cout  << "PHIREPRT : run "<< run.run() 
             << " start " << (run.beginTime().value()>>32)
             << " end "   << (run.endTime().value()>>32) 
             << " dur "   << (run.endTime().value()>>32)- (run.beginTime().value()>>32)
	  
             << " npass "      << eventsinrun_  << std::endl;
  eventsinrun_=0;        
 
  return ;

}

//_____________________________________________________________________________

void PhiSymmetryCalibration::getKfactors()
{

  float epsilon_T_eb[kNMiscalBinsEB];
  float epsilon_M_eb[kNMiscalBinsEB];

  float epsilon_T_ee[kNMiscalBinsEE];
  float epsilon_M_ee[kNMiscalBinsEE];

  std::vector<TGraph*>  k_barl_graph(kBarlRings);
  std::vector<TCanvas*> k_barl_plot(kBarlRings);

  //Create our own TF1 to avoid threading problems
  TF1 mypol1("mypol1","pol1");
  for (int ieta=0; ieta<kBarlRings; ieta++) {
    for (int imiscal=0; imiscal<kNMiscalBinsEB; imiscal++) {
      int middlebin =  int (kNMiscalBinsEB/2);
      epsilon_T_eb[imiscal] = etsum_barl_miscal_[imiscal][ieta]/etsum_barl_miscal_[middlebin][ieta] - 1.;
      epsilon_M_eb[imiscal] = miscalEB_[imiscal] - 1.;
    }
    k_barl_graph[ieta] = new TGraph (kNMiscalBinsEB,epsilon_M_eb,epsilon_T_eb);
    k_barl_graph[ieta]->Fit(&mypol1);

    ostringstream t;
    t<< "k_barl_" << ieta+1; 
    k_barl_plot[ieta] = new TCanvas(t.str().c_str(),"");
    k_barl_plot[ieta]->SetFillColor(10);
    k_barl_plot[ieta]->SetGrid();
    k_barl_graph[ieta]->SetMarkerSize(1.);
    k_barl_graph[ieta]->SetMarkerColor(4);
    k_barl_graph[ieta]->SetMarkerStyle(20);
    k_barl_graph[ieta]->GetXaxis()->SetLimits(-1.*kMiscalRangeEB,kMiscalRangeEB);
    k_barl_graph[ieta]->GetXaxis()->SetTitleSize(.05);
    k_barl_graph[ieta]->GetYaxis()->SetTitleSize(.05);
    k_barl_graph[ieta]->GetXaxis()->SetTitle("#epsilon_{M}");
    k_barl_graph[ieta]->GetYaxis()->SetTitle("#epsilon_{T}");
    k_barl_graph[ieta]->Draw("AP");

    k_barl_[ieta] = k_barl_graph[ieta]->GetFunction("pol1")->GetParameter(1);
    std::cout << "k_barl_[" << ieta << "]=" << k_barl_[ieta] << std::endl;
  }


  std::vector<TGraph*>  k_endc_graph(kEndcEtaRings);
  std::vector<TCanvas*> k_endc_plot(kEndcEtaRings);

  for (int ring=0; ring<kEndcEtaRings; ring++) {
    for (int imiscal=0; imiscal<kNMiscalBinsEE; imiscal++) {
      int middlebin =  int (kNMiscalBinsEE/2);
      epsilon_T_ee[imiscal] = etsum_endc_miscal_[imiscal][ring]/etsum_endc_miscal_[middlebin][ring] - 1.;
      epsilon_M_ee[imiscal] = miscalEE_[imiscal] - 1.;
    }
    k_endc_graph[ring] = new TGraph (kNMiscalBinsEE,epsilon_M_ee,epsilon_T_ee);
    k_endc_graph[ring]->Fit(&mypol1);

    ostringstream t;
    t<< "k_endc_"<< ring+1;
    k_endc_plot[ring] = new TCanvas(t.str().c_str(),"");
    k_endc_plot[ring]->SetFillColor(10);
    k_endc_plot[ring]->SetGrid();
    k_endc_graph[ring]->SetMarkerSize(1.);
    k_endc_graph[ring]->SetMarkerColor(4);
    k_endc_graph[ring]->SetMarkerStyle(20);
    k_endc_graph[ring]->GetXaxis()->SetLimits(-1*kMiscalRangeEE,kMiscalRangeEE);
    k_endc_graph[ring]->GetXaxis()->SetTitleSize(.05);
    k_endc_graph[ring]->GetYaxis()->SetTitleSize(.05);
    k_endc_graph[ring]->GetXaxis()->SetTitle("#epsilon_{M}");
    k_endc_graph[ring]->GetYaxis()->SetTitle("#epsilon_{T}");
    k_endc_graph[ring]->Draw("AP");

    k_endc_[ring] = k_endc_graph[ring]->GetFunction("pol1")->GetParameter(1);
    std::cout << "k_endc_[" << ring << "]=" << k_endc_[ring] << std::endl;
  }
 
  TFile f("PhiSymmetryCalibration_kFactors.root","recreate");
  for (int ieta=0; ieta<kBarlRings; ieta++) { 
    k_barl_plot[ieta]->Write();
    delete k_barl_plot[ieta]; 
    delete k_barl_graph[ieta];
  }
  for (int ring=0; ring<kEndcEtaRings; ring++) { 
    k_endc_plot[ring]->Write();
    delete k_endc_plot[ring];
    delete k_endc_graph[ring];
  }
  f.Close();

}





//_____________________________________________________________________________



void PhiSymmetryCalibration::setUp(const edm::EventSetup& setup){

  edm::ESHandle<EcalChannelStatus> chStatus;
  setup.get<EcalChannelStatusRcd>().get(chStatus);

  edm::ESHandle<CaloGeometry> geoHandle;
  setup.get<CaloGeometryRecord>().get(geoHandle);

  e_.setup(&(*geoHandle), &(*chStatus), statusThreshold_);
 
  
  if (reiteration_){   
    
    EcalCondHeader h;
    // namespace fs = boost::filesystem;
//     fs::path p(oldcalibfile_.c_str(),fs::native);
//     if (!fs::exists(p)) edm::LogError("PhiSym") << "File not found: " 
// 						<< oldcalibfile_ <<endl;
    
    edm::FileInPath fip("Calibration/EcalCalibAlgos/data/"+oldcalibfile_);
    

    
    int ret=
    EcalIntercalibConstantsXMLTranslator::readXML(fip.fullPath(),h,oldCalibs_);    
    if (ret) edm::LogError("PhiSym")<<"Error reading XML files"<<endl;;
    
  } else {
    // in fact if not reiterating, oldCalibs_ will never be used
    edm::ESHandle<EcalIntercalibConstants> pIcal;      
    setup.get<EcalIntercalibConstantsRcd>().get(pIcal);
    oldCalibs_=*pIcal;

  }
  
}


void PhiSymmetryCalibration::endLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const&){

  
  if ((lb.endTime().value()>>32)- (lb.beginTime().value()>>32) <60 ) 
    return;

  std::cout  << "PHILB : run "<< lb.run()
             << " id " << lb.id() 
             << " start " << (lb.beginTime().value()>>32)
             << " end "   << (lb.endTime().value()>>32) 
             << " dur "   << (lb.endTime().value()>>32)- (lb.beginTime().value()>>32)
    
             << " npass "      << eventsinlb_  << std::endl;
  
  eventsinlb_=0;

}
