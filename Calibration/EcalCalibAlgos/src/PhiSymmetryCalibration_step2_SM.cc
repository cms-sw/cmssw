#include "Calibration/EcalCalibAlgos/src/PhiSymmetryCalibration_step2_SM.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsXMLTranslator.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "TH2F.h"

#include "TH1F.h"
#include "TFile.h"

#include <fstream>
#include "boost/filesystem/operations.hpp"

using namespace std;



PhiSymmetryCalibration_step2_SM::~PhiSymmetryCalibration_step2_SM(){}


PhiSymmetryCalibration_step2_SM::PhiSymmetryCalibration_step2_SM(const edm::ParameterSet& iConfig){

  statusThreshold_ =
       iConfig.getUntrackedParameter<int>("statusThreshold",0);
  have_initial_miscalib_=
       iConfig.getUntrackedParameter<bool>("haveInitialMiscalib",false);
  initialmiscalibfile_=
    iConfig.getUntrackedParameter<std::string>("initialmiscalibfile",
					       "InitialMiscalib.xml"); 
  oldcalibfile_=
    iConfig.getUntrackedParameter<std::string>("oldcalibfile",
					       "EcalIntercalibConstants.xml");
  reiteration_ = iConfig.getUntrackedParameter<bool>("reiteration",false);
  firstpass_=true;
}

void PhiSymmetryCalibration_step2_SM::analyze( const edm::Event& ev, 
					    const edm::EventSetup& se){

  if (firstpass_) {
    setUp(se);
    firstpass_=false;
  }
}

void PhiSymmetryCalibration_step2_SM::setUp(const edm::EventSetup& se){

  edm::ESHandle<EcalChannelStatus> chStatus;
  se.get<EcalChannelStatusRcd>().get(chStatus);

  edm::ESHandle<CaloGeometry> geoHandle;
  se.get<CaloGeometryRecord>().get(geoHandle);

  barrelCells = geoHandle->getValidDetIds(DetId::Ecal, EcalBarrel);
  endcapCells = geoHandle->getValidDetIds(DetId::Ecal, EcalEndcap);

  e_.setup(&(*geoHandle), &(*chStatus), statusThreshold_);



  for (int sign=0; sign<kSides; sign++) {
    for (int ieta=0; ieta<kBarlRings; ieta++) {
      for (int iphi=0; iphi<kBarlWedges; iphi++) {
	int iphi_r=int(iphi/nscx);
	if( !e_.goodCell_barl[ieta][iphi][sign] ){
	  nBads_barl_SM_[ieta][iphi_r][sign]++;
	  // std::cout << "N BAD CELL " << nBads_barl_SM_[ieta][iphi_r][sign] << endl; 
         }
      }
    }
  }
  


  /// if a miscalibration was applied, load it, if not put it to 1
  if (have_initial_miscalib_){

    EcalCondHeader h;
    namespace fs = boost::filesystem;
    fs::path p(initialmiscalibfile_.c_str());
    if (!fs::exists(p)) edm::LogError("PhiSym") << "File not found: " 
						<< initialmiscalibfile_ <<endl;  
    
    int ret=
      EcalIntercalibConstantsXMLTranslator::readXML(initialmiscalibfile_,h,miscalib_);    
    if (ret) edm::LogError("PhiSym")<<"Error reading XML files"<<endl;
  } else {

    for (vector<DetId>::iterator it=barrelCells.begin(); it!=barrelCells.end(); ++it){
      miscalib_[*it]=1;
     }

    for (vector<DetId>::iterator it=endcapCells.begin(); it!=endcapCells.end(); ++it){
      miscalib_[*it]=1;
 
    }
  }
    
  // if we are reiterating, read constants from previous iter
  // if not put them to one
  if (reiteration_){
    
     
    EcalCondHeader h;
    namespace fs = boost::filesystem;
    fs::path p(oldcalibfile_.c_str());
    if (!fs::exists(p)) edm::LogError("PhiSym") << "File not found: " 
						<< oldcalibfile_ <<endl;  
    
    int ret=
      EcalIntercalibConstantsXMLTranslator::readXML(oldcalibfile_,h,
						    oldCalibs_); 
   
    if (ret) edm::LogError("PhiSym")<<"Error reading XML files"<<endl;;
    
  } else {
    
    for (vector<DetId>::iterator it=barrelCells.begin(); 
	 it!=barrelCells.end(); ++it)
      oldCalibs_[*it]=1;
     

    for (vector<DetId>::iterator it=endcapCells.begin(); 
	 it!=endcapCells.end(); ++it)
      oldCalibs_[*it]=1;
      
   
  } // else 
  
}


void PhiSymmetryCalibration_step2_SM::beginJob(){
  

  for (int ieta=0; ieta<kBarlRings; ieta++) {
    for (int iphi=0; iphi<kBarlWedges; iphi++) {
      for (int sign=0; sign<kSides; sign++) {
	int iphi_r=int(iphi/nscx);
	
	etsum_barl_[ieta][iphi][sign]=0.;
	nhits_barl_[ieta][iphi][sign]=0;
	esum_barl_[ieta][iphi][sign]=0.;
	etsum_barl_SM_[ieta][iphi_r][sign]=0;
	nBads_barl_SM_[ieta][iphi_r][sign]=0;
	epsilon_M_barl_SM_[ieta][iphi_r][sign]=0;
      }
    }						
    etsumMean_barl_SM_[ieta]=0.;
  }

  for (int ix=0; ix<kEndcWedgesX; ix++) {
    for (int iy=0; iy<kEndcWedgesY; iy++) {
      for (int sign=0; sign<kSides; sign++) {
	etsum_endc_[ix][iy][sign]=0.;
	nhits_endc_[ix][iy][sign]=0;
	esum_endc_[ix][iy][sign]=0.;

      }
    }
  }

  readEtSums();
  setupResidHistos();
}

void PhiSymmetryCalibration_step2_SM::endJob(){

  if (firstpass_) {
    edm::LogError("PhiSym")<< "Must process at least one event-Exiting" <<endl;
    return;
      
  }

  // Here the real calculation of constants happens

  // perform the area correction for endcap etsum
  // NOT  USED  ANYMORE

  /*
  for (int ix=0; ix<kEndcWedgesX; ix++) {
    for (int iy=0; iy<kEndcWedgesY; iy++) {

      int ring = e_.endcapRing_[ix][iy];

      if (ring!=-1) {
	for (int sign=0; sign<kSides; sign++) {
	  etsum_endc_uncorr[ix][iy][sign] = etsum_endc_[ix][iy][sign];
	  etsum_endc_[ix][iy][sign]*=meanCellArea_[ring]/cellArea_[ix][iy];
	}
      }
    }
  }
  */

  // ETsum histos, maps and other usefull histos (area,...)
  // are filled and saved here
  fillHistos();

  // write ETsum mean for all rings
  std::ofstream etsumMean_barl_out("etsumMean_barl.dat",ios::out);
  for (int ieta=0; ieta<kBarlRings; ieta++) {
    etsumMean_barl_out << ieta << " " << etsumMean_barl_[ieta] << endl;
  }
  etsumMean_barl_out.close();

  std::ofstream etsumMean_endc_out("etsumMean_endc.dat",ios::out);
  for (int ring=0; ring<kEndcEtaRings; ring++) {
    etsumMean_endc_out << e_.cellPos_[ring][50].eta() << " " << etsumMean_endc_[ring] << endl;
  }
  etsumMean_endc_out.close();
  

  // determine barrel calibration constants
  for (int ieta=0; ieta<kBarlRings; ieta++) {
    for (int iphi=0; iphi<kBarlWedges; iphi++) {
      for (int sign=0; sign<kSides; sign++) {
	
	
	// sc 
	int iphi_r = int(iphi/nscx);
      
      //if(nBads_barl_SM_[ieta][iphi_r][sign]>0){
      //  std::cout << "ETSUM" << etsum_barl_SM_[ieta][iphi_r][sign] << "  " <<ieta << " " << iphi_r << " " << sign << "  " << nBads_barl_SM_[ieta][iphi_r][sign]<< endl;
      //}      
		
	float epsilon_T_SM = 
	   etsum_barl_SM_[ieta][iphi_r][sign] /etsumMean_barl_SM_[ieta] -1.;
	
	epsilon_M_barl_SM_[ieta][iphi_r][sign] = epsilon_T_SM/k_barl_[ieta];
	
	if(e_.goodCell_barl[ieta][iphi][sign]){
	  float etsum = etsum_barl_[ieta][iphi][sign];
	  float epsilon_T = (etsum/etsumMean_barl_[ieta]) - 1.;
	  rawconst_barl[ieta][iphi][sign]  = epsilon_T + 1.;
	  epsilon_M_barl[ieta][iphi][sign] = epsilon_T/k_barl_[ieta];
	  

	} else {
	  rawconst_barl[ieta][iphi][sign]  = 1.;
	  epsilon_M_barl[ieta][iphi][sign] = 0.;
	} //if
      } //sign
    } //iphi
  } //ieta

    // determine endcap calibration constants
  for (int ix=0; ix<kEndcWedgesX; ix++) {
    for (int iy=0; iy<kEndcWedgesY; iy++) {
      for (int sign=0; sign<kSides; sign++) {
	int ring = e_.endcapRing_[ix][iy];
	if (ring!=-1 && e_.goodCell_endc[ix][iy][sign]) {
	  float etsum = etsum_endc_[ix][iy][sign];
	  float epsilon_T = (etsum/etsumMean_endc_[ring]) - 1.;
	  rawconst_endc[ix][iy][sign]  = epsilon_T + 1.;
	  epsilon_M_endc[ix][iy][sign] = epsilon_T/k_endc_[ring];	    
	} else {
	  epsilon_M_endc[ix][iy][0] = 0.;
	  epsilon_M_endc[ix][iy][1] = 0.;
	  rawconst_endc[ix][iy][0]  = 1.;
	  rawconst_endc[ix][iy][1]  = 1.;
	} //if
      } //sign
    } //iy
  } //ix



  // output sc calibration
  std::fstream scfile("sccalibration.dat",std::ios::out);
  for (int ieta =0; ieta< kBarlRings; ++ieta){
    for (int iphi_r =0; iphi_r< int(kBarlWedges/nscx);++iphi_r){
      for (int sign=0; sign<kSides; sign++) {
	scfile<< ieta << " " << iphi_r << " " <<sign << " " 
	      << 1/(1+epsilon_M_barl_SM_[ieta][iphi_r][sign] )<< std::endl;
      }
    }
  }

  std::string newcalibfile("EcalIntercalibConstants_new.xml");



  TFile ehistof("ehistos.root","recreate");  

  TH1D ebhisto("eb","eb",100, 0.,2.);

  std::vector<DetId>::const_iterator barrelIt=barrelCells.begin();
  for (; barrelIt!=barrelCells.end(); barrelIt++) {
    EBDetId eb(*barrelIt);
    int ieta = abs(eb.ieta())-1;
    int iphi = eb.iphi()-1;
    int sign = eb.zside()>0 ? 1 : 0;

    /// this is the new constant, or better, the correction to be applied
    /// to the old constant
    newCalibs_[eb] =  oldCalibs_[eb]/(1+epsilon_M_barl[ieta][iphi][sign]);

    if(e_.goodCell_barl[ieta][iphi][sign]){

      ebhisto.Fill(newCalibs_[eb]);
    
      /// residual miscalibraition  / expected precision
      miscal_resid_barl_histos[ieta]->Fill(miscalib_[eb]*newCalibs_[eb]);
      correl_barl_histos[ieta]->Fill(miscalib_[eb],newCalibs_[eb]);
	
    }
      
  }// barrelit

  TH1D eehisto("ee","ee",100, 0.,2.);
  std::vector<DetId>::const_iterator endcapIt=endcapCells.begin();

  for (; endcapIt!=endcapCells.end(); endcapIt++) {
    EEDetId ee(*endcapIt);
    int ix = ee.ix()-1;
    int iy = ee.iy()-1;
    int sign = ee.zside()>0 ? 1 : 0;
      
    newCalibs_[ee] = oldCalibs_[ee]/(1+epsilon_M_endc[ix][iy][sign]);
      
      
    if(e_.goodCell_endc[ix][iy][sign]){
 
      eehisto.Fill(newCalibs_[ee]);
      miscal_resid_endc_histos[e_.endcapRing_[ix][iy]]->Fill(miscalib_[ee]*
							     newCalibs_[ee]);;

      correl_endc_histos[e_.endcapRing_[ix][iy]]->Fill(miscalib_[ee],
						       newCalibs_[ee]);
   
    }
  }//endcapit
  // Write xml file
  EcalCondHeader header;
  header.method_="phi symmetry";
  header.version_="0";
  header.datasource_="testdata";
  header.since_=1;
  header.tag_="unknown";
  header.date_="Mar 24 1973";
 
  EcalIntercalibConstantsXMLTranslator::writeXML(newcalibfile,header,
						 newCalibs_ );  

  eehisto.Write();
  ebhisto.Write();
  ehistof.Close();

  fillConstantsHistos();
  
  outResidHistos();
  
}




void  PhiSymmetryCalibration_step2_SM::fillConstantsHistos(){
  
  TFile f("CalibHistos.root","recreate");  

  TH2F barreletamap("barreletamap","barreletamap",171, -85,86,100,0.,2.);
  TH2F barreletamapraw("barreletamapraw","barreletamapraw",171, -85,86,100,0.,2.);

  TH2F barrelmapold("barrelmapold","barrelmapold",360,1.,361.,171,-85.,86.);
  TH2F barrelmapnew("barrelmapnew","barrelmapnew",360,1.,361.,171,-85.,86.);
  TH2F barrelmapratio("barrelmapratio","barrelmapratio",360,1.,361.,171,-85.,86.);

  TH1F rawconst_endc_h("rawconst_endc","rawconst_endc",100,0.,2.);
  TH1F const_endc_h("const_endc","const_endc",100,0.,2.);

  TH1F oldconst_endc_h("oldconst_endc","oldconst_endc;oldCalib;",200,0,2);
  TH2F newvsraw_endc_h("newvsraw_endc","newvsraw_endc;rawConst;newCalib",200,0,2,200,0,2);

  TH2F endcapmapold_plus("endcapmapold_plus","endcapmapold_plus",100,1.,101.,100,1.,101.);
  TH2F endcapmapnew_plus("endcapmapnew_plus","endcapmapnew_plus",100,1.,101.,100,1.,101.);
  TH2F endcapmapratio_plus("endcapmapratio_plus","endcapmapratio_plus",100,1.,101.,100,1.,101.);

  TH2F endcapmapold_minus("endcapmapold_minus","endcapmapold_minus",100,1.,101.,100,1.,101.);
  TH2F endcapmapnew_minus("endcapmapnew_minus","endcapmapnew_minus",100,1.,101.,100,1.,101.);
  TH2F endcapmapratio_minus("endcapmapratio_minus","endcapmapratio_minus",100,1.,101.,100,1.,101.);


  for (int sign=0; sign<kSides; sign++) {

    int thesign = sign==1 ? 1:-1;

    for (int ieta=0; ieta<kBarlRings; ieta++) {
      for (int iphi=0; iphi<kBarlWedges; iphi++) {
	if(e_.goodCell_barl[ieta][iphi][sign]){

	  EBDetId eb(thesign*( ieta+1 ), iphi+1);
	  //int mod20= (iphi+1)%20;
	  //if (mod20==0 || mod20==1 ||mod20==2) continue;  // exclude SM boundaries
	  barreletamap.Fill(ieta*thesign + thesign,newCalibs_[eb]);
	  barreletamapraw.Fill(ieta*thesign + thesign,rawconst_barl[ieta][iphi][sign]);
	  
	  barrelmapold.Fill(iphi+1,ieta*thesign + thesign, oldCalibs_[eb]);
	  barrelmapnew.Fill(iphi+1,ieta*thesign + thesign, newCalibs_[eb]);
	  barrelmapratio.Fill(iphi+1,ieta*thesign + thesign, newCalibs_[eb]/oldCalibs_[eb]);
	}//if
      }//iphi
    }//ieta

    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	if (e_.goodCell_endc[ix][iy][sign]){
	  if (! EEDetId::validDetId(ix+1, iy+1,thesign)) continue;
	  EEDetId ee(ix+1, iy+1,thesign);

	  rawconst_endc_h.Fill(rawconst_endc[ix][iy][sign]);
	  const_endc_h.Fill(newCalibs_[ee]);
	  oldconst_endc_h.Fill(oldCalibs_[ee]);
	  newvsraw_endc_h.Fill(rawconst_endc[ix][iy][sign],newCalibs_[ee]);

	  if(sign==1){
	    endcapmapold_plus.Fill(ix+1,iy+1,oldCalibs_[ee]);
	    endcapmapnew_plus.Fill(ix+1,iy+1,newCalibs_[ee]);
	    endcapmapratio_plus.Fill(ix+1,iy+1,newCalibs_[ee]/oldCalibs_[ee]);
	  }
	  else{
	    endcapmapold_minus.Fill(ix+1,iy+1,oldCalibs_[ee]);
	    endcapmapnew_minus.Fill(ix+1,iy+1,newCalibs_[ee]);
	    endcapmapratio_minus.Fill(ix+1,iy+1,newCalibs_[ee]/oldCalibs_[ee]);
	  }

	}//if
      }//iy
    }//ix
    
  } // sides

  barreletamap.Write();
  barreletamapraw.Write();
  rawconst_endc_h.Write();
  const_endc_h.Write();
  oldconst_endc_h.Write();
  newvsraw_endc_h.Write();
  barrelmapold.Write();
  barrelmapnew.Write();
  barrelmapratio.Write();
  endcapmapold_plus.Write();
  endcapmapnew_plus.Write();
  endcapmapratio_plus.Write();
  endcapmapold_minus.Write();
  endcapmapnew_minus.Write();
  endcapmapratio_minus.Write();

  f.Close();
}









//_____________________________________________________________________________

void PhiSymmetryCalibration_step2_SM::fillHistos()
{
  TFile f("PhiSymmetryCalibration.root","recreate");

  std::vector<TH1F*> etsum_barl_histos(kBarlRings);
  std::vector<TH1F*> esum_barl_histos(kBarlRings);
  
  // determine ranges of ET sums to get histo bounds and book histos (barrel)
  for (int ieta=0; ieta<kBarlRings; ieta++) {
    float low=999999.;
    float high=0.;
    float low_e=999999.;
    float high_e=0.;

    for (int iphi=0; iphi<kBarlWedges; iphi++) {
      for (int sign=0; sign<kSides; sign++) {
	float etsum = etsum_barl_[ieta][iphi][sign];
	if (etsum<low && etsum!=0.) low=etsum;
	if (etsum>high) high=etsum;

	float esum = esum_barl_[ieta][iphi][sign];
	if (esum<low_e && esum!=0.) low_e=esum;
	if (esum>high_e) high_e=esum;
      }
    }
    
    ostringstream t;
    t << "etsum_barl_" << ieta+1;
    etsum_barl_histos[ieta]=new TH1F(t.str().c_str(),"",50,low-.2*low,high+.1*high);
    t.str("");

    t << "esum_barl_" << ieta+1;
    esum_barl_histos[ieta]=new TH1F(t.str().c_str(),"",50,low_e-.2*low_e,high_e+.1*high_e);
    t.str("");

    // fill barrel ET sum histos
    etsumMean_barl_[ieta]=0.;
    esumMean_barl_[ieta]=0.;
    
    for (int iphi=0; iphi<kBarlWedges; iphi++) {
      for (int sign=0; sign<kSides; sign++) {
	
	// mean for the SC
	int iphi_r = int(iphi/nscx);
	
	if( !(iphi%nscx)){ 
	 // bad channel correction
         etsum_barl_SM_[ieta][iphi_r][sign] = etsum_barl_SM_[ieta][iphi_r][sign]*nscx/(nscx- nBads_barl_SM_[ieta][iphi_r][sign]);      
//         std::cout << "ETSUM M " << ieta << " " << iphi_r << " " << 
//	     sign << " " << etsum_barl_SM_[ieta][iphi_r][sign] << "  " << nBads_barl_SM_[ieta][iphi_r][sign]<< endl;
	 etsumMean_barl_SM_[ieta] += 
	    etsum_barl_SM_[ieta][iphi_r][sign]/(2*int(kBarlWedges/nscx));
	}
	if(e_.goodCell_barl[ieta][iphi][sign]){
	  float etsum = etsum_barl_[ieta][iphi][sign];
	  float esum  = esum_barl_[ieta][iphi][sign];
	  etsum_barl_histos[ieta]->Fill(etsum);
	  esum_barl_histos[ieta]->Fill(esum);
	  etsumMean_barl_[ieta]+=etsum;


	  esumMean_barl_[ieta]+=esum;
	}
      }
    }

    etsum_barl_histos[ieta]->Write();
    esum_barl_histos[ieta]->Write();
    etsumMean_barl_[ieta]/=(720.-e_.nBads_barl[ieta]);
    esumMean_barl_[ieta]/=(720.-e_.nBads_barl[ieta]);
    
    delete etsum_barl_histos[ieta];
    delete esum_barl_histos[ieta]; //VS
  }

  std::vector<TH1F*> etsum_endc_histos(kEndcEtaRings);
  std::vector<TH1F*> etsum_endc_uncorr_histos(kEndcEtaRings);
  std::vector<TH1F*> esum_endc_histos(kEndcEtaRings);

  std::vector<TH2F*> etsumvsarea_endc_histos(kEndcEtaRings);
  std::vector<TH2F*> esumvsarea_endc_histos(kEndcEtaRings);

  // determine ranges of ET sums to get histo bounds and book histos (endcap)
  for (int ring=0; ring<kEndcEtaRings; ring++) {

    float low=FLT_MAX;
    float low_uncorr=FLT_MAX;
    float high=0.;
    float high_uncorr=0;
    float low_e=FLT_MAX;
    float high_e=0.;
    float low_a=1.;
    float high_a=0.;
    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	if (e_.endcapRing_[ix][iy]==ring) {
	  for (int sign=0; sign<kSides; sign++) {
	    float etsum = etsum_endc_[ix][iy][sign];
	    if (etsum<low && etsum!=0.) low=etsum;
	    if (etsum>high) high=etsum;

	    float etsum_uncorr = etsum_endc_uncorr[ix][iy][sign];
	    if (etsum_uncorr<low_uncorr && etsum_uncorr!=0.) low_uncorr=etsum_uncorr;
	    if (etsum_uncorr>high_uncorr) high_uncorr=etsum_uncorr;

	    float esum = esum_endc_[ix][iy][sign];
	    if (esum<low_e && esum!=0.) low_e=esum;
	    if (esum>high_e) high_e=esum;

	    float area = e_.cellArea_[ix][iy];
	    if (area<low_a) low_a=area;
	    if (area>high_a) high_a=area;
	  }
	}
      }
    }

    ostringstream t;
    t<<"etsum_endc_" << ring+1;
    etsum_endc_histos[ring]= new TH1F(t.str().c_str(),"",50,low-.2*low,high+.1*high);
    t.str("");

    t<<"etsum_endc_uncorr_" << ring+1;
    etsum_endc_uncorr_histos[ring]= new TH1F(t.str().c_str(),"",50,low_uncorr-.2*low_uncorr,high_uncorr+.1*high_uncorr);
    t.str("");

    t<<"esum_endc_" << ring+1;
    esum_endc_histos[ring]= new TH1F(t.str().c_str(),"",50,low_e-.2*low_e,high_e+.1*high_e);
    t.str("");

    t<<"etsumvsarea_endc_" << ring+1;
    etsumvsarea_endc_histos[ring]= new TH2F(t.str().c_str(),";A_{#eta#phi};#Sigma E_{T}",50,low_a,high_a,50,low,high);
    t.str("");

    t<<"esumvsarea_endc_" << ring+1;
    esumvsarea_endc_histos[ring]= new TH2F(t.str().c_str(),";A_{#eta#phi};#Sigma E",50,low_a,high_a,50,low_e,high_e);
    t.str("");

    // fill endcap ET sum histos
    etsumMean_endc_[ring]=0.;
    esumMean_endc_[ring]=0.;
    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	if (e_.endcapRing_[ix][iy]==ring) {
	  for (int sign=0; sign<kSides; sign++) {
	    if(e_.goodCell_endc[ix][iy][sign]){
	      float etsum = etsum_endc_[ix][iy][sign];
	      float esum  = esum_endc_[ix][iy][sign];
	      float etsum_uncorr = etsum_endc_uncorr[ix][iy][sign];
	      etsum_endc_histos[ring]->Fill(etsum);
	      etsum_endc_uncorr_histos[ring]->Fill(etsum_uncorr);
	      esum_endc_histos[ring]->Fill(esum);
	      
	      float area = e_.cellArea_[ix][iy];
	      etsumvsarea_endc_histos[ring]->Fill(area,etsum);
	      esumvsarea_endc_histos[ring]->Fill(area,esum);
	      
	      etsumMean_endc_[ring]+=etsum;
	      esumMean_endc_[ring]+=esum;
	    }
	  }
	}
      }
    }

    etsum_endc_histos[ring]->Write();
    etsum_endc_uncorr_histos[ring]->Write();
    esum_endc_histos[ring]->Write();
    etsumMean_endc_[ring]/=(float(e_.nRing_[ring]*2-e_.nBads_endc[ring]));
    esumMean_endc_[ring]/=(float(e_.nRing_[ring]*2-e_.nBads_endc[ring]));
    etsumvsarea_endc_histos[ring]->Write();
    esumvsarea_endc_histos[ring]->Write();

    delete etsum_endc_histos[ring];
    delete etsum_endc_uncorr_histos[ring];
    delete esum_endc_histos[ring];
    delete etsumvsarea_endc_histos[ring];
    delete esumvsarea_endc_histos[ring];
  }//ring


  // Maps of etsum in EB and EE
  TH2F barreletamap("barreletamap","barreletamap",171, -85,86,100,0,2);
  TH2F barrelmap("barrelmap","barrelmap - #frac{#Sigma E_{T}}{<#Sigma E_{T}>_{0}}",360,1,360, 171, -85,86);
  TH2F barrelmap_e("barrelmape","barrelmape - #frac{#Sigma E}{<#Sigma E>_{0}}",360,1,360, 171, -85,86);
  TH2F barrelmap_divided("barrelmapdiv","barrelmapdivided - #frac{#Sigma E_{T}}{hits}",360,1,360,171,-85,86);
  TH2F barrelmap_e_divided("barrelmapediv","barrelmapedivided - #frac{#Sigma E}{hits}",360,1,360,171,-85,86);
  TH2F endcmap_plus_corr("endcapmapplus_corrected","endcapmapplus - #frac{#Sigma E_{T}}{<#Sigma E_{T}>_{38}}",100,1,101,100,1,101);
  TH2F endcmap_minus_corr("endcapmapminus_corrected","endcapmapminus - #frac{#Sigma E_{T}}{<#Sigma E_{T}>_{38}}",100,1,101,100,1,101);
  TH2F endcmap_plus_uncorr("endcapmapplus_uncorrected","endcapmapplus_uncor - #frac{#Sigma E_{T}}{<#Sigma E_{T}>_{38}}",100,1,101,100,1,101);
  TH2F endcmap_minus_uncorr("endcapmapminus_uncorrected","endcapmapminus_uncor - #frac{#Sigma E_{T}}{<#Sigma E_{T}>_{38}}",100,1,101,100,1,101);
  TH2F endcmap_e_plus("endcapmapeplus","endcapmapeplus - #frac{#Sigma E}{<#Sigma E>_{38}}",100,1,101,100,1,101);
  TH2F endcmap_e_minus("endcapmapeminus","endcapmapeminus - #frac{#Sigma E}{<#Sigma E>_{38}}",100,1,101,100,1,101);

  for (int sign=0; sign<kSides; sign++) {

    int thesign = sign==1 ? 1:-1;

    for (int ieta=0; ieta<kBarlRings; ieta++) {
      for (int iphi=0; iphi<kBarlWedges; iphi++) {
	if(e_.goodCell_barl[ieta][iphi][sign]){
	  barrelmap.Fill(iphi+1,ieta*thesign + thesign, etsum_barl_[ieta][iphi][sign]/etsumMean_barl_[0]);
	  barrelmap_e.Fill(iphi+1,ieta*thesign + thesign, esum_barl_[ieta][iphi][sign]/esumMean_barl_[0]); //VS
	  if (!nhits_barl_[ieta][iphi][sign]) nhits_barl_[ieta][iphi][sign] =1;
	  barrelmap_divided.Fill( iphi+1,ieta*thesign + thesign, etsum_barl_[ieta][iphi][sign]/nhits_barl_[ieta][iphi][sign]);
	  barrelmap_e_divided.Fill( iphi+1,ieta*thesign + thesign, esum_barl_[ieta][iphi][sign]/nhits_barl_[ieta][iphi][sign]); //VS
	  //int mod20= (iphi+1)%20;
	  //if (mod20==0 || mod20==1 ||mod20==2) continue;  // exclude SM boundaries
	  barreletamap.Fill(ieta*thesign + thesign,etsum_barl_[ieta][iphi][sign]/etsumMean_barl_[0]);
	}//if
      }//iphi
    }//ieta

    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	if (sign==1) {
	  endcmap_plus_corr.Fill(ix+1,iy+1,etsum_endc_[ix][iy][sign]/etsumMean_endc_[38]);
	  endcmap_plus_uncorr.Fill(ix+1,iy+1,etsum_endc_uncorr[ix][iy][sign]/etsumMean_endc_[38]);
	  endcmap_e_plus.Fill(ix+1,iy+1,esum_endc_[ix][iy][sign]/esumMean_endc_[38]);
	}
	else{ 
	  endcmap_minus_corr.Fill(ix+1,iy+1,etsum_endc_[ix][iy][sign]/etsumMean_endc_[38]);
	  endcmap_minus_uncorr.Fill(ix+1,iy+1,etsum_endc_uncorr[ix][iy][sign]/etsumMean_endc_[38]);
	  endcmap_e_minus.Fill(ix+1,iy+1,esum_endc_[ix][iy][sign]/esumMean_endc_[38]);
	}
      }//iy
    }//ix

  }  //sign
  
 

  barreletamap.Write();
  barrelmap_divided.Write();
  barrelmap.Write();
  barrelmap_e_divided.Write();
  barrelmap_e.Write();
  endcmap_plus_corr.Write();
  endcmap_minus_corr.Write();
  endcmap_plus_uncorr.Write();
  endcmap_minus_uncorr.Write();
  endcmap_e_plus.Write();
  endcmap_e_minus.Write();


  vector<TH1F*> etavsphi_endc(kEndcEtaRings);
  vector<TH1F*> areavsphi_endc(kEndcEtaRings);
  vector<TH1F*> etsumvsphi_endcp_corr(kEndcEtaRings);
  vector<TH1F*> etsumvsphi_endcm_corr(kEndcEtaRings);
  vector<TH1F*> etsumvsphi_endcp_uncorr(kEndcEtaRings);
  vector<TH1F*> etsumvsphi_endcm_uncorr(kEndcEtaRings);
  vector<TH1F*> esumvsphi_endcp(kEndcEtaRings);
  vector<TH1F*> esumvsphi_endcm(kEndcEtaRings);

  std::vector<TH1F*> deltaeta_histos(kEndcEtaRings);
  std::vector<TH1F*> deltaphi_histos(kEndcEtaRings);

  for(int ring =0; ring<kEndcEtaRings;++ring){
    
    ostringstream t;
    t<< "etavsphi_endc_"<<ring;
    etavsphi_endc[ring] = new TH1F(t.str().c_str(), t.str().c_str(),e_.nRing_[ring],0,e_.nRing_[ring]);
    t.str("");

    t<< "areavsphi_endc_"<<ring;
    areavsphi_endc[ring] = new TH1F(t.str().c_str(), t.str().c_str(),e_.nRing_[ring],0,e_.nRing_[ring]);
    t.str("");

    t<< "etsumvsphi_endcp_corr_"<<ring;
    etsumvsphi_endcp_corr[ring] = new TH1F(t.str().c_str(), t.str().c_str(),e_.nRing_[ring],0,e_.nRing_[ring]);
    t.str("");

    t << "etsumvsphi_endcm_corr_"<<ring;
    etsumvsphi_endcm_corr[ring] = new TH1F(t.str().c_str(), t.str().c_str(),e_.nRing_[ring],0,e_.nRing_[ring]);
    t.str("");

    t << "etsumvsphi_endcp_uncorr_"<<ring;
    etsumvsphi_endcp_uncorr[ring] = new TH1F(t.str().c_str(), t.str().c_str(),e_.nRing_[ring],0,e_.nRing_[ring]);
    t.str("");

    t << "etsumvsphi_endcm_uncorr_"<<ring;
    etsumvsphi_endcm_uncorr[ring] = new TH1F(t.str().c_str(), t.str().c_str(),e_.nRing_[ring],0,e_.nRing_[ring]);
    t.str("");
    
    t << "esumvsphi_endcp_"<<ring;
    esumvsphi_endcp[ring] = new TH1F(t.str().c_str(), t.str().c_str(),e_.nRing_[ring],0,e_.nRing_[ring]);
    t.str("");

    t << "esumvsphi_endcm_"<<ring;
    esumvsphi_endcm[ring] = new TH1F(t.str().c_str(), t.str().c_str(),e_.nRing_[ring],0,e_.nRing_[ring]);
    t.str("");

    t << "deltaeta_" << ring;
    deltaeta_histos[ring]= new TH1F(t.str().c_str(),"",50,-.1,.1);
    t.str("");
    t << "deltaphi_" << ring;
    deltaphi_histos[ring]= new TH1F(t.str().c_str(),"",50,-.1,.1);
    t.str("");
  }

  for (int ix=0; ix<kEndcWedgesX; ix++) {
    for (int iy=0; iy<kEndcWedgesY; iy++) {

      int ring = e_.endcapRing_[ix][iy];
      if (ring!=-1) {
	int iphi_endc=-1;
	for (int ip=0; ip<e_.nRing_[ring]; ip++) {
	  if (e_.cellPhi_[ix][iy]==e_.phi_endc_[ip][ring]) iphi_endc=ip;
	}

	if(iphi_endc!=-1){
	  for (int sign=0; sign<kSides; sign++) {
	    if(e_.goodCell_endc[ix][iy][sign]){
	      if (sign==1){
		etsumvsphi_endcp_corr[ring]->Fill(iphi_endc,etsum_endc_[ix][iy][sign]);
		etsumvsphi_endcp_uncorr[ring]->Fill(iphi_endc,etsum_endc_uncorr[ix][iy][sign]);
		esumvsphi_endcp[ring]->Fill(iphi_endc,esum_endc_[ix][iy][sign]);
	      } else {
		etsumvsphi_endcm_corr[ring]->Fill(iphi_endc,etsum_endc_[ix][iy][sign]);
		etsumvsphi_endcm_uncorr[ring]->Fill(iphi_endc,etsum_endc_uncorr[ix][iy][sign]);
		esumvsphi_endcm[ring]->Fill(iphi_endc,esum_endc_[ix][iy][sign]);
	      }
	    }//if
	  }//sign
	  etavsphi_endc[ring]->Fill(iphi_endc,e_.cellPos_[ix][iy].eta());
	  areavsphi_endc[ring]->Fill(iphi_endc,e_.cellArea_[ix][iy]);
	} //if iphi_endc

      }//if ring
    }//iy
  } //ix



  for(int ring =0; ring<kEndcEtaRings;++ring){
    
    etavsphi_endc[ring]->Write();
    areavsphi_endc[ring]->Write();
    etsumvsphi_endcp_corr[ring]->Write();
    etsumvsphi_endcm_corr[ring]->Write();
    etsumvsphi_endcp_uncorr[ring]->Write();
    etsumvsphi_endcm_uncorr[ring]->Write();
    esumvsphi_endcp[ring]->Write();
    esumvsphi_endcm[ring]->Write();
    deltaeta_histos[ring]->Write();
    deltaphi_histos[ring]->Write();

    
    delete etsumvsphi_endcp_corr[ring];
    delete etsumvsphi_endcm_corr[ring];
    delete etsumvsphi_endcp_uncorr[ring];
    delete etsumvsphi_endcm_uncorr[ring];
    delete etavsphi_endc[ring];
    delete areavsphi_endc[ring];
    delete esumvsphi_endcp[ring];
    delete esumvsphi_endcm[ring];
    delete deltaeta_histos[ring];
    delete deltaphi_histos[ring];
  }


  f.Close();
}


void PhiSymmetryCalibration_step2_SM::readEtSums(){


  //read in ET sums
  
  int ieta,iphi,sign,ix,iy,dummy;
  double etsum;
  unsigned int nhits;
  std::ifstream etsum_barl_in("etsum_barl.dat", ios::in);
  while ( etsum_barl_in >> dummy >> ieta >> iphi >> sign >> etsum >> nhits ) {
    etsum_barl_[ieta][iphi][sign]+=etsum;
    nhits_barl_[ieta][iphi][sign]+=nhits;
    
    // fill etsums for the SM calibration
    int iphi_r = int(iphi/nscx);
    etsum_barl_SM_[ieta][iphi_r][sign]+= etsum;
//      etsum*nscx/(nscx- nBads_barl_SM_[ieta][iphi_r][sign]);
 //     if(nBads_barl_SM_[ieta][iphi_r][sign]>0){
 //       std::cout << "ETSUM" << etsum_barl_SM_[ieta][iphi_r][sign] << "  " << nscx << "  " << nBads_barl_SM_[ieta][iphi_r][sign]<< endl;
 //     }      
  }
 
  std::ifstream etsum_endc_in("etsum_endc.dat", ios::in);
  while ( etsum_endc_in >> dummy >> ix >> iy >> sign >> etsum >> nhits>>dummy ) {
    etsum_endc_[ix][iy][sign]+=etsum;
    nhits_endc_[ix][iy][sign]+=nhits;

  }
 
  std::ifstream k_barl_in("k_barl.dat", ios::in);
  for (int ieta=0; ieta<kBarlRings; ieta++) {
    k_barl_in >> dummy >> k_barl_[ieta];
  }
 
  std::ifstream k_endc_in("k_endc.dat", ios::in);
  for (int ring=0; ring<kEndcEtaRings; ring++) {
    k_endc_in >> dummy >> k_endc_[ring];
  }
  

}



void PhiSymmetryCalibration_step2_SM::setupResidHistos(){

  miscal_resid_barl_histos.resize(kBarlRings);
  correl_barl_histos.resize(kBarlRings);  
 

  for (int ieta=0; ieta<kBarlRings; ieta++) {
    ostringstream t1; 
    t1<<"mr_barl_"<<ieta+1;
    miscal_resid_barl_histos[ieta] = new TH1F(t1.str().c_str(),"",100,0.,2.);
    ostringstream t2;
    t2<<"co_barl_"<<ieta+1;
    correl_barl_histos[ieta] = new TH2F(t2.str().c_str(),"",50,.5,1.5,50,.5,1.5);
  }

  miscal_resid_endc_histos.resize(kEndcEtaRings);
  correl_endc_histos.resize(kEndcEtaRings);


  for (int ring=0; ring<kEndcEtaRings; ring++) {
    ostringstream t1;
    t1<<"mr_endc_"<< ring+1;
    miscal_resid_endc_histos[ring] = new TH1F(t1.str().c_str(),"",100,0.,2.);
    ostringstream t2;
    t2<<"co_endc_"<<ring+1;
    correl_endc_histos[ring] = new TH2F(t2.str().c_str(),"",50,.5,1.5,50,.5,1.5);
  }

 

  

}


void  PhiSymmetryCalibration_step2_SM::outResidHistos(){

  // output histograms of residual miscalibrations
  TFile f("PhiSymmetryCalibration_miscal_resid.root","recreate");
  for (int ieta=0; ieta<85; ieta++) {
    miscal_resid_barl_histos[ieta]->Write();
    correl_barl_histos[ieta]->Write();

    delete miscal_resid_barl_histos[ieta];
    delete correl_barl_histos[ieta];
  }
  
  for (int ring=0; ring<39; ring++) {
    miscal_resid_endc_histos[ring]->Write();
    correl_endc_histos[ring]->Write();
    
    delete  miscal_resid_endc_histos[ring];
    delete  correl_endc_histos[ring];

  }
  f.Close(); 
}
