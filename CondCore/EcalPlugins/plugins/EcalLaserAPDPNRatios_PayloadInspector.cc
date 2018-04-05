#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <memory>
#include <sstream>



namespace {
  enum {kEBChannels = 61200, kEEChannels = 14648};
  enum {MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360, EBhistEtaMax = 171};   // barrel lower and upper bounds on eta and phi
  enum {IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100, EEhistXMax = 220};           // endcaps lower and upper bounds on x and y

  /*******************************************************
   
     2d histogram of ECAL barrel APDPNRatios of 1 IOV 

  *******************************************************/


  // inherit from one of the predefined plot class: Histogram2D
  class EcalLaserAPDPNRatiosEBMap : public cond::payloadInspector::Histogram2D<EcalLaserAPDPNRatios> {

  public:
    EcalLaserAPDPNRatiosEBMap() : cond::payloadInspector::Histogram2D<EcalLaserAPDPNRatios>( "ECAL Barrel APDPNRatios - map ",
											     "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI + 1,
											     "ieta", EBhistEtaMax, -MAX_IETA, MAX_IETA + 1) {
    Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      for (auto const & iov: iovs) {
	std::shared_ptr<EcalLaserAPDPNRatios> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // set to -1 for ieta 0 (no crystal)
	  for(int iphi = 1; iphi < 361; iphi++) fillWithValue(iphi, 0, -1);

	  for(int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    float p2 = (payload->getLaserMap())[rawid].p2;
	    // fill the Histogram2D here
	    fillWithValue((EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), p2);
	  } // loop over cellid
	}  // if payload.get()
      }   // loop over IOV's (1 in this case)

      return true;
    }// fill method
  };

  class EcalLaserAPDPNRatiosEEMap : public cond::payloadInspector::Histogram2D<EcalLaserAPDPNRatios> {

  private:
    int EEhistSplit = 20;

  public:
    EcalLaserAPDPNRatiosEEMap() : cond::payloadInspector::Histogram2D<EcalLaserAPDPNRatios>( "ECAL Endcap APDPNRatios - map ",
											     "ix", EEhistXMax, IX_MIN, EEhistXMax + 1, 
											     "iy", IY_MAX, IY_MIN, IY_MAX + 1) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      for (auto const & iov: iovs) {
	std::shared_ptr<EcalLaserAPDPNRatios> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){

	  // set to -1 everywhwere
	  for(int ix = IX_MIN; ix < EEhistXMax + 1; ix++)
	    for(int iy = IY_MAX; iy < IY_MAX + 1; iy++)
	      fillWithValue(ix, iy, -1);

	  for(int cellid = 0; cellid < EEDetId::kSizeForDenseIndexing; ++cellid) {
	    if(!EEDetId::validHashIndex(cellid)) continue;
	    uint32_t rawid = EEDetId::unhashIndex(cellid);
	    float p2 = (payload->getLaserMap())[rawid].p2;
	    EEDetId myEEId(rawid);
	    if(myEEId.zside() == -1)
	      fillWithValue(myEEId.ix(), myEEId.iy(), p2);
	    else
	      fillWithValue(myEEId.ix() + IX_MAX + EEhistSplit, myEEId.iy(), p2);
	  }   // loop over cellid
	}    // payload
      }     // loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  /*************************************************
     2d plot of ECAL IntercalibConstants of 1 IOV
  *************************************************/
  class EcalLaserAPDPNRatiosPlot : public cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios> {

  public:
    EcalLaserAPDPNRatiosPlot() : cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios>("ECAL Laser APDPNRatios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      TH2F** barrel = new TH2F*[3];
      TH2F** endc_p = new TH2F*[3];
      TH2F** endc_m = new TH2F*[3];
      float pEBmin[3], pEEmin[3];

      for (int i = 0; i < 3; i++) {
	barrel[i] = new TH2F(Form("EBp%i", i),Form("EB p%i", i+1),MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
	endc_p[i] = new TH2F(Form("EE+p%i",i),Form("EE+ p%i",i+1),IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
	endc_m[i] = new TH2F(Form("EE-p%i",i),Form("EE- p%i",i+1),IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
	pEBmin[i] = 10.;
	pEEmin[i] = 10.;
      }

      auto iov = iovs.front();
      std::shared_ptr<EcalLaserAPDPNRatios> payload = fetchPayload( std::get<1>(iov) );
      unsigned long IOV = std::get<0>(iov);
      int run = 0;
      if(IOV < 4294967296) run = std::get<0>(iov);
      else   // time type IOV
	run = IOV >> 32;
      if( payload.get() ){
	// looping over the EB channels, via the dense-index, mapped into EBDetId's
	for(int cellid = 0; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {    // loop on EB cells
	  uint32_t rawid = EBDetId::unhashIndex(cellid);  
	  Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
	  Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
	  if(eta > 0.) eta = eta - 0.5;   //   0.5 to 84.5
	  else eta  = eta + 0.5;         //  -84.5 to -0.5
	  float p1 = (payload->getLaserMap())[rawid].p1;
	  if(p1 < pEBmin[0]) pEBmin[0] = p1;
	  float p2 = (payload->getLaserMap())[rawid].p2;
	  if(p2 < pEBmin[1]) pEBmin[1] = p2;
	  float p3 = (payload->getLaserMap())[rawid].p3;
	  if(p3 < pEBmin[2]) pEBmin[2] = p3;
	  barrel[0]->Fill(phi, eta, p1);
	  barrel[1]->Fill(phi, eta, p2);
	  barrel[2]->Fill(phi, eta, p3);
	}  // loop over cellid

	// looping over the EE channels
	for(int cellid = 0; cellid < EEDetId::kSizeForDenseIndexing; ++cellid) {
	  if(!EEDetId::validHashIndex(cellid)) continue;
	  uint32_t rawid = EEDetId::unhashIndex(cellid);
	  EEDetId myEEId(rawid);
	  float p1 = (payload->getLaserMap())[rawid].p1;
	  if(p1 < pEEmin[0]) pEEmin[0] = p1;
	  float p2 = (payload->getLaserMap())[rawid].p2;
	  if(p2 < pEEmin[1]) pEEmin[1] = p2;
	  float p3 = (payload->getLaserMap())[rawid].p3;
	  if(p3 < pEEmin[2]) pEEmin[2] = p3;
	  if (myEEId.zside() == 1) {
	    endc_p[0]->Fill(myEEId.ix(), myEEId.iy(), p1);
	    endc_p[1]->Fill(myEEId.ix(), myEEId.iy(), p2);
	    endc_p[2]->Fill(myEEId.ix(), myEEId.iy(), p3);
	  }
	  else { 
	    endc_m[0]->Fill(myEEId.ix(), myEEId.iy(), p1);
	    endc_m[1]->Fill(myEEId.ix(), myEEId.iy(), p2);
	    endc_m[2]->Fill(myEEId.ix(), myEEId.iy(), p3);
	  }
	}  // validDetId 
      }   // if payload.get()
      else return false;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      TCanvas canvas("CC map","CC map",2800,2600);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      if(IOV < 4294967296)
	t1.DrawLatex(0.5, 0.96, Form("Ecal Laser APD/PN, IOV %i", run));
      else {   // time type IOV
	time_t t = run;
	char buf[256];
	struct tm lt;
	localtime_r(&t, &lt);
	strftime(buf, sizeof(buf), "%F %R:%S", &lt);
	buf[sizeof(buf)-1] = 0;
	t1.DrawLatex(0.5, 0.96, Form("Ecal Laser APD/PN, IOV %s", buf));
      }

      float xmi[3] = {0.0 , 0.26, 0.74};
      float xma[3] = {0.26, 0.74, 1.00};
      TPad*** pad = new TPad**[3];
      for (int i = 0; i < 3; i++) {
	pad[i] = new TPad*[3];
	for (int obj = 0; obj < 3; obj++) {
	  float yma = 0.94 - (0.32 * i);
	  float ymi = yma - 0.28;
	  pad[i][obj] = new TPad(Form("p_%i_%i", obj, i),Form("p_%i_%i", obj, i),
				   xmi[obj], ymi, xma[obj], yma);
	  pad[i][obj]->Draw();
	}
      }

      for (int i = 0; i < 3; i++) {
	// compute histo limits with some rounding
	//	std::cout << " before " << pEBmin[i];
	float xmin = pEBmin[i] * 10.;
	int min = (int)xmin;
	pEBmin[i] = (float)min /10.;
	//	std::cout << " after " << pEBmin[i] << std::endl << " before " << pEEmin[i];
	xmin = pEEmin[i] * 10.;
	min = (int)xmin;
	pEEmin[i] = (float)min /10.;
	//	std::cout << " after " << pEEmin[i] << std::endl;
	pad[i][0]->cd();
	DrawEE(endc_m[i], pEEmin[i], 1.1);
	pad[i][1]->cd();
	DrawEB(barrel[i], pEBmin[i], 1.1);
	pad[i][2]->cd();
	DrawEE(endc_p[i], pEEmin[i], 1.1);
      }

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }// fill method
  };

  /*****************************************************************
     2d plot of ECAL IntercalibConstants difference between 2 IOVs
  ******************************************************************/
  class EcalLaserAPDPNRatiosDiff : public cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios> {

  public:
    EcalLaserAPDPNRatiosDiff() : cond::payloadInspector::PlotImage<EcalLaserAPDPNRatios>("ECAL Laser APDPNRatios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      TH2F** barrel = new TH2F*[3];
      TH2F** endc_p = new TH2F*[3];
      TH2F** endc_m = new TH2F*[3];
      float pEBmin[3], pEEmin[3], pEBmax[3], pEEmax[3];


      for (int i = 0; i < 3; i++) {
	barrel[i] = new TH2F(Form("EBp%i", i),Form("EB p%i", i+1),MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
	endc_p[i] = new TH2F(Form("EE+p%i",i),Form("EE+ p%i",i+1),IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
	endc_m[i] = new TH2F(Form("EE-p%i",i),Form("EE- p%i",i+1),IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
	pEBmin[i] = 10.;
	pEEmin[i] = 10.;
	pEBmax[i] = -10.;
	pEEmax[i] = -10.;
      }
      unsigned int run[2] = {0, 0}, irun = 0;
      unsigned long IOV = 0;
      float pEB[3][kEBChannels], pEE[3][kEEChannels];
      for ( auto const & iov: iovs) {
	std::shared_ptr<EcalLaserAPDPNRatios> payload = fetchPayload( std::get<1>(iov) );
	IOV = std::get<0>(iov);
	if(IOV < 4294967296) run[irun] = std::get<0>(iov);
	else   // time type IOV
	  run[irun] = IOV >> 32;
	if( payload.get() ){
	// looping over the EB channels, via the dense-index, mapped into EBDetId's
	  for(int cellid = 0; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {    // loop on EB cells
	    uint32_t rawid = EBDetId::unhashIndex(cellid);  
	    if(irun == 0) {
	      pEB[0][cellid] = (payload->getLaserMap())[rawid].p1;
	      pEB[1][cellid] = (payload->getLaserMap())[rawid].p2;
	      pEB[2][cellid] = (payload->getLaserMap())[rawid].p3;
	    }
	    else {
	      Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
	      Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
	      if(eta > 0.) eta = eta - 0.5;   //   0.5 to 84.5
	      else eta  = eta + 0.5;         //  -84.5 to -0.5
	      double diff = (payload->getLaserMap())[rawid].p1 - pEB[0][cellid];
	      if(diff < pEBmin[0]) pEBmin[0] = diff;
	      if(diff > pEBmax[0]) pEBmax[0] = diff;
	      barrel[0]->Fill(phi, eta, diff);
	      diff = (payload->getLaserMap())[rawid].p2 - pEB[1][cellid];
	      if(diff < pEBmin[1]) pEBmin[1] = diff;
	      if(diff > pEBmax[1]) pEBmax[1] = diff;
	      barrel[1]->Fill(phi, eta, diff);
	      diff = (payload->getLaserMap())[rawid].p3 - pEB[2][cellid];
	      if(diff < pEBmin[2]) pEBmin[2] = diff;
	      if(diff > pEBmax[2]) pEBmax[2] = diff;
	      barrel[2]->Fill(phi, eta, diff);
	    }
	  }  // loop over cellid

	  // looping over the EE channels
	  for(int cellid = 0; cellid < EEDetId::kSizeForDenseIndexing; ++cellid) {
	    if(!EEDetId::validHashIndex(cellid)) continue;
	    uint32_t rawid = EEDetId::unhashIndex(cellid);
	    EEDetId myEEId(rawid);
	    if(irun == 0) {
	      pEE[0][cellid] = (payload->getLaserMap())[rawid].p1;
	      pEE[1][cellid] = (payload->getLaserMap())[rawid].p2;
	      pEE[2][cellid] = (payload->getLaserMap())[rawid].p3;
	    }
	    else {
	      double diff1 = (payload->getLaserMap())[rawid].p1 - pEE[0][cellid];
	      if(diff1 < pEEmin[0]) pEEmin[0] = diff1;
	      if(diff1 > pEEmax[0]) pEEmax[0] = diff1;
	      double diff2 = (payload->getLaserMap())[rawid].p2 - pEE[1][cellid];
	      if(diff2 < pEEmin[1]) pEEmin[1] = diff2;
	      if(diff2 > pEEmax[1]) pEEmax[1] = diff2;
	      double diff3 = (payload->getLaserMap())[rawid].p3 - pEE[2][cellid];
	      if(diff3 < pEEmin[2]) pEEmin[2] = diff3;
	      if(diff3 > pEEmax[2]) pEEmax[2] = diff3;
	      if (myEEId.zside() == 1) {
		endc_p[0]->Fill(myEEId.ix(), myEEId.iy(), diff1);
		endc_p[1]->Fill(myEEId.ix(), myEEId.iy(), diff2);
		endc_p[2]->Fill(myEEId.ix(), myEEId.iy(), diff3);
	      }
	      else { 
		endc_m[0]->Fill(myEEId.ix(), myEEId.iy(), diff1);
		endc_m[1]->Fill(myEEId.ix(), myEEId.iy(), diff2);
		endc_m[2]->Fill(myEEId.ix(), myEEId.iy(), diff3);
	      }
	    }
	  }   // loop over cellid
	}  //  if payload.get()
	else return false;
	irun++;
      }      // loop over IOVs

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      TCanvas canvas("CC map","CC map",2800,2600);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      if(IOV < 4294967296) {
	t1.SetTextSize(0.05);
	t1.DrawLatex(0.5, 0.96, Form("Ecal Laser APD/PN, IOV %i - %i", run[1], run[0]));
      }
      else {   // time type IOV
	time_t t = run[0];
	char buf0[256], buf1[256];
	struct tm lt;
	localtime_r(&t, &lt);
	strftime(buf0, sizeof(buf0), "%F %R:%S", &lt);
	buf0[sizeof(buf0)-1] = 0;
	t = run[1];
	localtime_r(&t, &lt);
	strftime(buf1, sizeof(buf1), "%F %R:%S", &lt);
	buf1[sizeof(buf1)-1] = 0;
	t1.SetTextSize(0.015);
	t1.DrawLatex(0.5, 0.96, Form("Ecal Laser APD/PN, IOV %s - %s", buf1, buf0));
      }

      float xmi[3] = {0.0 , 0.24, 0.76};
      float xma[3] = {0.24, 0.76, 1.00};
      TPad*** pad = new TPad**[3];
      for (int i = 0; i < 3; i++) {
	pad[i] = new TPad*[3];
	for (int obj = 0; obj < 3; obj++) {
	  float yma = 0.94 - (0.32 * i);
	  float ymi = yma - 0.28;
	  pad[i][obj] = new TPad(Form("p_%i_%i", obj, i),Form("p_%i_%i", obj, i),
				   xmi[obj], ymi, xma[obj], yma);
	  pad[i][obj]->Draw();
	}
      }

     for (int i = 0; i < 3; i++) {
	// compute histo limits with some rounding
       //       std::cout << " before min " << pEBmin[i] << " max " << pEBmax[i];
       float xmin = (pEBmin[i] - 0.009) * 100.;
       int min = (int)xmin;
       pEBmin[i] = (float)min /100.;
       float xmax = (pEBmax[i] + 0.009) * 100.;
       int max = (int)xmax;
       pEBmax[i] = (float)max /100.;
       //       std::cout << " after min " << pEBmin[i] << " max " << pEBmax[i] << std::endl << " before min " << pEEmin[i] << " max " << pEEmax[i];
       xmin = (pEEmin[i] + 0.009) * 100.;
       min = (int)xmin;
       pEEmin[i] = (float)min /100.;
       xmax = (pEEmax[i] + 0.009) * 100.;
       max = (int)xmax;
       pEEmax[i] = (float)max /100.;
       //       std::cout << " after min " << pEEmin[i]  << " max " << pEEmax[i]<< std::endl;
       pad[i][0]->cd();
       DrawEE(endc_m[i], pEEmin[i], pEEmax[i]);
       endc_m[i]->GetZaxis()->SetLabelSize(0.02);
       pad[i][1]->cd();
       DrawEB(barrel[i], pEBmin[i], pEBmax[i]);
       pad[i][2]->cd();
       DrawEE(endc_p[i], pEEmin[i], pEEmax[i]);
       endc_p[i]->GetZaxis()->SetLabelSize(0.02);
     }

     std::string ImageName(m_imageFileName);
     canvas.SaveAs(ImageName.c_str());
     return true;
    }// fill method
  };

} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalLaserAPDPNRatios){
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEBMap);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosEEMap);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalLaserAPDPNRatiosDiff);
}
