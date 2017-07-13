#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <memory>
#include <sstream>



namespace {
  enum {kEBChannels = 61200, kEEChannels = 14648};
  enum {MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360};   // barrel lower and upper bounds on eta and phi
  enum {IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100};         // endcaps lower and upper bounds on x and y

  /*******************************************************
   
     2d histogram of ECAL barrel Intercalib Constants of 1 IOV 

  *******************************************************/

  // inherit from one of the predefined plot class: Histogram2D
  class EcalIntercalibConstantsEBMap : public cond::payloadInspector::Histogram2D<EcalIntercalibConstants> {

  private:
    int EBhistEtaMax = 2 * EBDetId::MAX_IETA + 1;

  public:
    EcalIntercalibConstantsEBMap() : cond::payloadInspector::Histogram2D<EcalIntercalibConstants>("ECAL Barrel Intercalib Constants - map ",
												  "iphi", EBDetId::MAX_IPHI, EBDetId::MIN_IPHI, EBDetId::MAX_IPHI + 1,
												  "ieta", EBhistEtaMax, -EBDetId::MAX_IETA, EBDetId::MAX_IETA + 1) {
      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for (auto const & iov: iovs) {
	std::shared_ptr<EcalIntercalibConstants> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  // set to -1 for ieta 0 (no crystal)
	  for(int iphi = EBDetId::MIN_IPHI; iphi < EBDetId::MAX_IPHI+1; iphi++) fillWithValue(iphi, 0, -1);

	  for(int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);

	    // check the existence of ECAL Intercalib Constants, for a given ECAL barrel channel
	    EcalFloatCondObjectContainer::const_iterator value_ptr =  payload->find(rawid);
	    if (value_ptr == payload->end()) continue; // cell absent from payload
	    float weight = (float)(*value_ptr);

	    // fill the Histogram2D here
	    fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta(), weight);
	  }// loop over cellid
	}// if payload.get()
      }// loop over IOV's (1 in this case)

      return true;
    }// fill method
  };

  class EcalIntercalibConstantsEEMap : public cond::payloadInspector::Histogram2D<EcalIntercalibConstants> {

  private:
    int EEhistSplit = 20;
    int EEhistXMax = 2 * EEDetId::IX_MAX +  EEhistSplit;

  public:
    EcalIntercalibConstantsEEMap() : cond::payloadInspector::Histogram2D<EcalIntercalibConstants>( "ECAL Endcap Intercalib Constants - map ",
												   "ix", EEhistXMax, EEDetId::IX_MIN, EEhistXMax + 1, 
												   "iy", EEDetId::IY_MAX, EEDetId::IY_MIN, EEDetId::IY_MAX + 1) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for (auto const & iov: iovs) {
	std::shared_ptr<EcalIntercalibConstants> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // set to -1 everywhwere
	  for(int ix = EEDetId::IX_MIN; ix < EEhistXMax + 1; ix++)
	    for(int iy = EEDetId::IY_MAX; iy < EEDetId::IY_MAX + 1; iy++)
	      fillWithValue(ix, iy, -1);

	  for (int cellid = 0;  cellid < EEDetId::kSizeForDenseIndexing; ++cellid){    // loop on EE cells
	    if (EEDetId::validHashIndex(cellid)){  
	      uint32_t rawid = EEDetId::unhashIndex(cellid);
	      EcalFloatCondObjectContainer::const_iterator value_ptr =  payload->find(rawid);
	      if (value_ptr == payload->end()) continue; // cell absent from payload
	      float weight = (float)(*value_ptr);
	      EEDetId myEEId(rawid);
	      if(myEEId.zside() == -1)
		fillWithValue(myEEId.ix(), myEEId.iy(), weight);
	      else
		fillWithValue(myEEId.ix() + EEDetId::IX_MAX + EEhistSplit, myEEId.iy(), weight);
	    }  // validDetId 
	  }   // loop over cellid
	}    // payload
      }     // loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  /*************************************************
     2d plot of ECAL IntercalibConstants of 1 IOV
  *************************************************/
  class EcalIntercalibConstantsPlot : public cond::payloadInspector::PlotImage<EcalIntercalibConstants> {

  public:
    EcalIntercalibConstantsPlot() : cond::payloadInspector::PlotImage<EcalIntercalibConstants>("ECAL Intercalib Constants - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      TH2F* barrel = new TH2F("EB", "mean EB", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p = new TH2F("EE+", "mean EE+", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m = new TH2F("EE-", "mean EE-", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);

      auto iov = iovs.front();
      std::shared_ptr<EcalIntercalibConstants> payload = fetchPayload( std::get<1>(iov) );
      unsigned int run = std::get<0>(iov);
      if( payload.get() ){
	if (!payload->barrelItems().size()) return false;
	for(int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
	  uint32_t rawid = EBDetId::unhashIndex(cellid);
	  EcalFloatCondObjectContainer::const_iterator value_ptr =  payload->find(rawid);
	  if (value_ptr == payload->end()) continue; // cell absent from payload
	  float weight = (float)(*value_ptr);
	  Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
	  Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
	  if(eta > 0.) eta = eta - 0.5;   //   0.5 to 84.5
	  else eta  = eta + 0.5;         //  -84.5 to -0.5
	  barrel->Fill(phi, eta, weight);
	}// loop over cellid

	if (!payload->endcapItems().size()) return false;
	// looping over the EE channels
	for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	  for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	    for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
	      if(EEDetId::validDetId(ix, iy, iz)) {
		EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		uint32_t rawid = myEEId.rawId();
		EcalFloatCondObjectContainer::const_iterator value_ptr =  payload->find(rawid);
		if (value_ptr == payload->end()) continue; // cell absent from payload
		float weight = (float)(*value_ptr);
		if(iz == 1)
		  endc_p->Fill(ix, iy, weight);
		else
		  endc_m->Fill(ix, iy, weight);
	      }  // validDetId 
      }    // payload

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      TCanvas canvas("CC map","CC map", 1600, 450);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal IntercalibConstants, IOV %i", run));

      float xmi[3] = {0.0 , 0.24, 0.76};
      float xma[3] = {0.24, 0.76, 1.00};
      TPad** pad = new TPad*;
      for (int obj = 0; obj < 3; obj++) {
	pad[obj] = new TPad(Form("p_%i", obj),Form("p_%i", obj), xmi[obj], 0.0, xma[obj], 0.94);
	pad[obj]->Draw();
      }
      //      EcalDrawMaps ICMap;
      pad[0]->cd();
      //      ICMap.DrawEE(endc_m, 0., 2.);
      DrawEE(endc_m, 0., 2.5);
      pad[1]->cd();
      //      ICMap.DrawEB(barrel, 0., 2.);
      DrawEB(barrel, 0., 2.5);
      pad[2]->cd();
      //      ICMap.DrawEE(endc_p, 0., 2.);
      DrawEE(endc_p, 0., 2.5);
      canvas.SaveAs("ecalICmap.png");
      return true;
    }// fill method
  };

} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE( EcalIntercalibConstants ){
  PAYLOAD_INSPECTOR_CLASS( EcalIntercalibConstantsEBMap );
  PAYLOAD_INSPECTOR_CLASS( EcalIntercalibConstantsEEMap );
  PAYLOAD_INSPECTOR_CLASS( EcalIntercalibConstantsPlot );
}
