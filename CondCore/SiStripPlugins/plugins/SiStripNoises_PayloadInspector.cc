/*!
  \file SiStripNoises_PayloadInspector
  \Payload Inspector Plugin for SiStrip Noises
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/09/21 13:59:56 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h" 

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

// auxilliary functions
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CalibTracker/SiStripCommon/interface/StandaloneTrackerTopology.h" 

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <memory>
#include <sstream>
#include <iostream>

// include ROOT
#include "TH2F.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

namespace {

  /************************************************
    test class
  *************************************************/

  class SiStripNoisesTest : public cond::payloadInspector::Histogram1D<SiStripNoises> {
    
  public:
    SiStripNoisesTest() : cond::payloadInspector::Histogram1D<SiStripNoises>("SiStrip Noise test",
									     "SiStrip Noise test", 10,0.0,10.0),
			  m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())}
    {
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripNoises> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  fillWithValue(1.);
	 
	  std::stringstream ss;
	  ss << "Summary of strips noises:" << std::endl;

	  //payload->printDebug(ss);
	  payload->printSummary(ss,&m_trackerTopo);

	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  // for (const auto & d : detid) {
	  //   int nstrip=0;
	  //   SiStripNoises::Range range=payload->getRange(d);
	  //   for( int it=0; it < (range.second-range.first)*8/9; ++it ){
	  //     auto noise = payload->getNoise(it,range);
	  //     nstrip++;
	  //     ss << "DetId="<< d << " Strip=" << nstrip <<": "<< noise << std::endl;
	  //   } // end of loop on strips
	  // } // end of loop on detIds
	  
	  std::cout<<ss.str()<<std::endl;
 
	}// payload
      }// iovs
      return true;
    }// fill
  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************
    1d histogram of SiStripNoises of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripNoiseValue : public cond::payloadInspector::Histogram1D<SiStripNoises> {
    
  public:
    SiStripNoiseValue() : cond::payloadInspector::Histogram1D<SiStripNoises>("SiStrip Noise values",
									     "SiStrip Noise values", 100,0.0,10.0){
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripNoises> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  for (const auto & d : detid) {
	    SiStripNoises::Range range=payload->getRange(d);
	    for( int it=0; it < (range.second-range.first)*8/9; ++it ){
	      auto noise = payload->getNoise(it,range);
	      //to be used to fill the histogram
	      fillWithValue(noise);
	    }// loop over APVs
	  } // loop over detIds
	}// payload
      }// iovs
      return true;
    }// fill
  };
  
  /************************************************
    SiStrip Noise Tracker Map 
  *************************************************/

  template<SiStripPI::estimator est>  class SiStripNoiseTrackerMap : public cond::payloadInspector::PlotImage<SiStripNoises> {
    
  public:
    SiStripNoiseTrackerMap() : cond::payloadInspector::PlotImage<SiStripNoises> ( "Tracker Map of SiStripNoise "+estimatorType(est)+" per module" )
    {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripNoises> payload = fetchPayload( std::get<1>(iov) );

      std::string titleMap = "Tracker Map of Noise "+estimatorType(est)+" per module (payload : "+std::get<1>(iov)+")";
      
      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripNoises"));
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      // storage of info
      std::map<unsigned int,float> info_per_detid;
      
      SiStripNoises::RegistryIterator rit=payload->getRegistryVectorBegin(), erit=payload->getRegistryVectorEnd();
      uint16_t Nstrips;
      std::vector<float> vstripnoise;
      double mean,rms,min, max;
      for(;rit!=erit;++rit){
	Nstrips = (rit->iend-rit->ibegin)*8/9; //number of strips = number of chars * char size / strip noise size
	vstripnoise.resize(Nstrips);
	payload->allNoises(vstripnoise,make_pair(payload->getDataVectorBegin()+rit->ibegin,payload->getDataVectorBegin()+rit->iend));
	mean=0; rms=0; min=10000; max=0;  
	
	DetId detId(rit->detid);
	
	for(size_t i=0;i<Nstrips;++i){
	  mean+=vstripnoise[i];
	  rms+=vstripnoise[i]*vstripnoise[i];
	  if(vstripnoise[i]<min) min=vstripnoise[i];
	  if(vstripnoise[i]>max) max=vstripnoise[i];
	}
	
	mean/=Nstrips;
	if((rms/Nstrips-mean*mean)>0.){
	  rms = sqrt(rms/Nstrips-mean*mean);
	} else {
	  rms=0.;
	}       

	switch(est){
	case SiStripPI::min:
	  info_per_detid[rit->detid]=min;
	  break;
	case SiStripPI::max:
	  info_per_detid[rit->detid]=max;
	  break;
	case SiStripPI::mean:
	  info_per_detid[rit->detid]=mean;
	  break;
	case SiStripPI::rms:
	  info_per_detid[rit->detid]=rms;
	  break;
	default:
	  edm::LogWarning("LogicError") << "Unknown estimator: " <<  est; 
	  break;
	}	
      }

      // loop on the map
      for (const auto &item : info_per_detid){
	tmap->fill(item.first,item.second);
      }

      auto range = SiStripPI::getTheRange(info_per_detid);
      
      //=========================
      
      std::string fileName(m_imageFileName);
      if(est==SiStripPI::rms && (range.first<0.)){
	tmap->save(true,0.,range.second,fileName);
      } else {
	tmap->save(true,range.first,range.second,fileName);
      }

      return true;
    }
  };

  typedef SiStripNoiseTrackerMap<SiStripPI::min>  SiStripNoiseMin_TrackerMap;
  typedef SiStripNoiseTrackerMap<SiStripPI::max>  SiStripNoiseMax_TrackerMap;
  typedef SiStripNoiseTrackerMap<SiStripPI::mean> SiStripNoiseMean_TrackerMap;
  typedef SiStripNoiseTrackerMap<SiStripPI::rms>  SiStripNoiseRMS_TrackerMap;

  /************************************************
  SiStrip Noise Tracker Summaries 
  *************************************************/
  
   template<SiStripPI::estimator est> class SiStripNoiseByPartition : public cond::payloadInspector::PlotImage<SiStripNoises> {
  public:
     SiStripNoiseByPartition() : cond::payloadInspector::PlotImage<SiStripNoises>( "SiStrip Noise "+estimatorType(est)+" by Partition" ),
      m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())}
    {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripNoises> payload = fetchPayload( std::get<1>(iov) );
      
      SiStripDetSummary summaryNoise{&m_trackerTopo};

      SiStripNoises::RegistryIterator rit=payload->getRegistryVectorBegin(), erit=payload->getRegistryVectorEnd();
      uint16_t Nstrips;
      std::vector<float> vstripnoise;
      double mean,rms,min, max;
      for(;rit!=erit;++rit){
	Nstrips = (rit->iend-rit->ibegin)*8/9; //number of strips = number of chars * char size / strip noise size
	vstripnoise.resize(Nstrips);
	payload->allNoises(vstripnoise,make_pair(payload->getDataVectorBegin()+rit->ibegin,payload->getDataVectorBegin()+rit->iend));
	
	mean=0; rms=0; min=10000; max=0;  
	
	DetId detId(rit->detid);
	
	for(size_t i=0;i<Nstrips;++i){
	  mean+=vstripnoise[i];
	  rms+=vstripnoise[i]*vstripnoise[i];
	  if(vstripnoise[i]<min) min=vstripnoise[i];
	  if(vstripnoise[i]>max) max=vstripnoise[i];
	}
	
	mean/=Nstrips;
	if((rms/Nstrips-mean*mean)>0.){
	  rms = sqrt(rms/Nstrips-mean*mean);
	} else {
	  rms=0.;
	}       

	switch(est){
	case SiStripPI::min:
	  summaryNoise.add(detId,min);
	  break;
	case SiStripPI::max:
	  summaryNoise.add(detId,max);
	  break;
	case SiStripPI::mean:
	  summaryNoise.add(detId,mean);
	  break;
	case SiStripPI::rms:
	  summaryNoise.add(detId,rms);
	  break;
	default:
	  edm::LogWarning("LogicError") << "Unknown estimator: " <<  est; 
	  break;
	}

      }

      std::map<unsigned int, SiStripDetSummary::Values> map = summaryNoise.getCounts();
      //=========================
      
      TCanvas canvas("Partion summary","partition summary",1200,1000); 
      canvas.cd();
      auto h1 = std::unique_ptr<TH1F>(new TH1F("byPartition",Form("Average by partition of %s SiStrip Noise per module;;average SiStrip Noise %s [ADC counts]",estimatorType(est).c_str(),estimatorType(est).c_str()),map.size(),0.,map.size()));
      h1->SetStats(false);
      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.17);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::vector<int> boundaries;
      unsigned int iBin=0;

      std::string detector;
      std::string currentDetector;

      for (const auto &element : map){
	iBin++;
	int count   = element.second.count;
	double mean = (element.second.mean)/count;
	double rms  = (element.second.rms)/count - mean*mean;

	if(rms <= 0)
	  rms = 0;
	else
	  rms = sqrt(rms);

	if(currentDetector.empty()) currentDetector="TIB";
	
	switch ((element.first)/1000) 
	  {
	  case 1:
	    detector = "TIB";
	    break;
	  case 2:
	    detector = "TOB";
	    break;
	  case 3:
	    detector = "TEC";
	    break;
	  case 4:
	    detector = "TID";
	    break;
	  }

	h1->SetBinContent(iBin,mean);
	h1->GetXaxis()->SetBinLabel(iBin,SiStripPI::regionType(element.first));
	h1->GetXaxis()->LabelsOption("v");
	
	if(detector!=currentDetector) {
	  boundaries.push_back(iBin);
	  currentDetector=detector;
	}
      }

      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1);
      h1->SetMaximum(h1->GetMaximum()*1.1);
      h1->Draw("HIST");
      h1->Draw("Psame");
	    
      canvas.Update();
      
      TLine l[boundaries.size()];
      unsigned int i=0;
      for (const auto & line : boundaries){
	l[i] = TLine(h1->GetBinLowEdge(line),canvas.GetUymin(),h1->GetBinLowEdge(line),canvas.GetUymax());
	l[i].SetLineWidth(1);
	l[i].SetLineStyle(9);
	l[i].SetLineColor(2);
	l[i].Draw("same");
	i++;
      }
      
      TLegend legend = TLegend(0.52,0.82,0.95,0.9);
      legend.SetHeader((std::get<1>(iov)).c_str(),"C"); // option "C" allows to center the header
      legend.AddEntry(h1.get(),("IOV: "+std::to_string(std::get<0>(iov))).c_str(),"PL");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  private:
    TrackerTopology m_trackerTopo;
  };

  typedef SiStripNoiseByPartition<SiStripPI::mean> SiStripNoiseMeanByPartition;
  typedef SiStripNoiseByPartition<SiStripPI::min>  SiStripNoiseMinByPartition;
  typedef SiStripNoiseByPartition<SiStripPI::max>  SiStripNoiseMaxByPartition;
  typedef SiStripNoiseByPartition<SiStripPI::rms>  SiStripNoiseRMSByPartition;

  /************************************************
    Noise linearity
  *************************************************/
  class SiStripNoiseLinearity : public cond::payloadInspector::PlotImage<SiStripNoises> {
  public:
    SiStripNoiseLinearity() : cond::payloadInspector::PlotImage<SiStripNoises>( "Linearity of Strip Noise as a fuction of strip length" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripNoises> payload = fetchPayload( std::get<1>(iov) );

      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);
      
      std::map<float,std::tuple<int,float,float> > noisePerStripLength;

      for (const auto & d : detid) {
	SiStripNoises::Range range=payload->getRange(d);
	for( int it=0; it < (range.second-range.first)*8/9; ++it ){
	  auto noise = payload->getNoise(it,range);
	  //to be used to fill the histogram
	  float stripL = reader->getNumberOfApvsAndStripLength(d).second;
	  std::get<0>(noisePerStripLength[stripL])+=1;
	  std::get<1>(noisePerStripLength[stripL])+=noise;
	  std::get<2>(noisePerStripLength[stripL])+=(noise*noise);
	}// loop over strips
      } // loop over detIds
      
      TCanvas canvas("Noise linearity","noise linearity",1200,1000); 
      canvas.cd();

      std::vector<float> x;  x.reserve(noisePerStripLength.size());
      std::vector<float> y;  y.reserve(noisePerStripLength.size());
      std::vector<float> ex; ex.reserve(noisePerStripLength.size());
      std::vector<float> ey; ey.reserve(noisePerStripLength.size());

      for (const auto &element : noisePerStripLength ){
	x.push_back(element.first);
	ex.push_back(0.);
	float sum     = std::get<1>(element.second);
	float sum2    = std::get<2>(element.second);
	float nstrips = std::get<0>(element.second);
	float mean    = sum/nstrips;
	float rms     = (sum2/nstrips-mean*mean)>0. ? sqrt(sum2/nstrips-mean*mean) : 0.;
	y.push_back(mean);
	ey.push_back(rms);
	//std::cout<<" strip lenght: " << element.first << " avg noise=" << mean <<" +/-" << rms << std::endl;
      }

      auto graph = std::unique_ptr<TGraphErrors>(new TGraphErrors(noisePerStripLength.size(),&x[0], &y[0],&ex[0],&ey[0]));
      graph->SetTitle("SiStrip Noise Linearity");
      graph->GetXaxis()->SetTitle("Strip length [cm]");
      graph->GetYaxis()->SetTitle("Average Strip Noise [ADC counts]");
      graph->SetMarkerColor(kBlue);
      graph->SetMarkerStyle(20);
      graph->SetMarkerSize(1.5);
      canvas.SetBottomMargin(0.13);
      canvas.SetLeftMargin(0.17);
      canvas.SetTopMargin(0.08);
      canvas.SetRightMargin(0.05);
      canvas.Modified();
      canvas.cd();

      graph->GetXaxis()->CenterTitle(true);
      graph->GetYaxis()->CenterTitle(true);
      graph->GetXaxis()->SetTitleFont(42); 
      graph->GetYaxis()->SetTitleFont(42);  
      graph->GetXaxis()->SetTitleSize(0.05);
      graph->GetYaxis()->SetTitleSize(0.05);
      graph->GetXaxis()->SetTitleOffset(1.1);
      graph->GetYaxis()->SetTitleOffset(1.3);
      graph->GetXaxis()->SetLabelFont(42);
      graph->GetYaxis()->SetLabelFont(42);
      graph->GetYaxis()->SetLabelSize(.05);
      graph->GetXaxis()->SetLabelSize(.05);
      
      graph->Draw("AP");
      graph->Fit("pol1");
      //Access the fit resuts
      TF1 *f1 = graph->GetFunction("pol1");
      f1->SetLineWidth(2);
      f1->SetLineColor(kBlue);
      f1->Draw("same");

      auto fits = std::unique_ptr<TPaveText>(new TPaveText(0.2,0.72,0.6,0.9,"NDC"));
      char buffer[255];
      sprintf(buffer,"fit function: p_{0} + p_{1} * l_{strip}");
      fits->AddText(buffer);
      sprintf(buffer,"p_{0} : %5.2f [ADC counts]",f1->GetParameter(0));
      fits->AddText(buffer);
      sprintf(buffer,"p_{1} : %5.2f [ADC counts/cm]",f1->GetParameter(1));
      fits->AddText(buffer);
      sprintf(buffer,"#chi^{2}/ndf = %5.2f / %i ",f1->GetChisquare(),f1->GetNDF());
      fits->AddText(buffer);
      fits->SetTextFont(42);
      fits->SetTextColor(kBlue);
      fits->SetFillColor(0);
      fits->SetTextSize(0.03);
      fits->SetBorderSize(1);
      fits->SetLineColor(kBlue);
      fits->SetMargin(0.05);
      fits->SetTextAlign(12);
      fits->Draw();


      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      delete f1;
      delete reader;
      return true;
    }
  };

  /************************************************
   template Noise history per subdetector
  *************************************************/

  template <StripSubdetector::SubDetector sub> class NoiseHistory : public cond::payloadInspector::HistoryPlot<SiStripNoises,std::pair<double,double> > {
  public:
    NoiseHistory(): cond::payloadInspector::HistoryPlot<SiStripNoises,std::pair<double,double> >( "Average "+SiStripPI::getStringFromSubdet(sub)+" noise vs run number", "average "+SiStripPI::getStringFromSubdet(sub)+" Noise"){
    }
    
    std::pair<double,double> getFromPayload( SiStripNoises& payload ) override{
      
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);
      
      int nStrips=0;
      float sum=0., sum2=0.;
      for (const auto & d : detid) {
	int subid = DetId(d).subdetId();
	if(subid!=sub) continue;
	SiStripNoises::Range range=payload.getRange(d);
	for( int it=0; it < (range.second-range.first)*8/9; ++it ){
	  nStrips++;
	  auto noise = payload.getNoise(it,range);
	  sum+=noise;
	  sum2+=(noise*noise);
	} // loop on strips
      } // loop on detIds

      float mean = sum/nStrips;
      float rms  = (sum2/nStrips-mean*mean)>0. ? sqrt(sum2/nStrips-mean*mean) : 0.; 

      return std::make_pair(mean,rms);

    } // close getFromPayload
  };

  typedef NoiseHistory<StripSubdetector::TIB> TIBNoiseHistory;
  typedef NoiseHistory<StripSubdetector::TOB> TOBNoiseHistory;
  typedef NoiseHistory<StripSubdetector::TID> TIDNoiseHistory;
  typedef NoiseHistory<StripSubdetector::TEC> TECNoiseHistory;


  /************************************************
   template Noise run history  per subdetector
  *************************************************/

  template <StripSubdetector::SubDetector sub> class NoiseRunHistory : public cond::payloadInspector::RunHistoryPlot<SiStripNoises,std::pair<double,double> > {
  public:
    NoiseRunHistory(): cond::payloadInspector::RunHistoryPlot<SiStripNoises,std::pair<double,double> >( "Average "+SiStripPI::getStringFromSubdet(sub)+" noise vs run number", "average "+SiStripPI::getStringFromSubdet(sub)+" Noise"){
    }
    
    std::pair<double,double> getFromPayload( SiStripNoises& payload ) override{
      
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);
      
      int nStrips=0;
      float sum=0., sum2=0.;
      for (const auto & d : detid) {
	int subid = DetId(d).subdetId();
	if(subid!=sub) continue;
	SiStripNoises::Range range=payload.getRange(d);
	for( int it=0; it < (range.second-range.first)*8/9; ++it ){
	  nStrips++;
	  auto noise = payload.getNoise(it,range);
	  sum+=noise;
	  sum2+=(noise*noise);
	} // loop on strips
      } // loop on detIds

      float mean = sum/nStrips;
      float rms  = (sum2/nStrips-mean*mean)>0. ? sqrt(sum2/nStrips-mean*mean) : 0.; 

      return std::make_pair(mean,rms);

    } // close getFromPayload
  };

  typedef NoiseRunHistory<StripSubdetector::TIB> TIBNoiseRunHistory;
  typedef NoiseRunHistory<StripSubdetector::TOB> TOBNoiseRunHistory;
  typedef NoiseRunHistory<StripSubdetector::TID> TIDNoiseRunHistory;
  typedef NoiseRunHistory<StripSubdetector::TEC> TECNoiseRunHistory;

  /************************************************
   template Noise Time history per subdetector
  *************************************************/

  template <StripSubdetector::SubDetector sub> class NoiseTimeHistory : public cond::payloadInspector::TimeHistoryPlot<SiStripNoises,std::pair<double,double> > {
  public:
    NoiseTimeHistory(): cond::payloadInspector::TimeHistoryPlot<SiStripNoises,std::pair<double,double> >( "Average "+SiStripPI::getStringFromSubdet(sub)+" noise vs run number", "average "+SiStripPI::getStringFromSubdet(sub)+" Noise"){
    }
    
    std::pair<double,double> getFromPayload( SiStripNoises& payload ) override{
      
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);
      
      int nStrips=0;
      float sum=0., sum2=0.;
      for (const auto & d : detid) {
	int subid = DetId(d).subdetId();
	if(subid!=sub) continue;
	SiStripNoises::Range range=payload.getRange(d);
	for( int it=0; it < (range.second-range.first)*8/9; ++it ){
	  nStrips++;
	  auto noise = payload.getNoise(it,range);
	  sum+=noise;
	  sum2+=(noise*noise);
	} // loop on strips
      } // loop on detIds

      float mean = sum/nStrips;
      float rms  = (sum2/nStrips-mean*mean)>0. ? sqrt(sum2/nStrips-mean*mean) : 0.; 

      return std::make_pair(mean,rms);

    } // close getFromPayload
  };

  typedef NoiseTimeHistory<StripSubdetector::TIB> TIBNoiseTimeHistory;
  typedef NoiseTimeHistory<StripSubdetector::TOB> TOBNoiseTimeHistory;
  typedef NoiseTimeHistory<StripSubdetector::TID> TIDNoiseTimeHistory;
  typedef NoiseTimeHistory<StripSubdetector::TEC> TECNoiseTimeHistory;

} // close namespace

PAYLOAD_INSPECTOR_MODULE(SiStripNoises){
  PAYLOAD_INSPECTOR_CLASS(SiStripNoisesTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseValue);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseMin_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseMax_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseMean_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseRMS_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseMeanByPartition);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseMinByPartition);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseMaxByPartition);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseRMSByPartition);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseLinearity);
  PAYLOAD_INSPECTOR_CLASS(TIBNoiseHistory);
  PAYLOAD_INSPECTOR_CLASS(TOBNoiseHistory);
  PAYLOAD_INSPECTOR_CLASS(TIDNoiseHistory);
  PAYLOAD_INSPECTOR_CLASS(TECNoiseHistory);
  PAYLOAD_INSPECTOR_CLASS(TIBNoiseRunHistory);
  PAYLOAD_INSPECTOR_CLASS(TOBNoiseRunHistory);
  PAYLOAD_INSPECTOR_CLASS(TIDNoiseRunHistory);
  PAYLOAD_INSPECTOR_CLASS(TECNoiseRunHistory);
  PAYLOAD_INSPECTOR_CLASS(TIBNoiseTimeHistory);
  PAYLOAD_INSPECTOR_CLASS(TOBNoiseTimeHistory);
  PAYLOAD_INSPECTOR_CLASS(TIDNoiseTimeHistory);
  PAYLOAD_INSPECTOR_CLASS(TECNoiseTimeHistory);
}
