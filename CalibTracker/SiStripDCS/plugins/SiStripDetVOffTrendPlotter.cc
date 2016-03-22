#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <sstream>

#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"

#include <TROOT.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TLegend.h>
#include <TGraph.h>
#include <TH1.h>


class SiStripDetVOffTrendPlotter : public edm::EDAnalyzer {
public:
  const std::vector<Color_t> PLOT_COLORS {kBlack, kRed, kBlue, kOrange, kMagenta};

  explicit SiStripDetVOffTrendPlotter(const edm::ParameterSet& iConfig );
  virtual ~SiStripDetVOffTrendPlotter();
  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void endJob();

private:
  std::string formatIOV(cond::Time_t iov, std::string format="%Y-%m-%d__%H_%M_%S");
  void prepGraph(TGraph *gr, TString name, TString title, Color_t color);

  cond::persistency::ConnectionPool m_connectionPool;
  std::string m_condDb;
  std::vector<std::string> m_plotTags;

  // Time interval (in hours) to go back for the plotting.
  int m_interval;
  // Or manually specify the start/end time. Format: "2002-01-20 23:59:59.000". Set m_interval to non-positive number to use this.
  std::string m_startTime;
  std::string m_endTime;
  // Set the plot format. Default: png.
  std::string m_plotFormat;
  // Specify output root file name. Leave empty if do not want to save plots in a root file.
  std::string m_outputFile;
  TFile *fout;
  edm::Service<SiStripDetInfoFileReader> detidReader;
};

SiStripDetVOffTrendPlotter::SiStripDetVOffTrendPlotter(const edm::ParameterSet& iConfig):
    m_connectionPool(),
    m_condDb( iConfig.getParameter< std::string >("conditionDatabase") ),
    m_plotTags( iConfig.getParameter< std::vector<std::string> >("plotTags") ),
    m_interval( iConfig.getParameter< int >("timeInterval") ),
    m_startTime( iConfig.getUntrackedParameter< std::string >("startTime", "") ),
    m_endTime( iConfig.getUntrackedParameter< std::string >("endTime", "") ),
    m_plotFormat( iConfig.getUntrackedParameter< std::string >("plotFormat", "png") ),
    m_outputFile( iConfig.getUntrackedParameter< std::string >("outputFile", "") ),
    fout(nullptr){
  m_connectionPool.setParameters( iConfig.getParameter<edm::ParameterSet>("DBParameters")  );
  m_connectionPool.configure();
  if (m_outputFile!="") fout = new TFile(m_outputFile.data(), "RECREATE");
}

SiStripDetVOffTrendPlotter::~SiStripDetVOffTrendPlotter() {
  if (fout) fout->Close();
}

void SiStripDetVOffTrendPlotter::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {

  // get total number of modules
  auto num_modules = detidReader->getAllDetIds().size();

  // get start and end time for DB query
  boost::posix_time::ptime p_start, p_end;
  if (m_interval > 0){
    // from m_interval hours ago to now
    p_end = boost::posix_time::second_clock::universal_time();
    p_start = p_end - boost::posix_time::hours(m_interval);
  }else {
    // use start and end time from config file
    p_start = boost::posix_time::time_from_string(m_startTime);
    p_end   = boost::posix_time::time_from_string(m_endTime);
  }
  cond::Time_t startIov = cond::time::from_boost(p_start);
  cond::Time_t endIov   = cond::time::from_boost(p_end);
  if (startIov > endIov)
    throw cms::Exception("endTime must be greater than startTime!");
  edm::LogInfo("SiStripDetVOffTrendPlotter") << "[SiStripDetVOffTrendPlotter::" << __func__ << "] "
      << "Set start IOV " << startIov << " (" << boost::posix_time::to_simple_string(p_start) << ")"
      << "\n ... Set end IOV " << endIov << " (" << boost::posix_time::to_simple_string(p_end) << ")" ;

  // open db session
  edm::LogInfo("SiStripDetVOffTrendPlotter") << "[SiStripDetVOffTrendPlotter::" << __func__ << "] "
      << "Query the condition database " << m_condDb;
  cond::persistency::Session condDbSession = m_connectionPool.createSession( m_condDb );
  condDbSession.transaction().start( true );

  // loop over all tags to plot
  std::vector<TGraph*> hvgraphs, lvgraphs;
  TLegend *leg_hv = new TLegend(0.75, 0.87, 0.99, 0.99);
  TLegend *leg_lv = new TLegend(0.75, 0.87, 0.99, 0.99);
  for (unsigned itag=0; itag<m_plotTags.size(); ++itag){
    auto tag = m_plotTags.at(itag);
    auto color = itag<PLOT_COLORS.size() ? PLOT_COLORS.at(itag) : kGreen+itag;

    std::vector<double> vTime;
    std::vector<double> vHVOffPercent, vLVOffPercent;

    // query the database
    edm::LogInfo("SiStripDetVOffTrendPlotter") << "[SiStripDetVOffTrendPlotter::" << __func__ << "] "
        << "Reading IOVs from tag " << tag;
    cond::persistency::IOVProxy iovProxy = condDbSession.readIov(tag, true); // load all?
    auto iiov = iovProxy.find(startIov);
    auto eiov = iovProxy.find(endIov);
    int niov = 0;
    while (iiov != iovProxy.end() && iiov != eiov){
      // convert cond::Time_t to seconds since epoch
      vTime.push_back(cond::time::unpack((*iiov).since).first);
      auto payload = condDbSession.fetchPayload<SiStripDetVOff>( (*iiov).payloadId );
      vHVOffPercent.push_back( 1.0*payload->getHVoffCounts()/num_modules );
      vLVOffPercent.push_back( 1.0*payload->getLVoffCounts()/num_modules );
      // debug
      std::cout << boost::posix_time::to_simple_string(cond::time::to_boost((*iiov).since))
          << " (" << (*iiov).since <<  ")"
          << ", # HV Off=" << std::setw(6) << payload->getHVoffCounts()
          << ", # LV Off=" << std::setw(6) << payload->getLVoffCounts() << std::endl;
      ++iiov;
      ++niov;
    }
    edm::LogInfo("SiStripDetVOffTrendPlotter") << "[SiStripDetVOffTrendPlotter::" << __func__ << "] "
        << "Read " << niov << " IOVs from tag " << tag << " in the specified interval.";

    TGraph *hv = new TGraph(vTime.size(), vTime.data(), vHVOffPercent.data());
    prepGraph(hv, TString("HVOff_")+tag, ";UTC;Fraction of HV off", color);
    leg_hv->AddEntry(hv, tag.data(), "LP");

    TGraph *lv = new TGraph(vTime.size(), vTime.data(), vLVOffPercent.data());
    prepGraph(lv, TString("LVOff_")+tag, ";UTC;Fraction of LV off", color);
    leg_lv->AddEntry(lv, tag.data(), "LP");

    hvgraphs.push_back(hv);
    lvgraphs.push_back(lv);
  }

  condDbSession.transaction().commit();

  // Make plots
  TCanvas c("c", "c", 900, 600);
  c.SetTopMargin(0.12);
  c.SetBottomMargin(0.08);
  c.SetGridx();
  c.SetGridy();
  for (const auto hv : hvgraphs){
    if (hv==hvgraphs.front())
      hv->Draw("ALP");
    else
      hv->Draw("ALPsame");
    if (fout){
      fout->cd();
      hv->Write();
    }
  }
  leg_hv->Draw();
  c.Print( ("HVOff_from_" + formatIOV(startIov) + "_to_" + formatIOV(endIov) + ".pdf").data() );

  c.Clear();
  for (const auto lv : lvgraphs){
    if (lv==hvgraphs.front())
      lv->Draw("ALP");
    else
      lv->Draw("ALPsame");
    if (fout){
      fout->cd();
      lv->Write();
    }
  }
  leg_lv->Draw();
  c.Print( ("LVOff_from_" + formatIOV(startIov) + "_to_" + formatIOV(endIov) + ".pdf").data() );

}

void SiStripDetVOffTrendPlotter::endJob() {
}

std::string SiStripDetVOffTrendPlotter::formatIOV(cond::Time_t iov, std::string format) {
  auto facet = new boost::posix_time::time_facet(format.c_str());
   std::ostringstream stream;
   stream.imbue(std::locale(stream.getloc(), facet));
   stream << cond::time::to_boost(iov);
   return stream.str();
}

void SiStripDetVOffTrendPlotter::prepGraph(TGraph* gr, TString name, TString title, Color_t color) {
  gr->SetName(name);
  gr->SetTitle(title);
  gr->SetLineColor(color);
  gr->SetMarkerStyle(20);
  gr->SetMarkerColor(color);
  gr->GetXaxis()->SetTimeDisplay(1);
  gr->GetXaxis()->SetTimeOffset(0, "gmt");
  gr->GetYaxis()->SetRangeUser(0, 1.01);
}


DEFINE_FWK_MODULE(SiStripDetVOffTrendPlotter);

