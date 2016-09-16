#ifndef TauDQMHistPlotter_h
#define TauDQMHistPlotter_h

/** \class TauDQMHistPlotter
 *  
 *  Class to plot histograms and create a postscript file
 *
 *  \author Christian Veelken, UC Davis
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMDefinitions.h"

#include <TH1.h>
#include <TLegend.h>
#include <TPaveText.h>

#include <string>
#include <vector>
#include <map>

class TauDQMHistPlotter : public edm::EDAnalyzer
{
  typedef std::vector<std::string> vstring;

  struct cfgEntryProcess 
  {
    cfgEntryProcess(const std::string&, const edm::ParameterSet&);
    void print() const;
    std::string name_;
    std::string dqmDirectory_;
    std::string legendEntry_;
    std::string legendEntryErrorBand_;
    std::string type_;
  };

  struct cfgEntryAxisX
  {
    explicit cfgEntryAxisX(const std::string&, const edm::ParameterSet&);
    void print() const;
    void applyTo(TH1*) const;
    std::string name_;
    double minX_;
    double maxX_;
    std::string xAxisTitle_;
    double xAxisTitleOffset_;
    double xAxisTitleSize_;
  };

  struct cfgEntryAxisY
  {
    explicit cfgEntryAxisY(const std::string&, const edm::ParameterSet&);
    void print() const;
    static void setNorm(double yAxisNorm) { yAxisNorm_ = yAxisNorm; }
    void applyTo(TH1*) const;
    std::string name_;
    double minY_linear_;
    double minY_log_;
    double maxY_linear_;
    double maxY_log_;
    std::string yScale_;
    std::string yAxisTitle_;
    double yAxisTitleOffset_;
    double yAxisTitleSize_;
    static double yAxisNorm_;
  };

  struct cfgEntryLegend
  {
    cfgEntryLegend(const std::string&, const edm::ParameterSet&);
    void print() const;
    void applyTo(TLegend*) const;
    std::string name_;
    double posX_;
    double posY_;
    double sizeX_;
    double sizeY_;
    std::string header_;
    std::string option_;
    int borderSize_;
    int fillColor_;
  };

  struct cfgEntryLabel
  {
    cfgEntryLabel(const std::string&, const edm::ParameterSet&);
    void print() const;
    void applyTo(TPaveText*) const;
    std::string name_;
    double posX_;
    double posY_;
    double sizeX_;
    double sizeY_;
    std::string option_;
    int borderSize_;
    int fillColor_;
    int textColor_;
    double textSize_;
    int textAlign_;
    double textAngle_;
    vstring text_;
  };

  struct cfgEntryDrawOption 
  {
    cfgEntryDrawOption(const std::string&, const edm::ParameterSet&);
    cfgEntryDrawOption(const std::string&, const cfgEntryDrawOption&);
    void print() const;
    void applyTo(TH1*) const;
    std::string name_;
    int markerColor_;
    double markerSize_;
    int markerStyle_;
    int lineColor_;
    int lineStyle_;
    int lineWidth_;
    int fillColor_;
    int fillStyle_;
    std::string drawOption_;
    std::string drawOptionLegend_;
  };

  struct plotDefEntry
  {
    plotDefEntry(const std::string&, const std::string&, const std::string&, const std::string&, const std::string&, bool);
    plotDefEntry(const plotDefEntry&);
    void print() const;
    std::string dqmMonitorElement_;
    std::string drawOptionEntry_;
    std::string legendEntry_;
    std::string legendEntryErrorBand_;
    std::string process_;
    bool doStack_;
    bool isErrorBand_;
  };
  
  typedef std::list<plotDefEntry> plotDefList;

  struct cfgEntryDrawJob 
  {
    cfgEntryDrawJob(const std::string&, const plotDefList&, const std::string&, const std::string&, const std::string&, const std::string&, const vstring&);
    void print() const;
    std::string name_;
    plotDefList plots_;
    std::string title_;
    std::string xAxis_;
    std::string yAxis_;
    std::string legend_;
    vstring labels_;
  };

 public:
  explicit TauDQMHistPlotter(const edm::ParameterSet&);
  virtual ~TauDQMHistPlotter();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(){}  
  virtual void endRun(const edm::Run& r, const edm::EventSetup& c);

private:
  std::map<std::string, cfgEntryProcess> processes_;
  std::map<std::string, cfgEntryAxisX> xAxes_;
  std::map<std::string, cfgEntryAxisY> yAxes_;
  std::map<std::string, cfgEntryLegend> legends_;
  std::map<std::string, cfgEntryLabel> labels_;
  std::map<std::string, cfgEntryDrawOption> drawOptionEntries_;
  std::list<cfgEntryDrawJob> drawJobs_;
  int canvasSizeX_;
  int canvasSizeY_;
  std::string outputFilePath_;
  std::string outputFileName_;
  std::string indOutputFileName_;
  int cfgError_;
  bool toFile_;
};

#endif


